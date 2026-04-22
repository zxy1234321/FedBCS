import os
import torch
from argparse import Namespace
from fdmodels.utils.federated_model import FederatedModel
from dataset.utils.federated_dataset import FederatedDataset
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from utils.conf import get_device
from utils.util import generate_online_clients_sequence
from monai.networks import one_hot
import heapq
from monai.transforms import Compose, Activations, AsDiscrete, EnsureType

post_pred = Compose(
    [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
)
from monai.data import decollate_batch

post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])
TOP_K = 10


def global_evaluate_pl(model: FederatedModel, test_dl: DataLoader, setting: str, name: str, device=None) -> Tuple[list, list]:
    accs = []
    hd95s = []
    assds = []
    net = model.global_net
    status = net.training
    net.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(percentile=95, reduction="mean_batch")
    assd_metric = SurfaceDistanceMetric(include_background=False, reduction="mean_batch")

    for j, dl in enumerate(test_dl):
        metric_list = 0.0
        for batch_idx, sampled_batch in enumerate(dl):
            with torch.no_grad():
                img, label = sampled_batch["image"], sampled_batch["label"]
                if label.ndim == 3:
                    label = label.unsqueeze(1)
                label = label.to(device)
                val_outputs = net(img.to(device))
                val_labels_onehot = one_hot(label, num_classes=model.N_CLASS)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels_onehot = [post_gt(i) for i in decollate_batch(val_labels_onehot)]
                val_outputs = [v.to(device) for v in val_outputs]
                val_labels_onehot = [v.to(device) for v in val_labels_onehot]
                dice_metric(val_outputs, val_labels_onehot)
                hd95_metric(val_outputs, val_labels_onehot)
                assd_metric(val_outputs, val_labels_onehot)

        dice = dice_metric.aggregate(reduction='mean_batch')
        hd95 = hd95_metric.aggregate()
        assd = assd_metric.aggregate()
        dice_metric.reset()
        hd95_metric.reset()
        assd_metric.reset()

        accs.append(dice)
        hd95s.append(hd95)
        assds.append(assd)

    net.train(status)
    return accs, hd95s, assds


def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace, log_filepath) -> None:
    model.N_CLASS = private_dataset.N_CLASS
    selected_domain_list = private_dataset.DOMAINS_LIST
    writer = model.writer
    domains_len = len(selected_domain_list)
    best_model_records = []

    print(f'domains len: {domains_len}')
    print(f'selected_domain_list: {selected_domain_list}')
    pri_train_loaders, test_loaders = private_dataset.get_data_loaders()
    model.trainloaders = pri_train_loaders
    if hasattr(model, 'ini'):
        model.ini()
    model.online_clients_sequence = generate_online_clients_sequence(args.communication_epoch, args.parti_num, args.online_ratio)
    best_acc = 0
    best_dice = 0

    Epoch = args.communication_epoch
    print('Epoch:', Epoch)
    device = get_device(device_id=args.device_id)

    if args.txt_log:
        txt_path = os.path.join(log_filepath, 'log.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Experiment Parameters:\n")
            f.write("=" * 80 + "\n")
            for key, value in sorted(vars(args).items()):
                f.write(f"{key}: {value}\n")
            f.write("=" * 80 + "\n\n")

    for epoch_index in range(Epoch):

        model.epoch_index = epoch_index
        if hasattr(model, 'loc_update'):
            epoch_loc_loss_dict = model.loc_update(pri_train_loaders)
        accs, hd95s, assds = global_evaluate_pl(model, test_loaders, private_dataset.SETTING, private_dataset.NAME, device=device)
        accs = [acc.cpu() if isinstance(acc, torch.Tensor) else acc for acc in accs]
        hd95s = [hd95.cpu() if isinstance(hd95, torch.Tensor) else hd95 for hd95 in hd95s]
        assds = [assd.cpu() if isinstance(assd, torch.Tensor) else assd for assd in assds]

        mean_acc = round(float(torch.mean(torch.stack(accs))), 5)
        mean_hd95 = round(float(torch.mean(torch.stack(hd95s))), 5)
        mean_assd = round(float(torch.mean(torch.stack(assds))), 5)

        writer.add_scalar('com/mean_acc', mean_acc, epoch_index)
        writer.add_scalar('com/hd95', mean_hd95, epoch_index)
        writer.add_scalar('com/assd', mean_assd, epoch_index)

        mean_acc_per_class = [round(float(np.mean([acc[i] for acc in accs])), 5) for i in range(len(accs[0]))]
        mean_assd_per_class = [round(float(np.mean([assd[i] for assd in assds])), 5) for i in range(len(assds[0]))]
        mean_hd95_per_class = [round(float(np.mean([hd95[i] for hd95 in hd95s])), 5) for i in range(len(hd95s[0]))]
        writer.add_scalar('com/mean_acc_per_class_1', mean_acc_per_class[1], epoch_index)

        current_dice = mean_acc_per_class[1]
        if current_dice > best_dice:
            best_dice = current_dice

        if model.N_CLASS > 2:
            writer.add_scalar('com/mean_hd95_per_class_1', mean_hd95_per_class[1], epoch_index)
            writer.add_scalar('com/mean_hd95_per_class_0', mean_hd95_per_class[0], epoch_index)
            writer.add_scalar('com/mean_assd_per_class_1', mean_assd_per_class[1], epoch_index)
            writer.add_scalar('com/mean_assd_per_class_0', mean_assd_per_class[0], epoch_index)

        class_1_acc = mean_acc_per_class[1]
        global_model_filename = f'global_{epoch_index}_{class_1_acc:.5f}.ckpt'
        global_model_path = os.path.join(log_filepath, global_model_filename)
        torch.save(model.global_net.state_dict(), global_model_path)

        new_record = (class_1_acc, epoch_index, global_model_path)

        if len(best_model_records) < TOP_K:
            heapq.heappush(best_model_records, new_record)
        else:
            if class_1_acc > best_model_records[0][0]:
                popped_record = heapq.heappushpop(best_model_records, new_record)
                if os.path.exists(popped_record[2]):
                    os.remove(popped_record[2])
            else:
                if os.path.exists(global_model_path):
                    os.remove(global_model_path)

        print(f'The {epoch_index} Communication Accuracy: {mean_acc}, HD95: {mean_hd95}, ASSD: {mean_assd}, Method: {model.args.model}')
        print(f'Dice: {current_dice:.5f}, Best Dice: {best_dice:.5f}')
        print(f'Per class Accuracies: {mean_acc_per_class}')
        print(f'Per class HD95s: {mean_hd95_per_class}')
        print(f'Per class ASSDs: {mean_assd_per_class}')
        print(f'Accuracies: {accs}')
        print(f'HD95s: {hd95s}')
        print(f'ASSDs: {assds}')

        if args.txt_log:
            txt_path = os.path.join(log_filepath, 'log.txt')
            with open(txt_path, 'a') as f:
                f.write(f'selected_domain_list: {selected_domain_list}\n')
                f.write(f'The {epoch_index} Communication Accuracy: {mean_acc}, HD95: {mean_hd95}, ASSD: {mean_assd}, Method: {model.args.model}\n')
                f.write(f'Dice: {current_dice:.5f}, Best Dice: {best_dice:.5f}\n')
                f.write(f'Per class Accuracies: {mean_acc_per_class}\n')
                f.write(f'Per class HD95s: {mean_hd95_per_class}\n')
                f.write(f'Per class ASSDs: {mean_assd_per_class}\n')
                f.write(f'Accuracies: {accs}\n')
                f.write(f'HD95s: {hd95s}\n')
                f.write(f'ASSDs: {assds}\n')

    print(f"\nTop-{TOP_K} Models Summary (Based on Class 1 Accuracy):")
    for i, record in enumerate(sorted(best_model_records, key=lambda x: -x[0])):
        print(f"Rank {i+1}: Class 1 Acc {record[0]:.5f} at Epoch {record[1]} | Path: {record[2]}")

    if args.txt_log:
        txt_path = os.path.join(log_filepath, 'log.txt')
        with open(txt_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Top-{TOP_K} Models Summary (Based on Class 1 Accuracy):\n")
            f.write("=" * 80 + "\n")
            for i, record in enumerate(sorted(best_model_records, key=lambda x: -x[0])):
                f.write(f"Rank {i+1}: Class 1 Acc {record[0]:.5f} at Epoch {record[1]} | Path: {record[2]}\n")
            f.write("=" * 80 + "\n")

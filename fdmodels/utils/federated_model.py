import copy
import os
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from utils.conf import checkpoint_path
from utils.util import create_if_not_exists
from utils.logger import Logger
from config_examples.training_example import config
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class FederatedModel(nn.Module):
    NAME = None
    N_CLASS = None

    def __init__(self, nets_list: list,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(FederatedModel, self).__init__()
        self.nets_list = nets_list
        self.args = args
        self.transform = transform

        self.random_state = np.random.RandomState()
        self.online_num = self.args.parti_num

        self.global_net = None
        self.device = get_device(device_id=self.args.device_id)
        self.config = config
        self.instantiate_logger()
        self.store_config()
        self.local_epoch = args.local_epoch
        self.local_lr = args.local_lr
        if args.dataset == 'mri' or args.optim == 'adam':
            self.optimizers = [optim.Adam(nets_list[i].parameters(), lr=self.local_lr, weight_decay=0.0001) for i in
                       range(self.args.parti_num)]
            self.logger.info("Using Adam optimizer")
        elif args.optim == 'sgd':
            self.optimizers = [optim.SGD(nets_list[i].parameters(), lr=self.local_lr, weight_decay=0.0001) for i in
                       range(self.args.parti_num)]
            self.logger.info("Using SGD optimizer")
        else:
            raise ValueError(f"Unsupported optimizer type: {args.optim}")
        self.trainloaders = None
        self.testlodaers = None

        self.epoch_index = 0

        self.net_to_device()

        self.checkpoints = [[] for _ in range(self.args.parti_num)]
        self.top_k = 3
        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.global_steps = {i: 0 for i in range(len(nets_list))}
        self.global_epochs = {i: 0 for i in range(len(nets_list))}
        self.online_clients_sequence = None

    def net_to_device(self):
        for net in self.nets_list:
            net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        pass

    def instantiate_logger(self):
        logger = Logger(
            level='DEBUG',
            log_dir=Path(self.args.log_dir).resolve(),
            comment="logs",
            use_timestamp=False,
        )
        self.logger = logger.create_logger()
        return self.logger

    def store_config(self):
        with open(Path(self.args.log_dir).resolve() / "config.txt", "w") as f:
            f.write(str(self.config))
            f.write("\n")
            f.write(str(self.args))
        self.logger.info("Stored config file in log directory: {}".format(self.args.log_dir))

    def col_update(self, communication_idx, publoader):
        pass

    def loc_update(self, priloader_list):
        pass

    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                self.nets_list[j].load_state_dict(torch.load(save_path, self.device))
        else:
            pass

    def copy_nets2_prevnets(self):
        nets_list = self.nets_list
        prev_nets_list = self.prev_nets_list
        for net_id, net in enumerate(nets_list):
            net_para = net.state_dict()
            prev_net = prev_nets_list[net_id]
            prev_net.load_state_dict(net_para)

    def aggregate_nets(self, freq=None):
        with torch.no_grad():
            global_net = self.global_net
            nets_list = self.nets_list

            online_clients = self.online_clients
            global_w = self.global_net.state_dict()

            if self.args.averaing == 'weight' and freq == None:
                online_clients_dl = [self.trainloaders[online_clients_index] for online_clients_index in online_clients]
                online_clients_len = [len(dl.dataset) for dl in online_clients_dl]
                online_clients_all = np.sum(online_clients_len)
                freq = online_clients_len / online_clients_all
            elif freq == None:
                parti_num = len(online_clients)
                freq = [1 / parti_num for _ in range(parti_num)]

            first = True
            for index, net_id in enumerate(online_clients):
                net = nets_list[net_id]
                net_para = net.state_dict()
                if first:
                    first = False
                    for key in net_para:
                        global_w[key] = net_para[key] * freq[index]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * freq[index]

            global_net.load_state_dict(global_w)

            for _, net in enumerate(nets_list):
                net.load_state_dict(global_net.state_dict())

    def _save_checkpoint(self, idx, global_epoch, model, optimizer, val_metrics):
        arch = type(model).__name__
        binary_dice = val_metrics.get("binary_dice", 0)
        binary_dice = round(binary_dice, 4)

        checkpoint = {
            "global_epoch": global_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "binary_dice": binary_dice,
            "logdir": self.args.log_dir,
            "arch": arch,
        }
        checkpoint_dir = os.path.join(self.args.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"client_{idx}_dice_{binary_dice}_epoch_{global_epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints[idx].append((binary_dice, checkpoint_path))

        self.checkpoints[idx] = sorted(self.checkpoints[idx], key=lambda x: x[0], reverse=True)[:self.top_k]

        for _, path in self.checkpoints[idx][self.top_k:]:
            if os.path.exists(path):
                os.remove(path)

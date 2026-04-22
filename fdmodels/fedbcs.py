import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from fdmodels.utils.federated_model import FederatedModel
from torch.utils.tensorboard import SummaryWriter
import torch
from utils.finch import FINCH
import numpy as np
import torch.nn.functional as F
from utils.loss import KDloss, JointLoss


def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos


class FedBCS(FederatedModel):
    NAME = 'fedbcs'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedBCS, self).__init__(nets_list, args, transform)
        self.global_protos = {}
        self.local_protos = {}
        self.infoNCET = args.infoNCET
        self.alp = args.alp
        self.selected_layers = [int(x) for x in args.layer_config.split(',')]
        print(f"Layer config: {self.selected_layers}")

    def ini(self):
        self.global_net = self.nets_list[0]
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def proto_aggregation(self, local_protos_list):
        agg_protos_layer = {}
        for idx in self.online_clients:
            local_protos = local_protos_list[idx]
            for layer_index, layer_protos in local_protos.items():
                 if layer_index not in agg_protos_layer:
                    agg_protos_layer[layer_index] = {}
                 for label, proto_list in layer_protos.items():
                    if label in agg_protos_layer[layer_index]:
                        agg_protos_layer[layer_index][label].append(proto_list)
                    else:
                        agg_protos_layer[layer_index][label] = [proto_list]

        for layer_index, layer_protos in agg_protos_layer.items():
             for [label, proto_list] in layer_protos.items():
                 if len(proto_list) > 1:
                     proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
                     proto_list = np.array(proto_list)

                     c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                                 ensure_early_exit=False, verbose=True)

                     m, n = c.shape
                     class_cluster_list = []
                     for index in range(m):
                         class_cluster_list.append(c[index, -1])

                     class_cluster_array = np.array(class_cluster_list)
                     uniqure_cluster = np.unique(class_cluster_array).tolist()
                     agg_selected_proto = []

                     for _, cluster_index in enumerate(uniqure_cluster):
                         selected_array = np.where(class_cluster_array == cluster_index)
                         selected_proto_list = proto_list[selected_array]
                         proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                         agg_selected_proto.append(torch.tensor(proto))
                     agg_protos_layer[layer_index][label] = agg_selected_proto
                 else:
                     agg_protos_layer[layer_index][label] = [proto_list[0].data]

        return agg_protos_layer

    def compute_L_MP_sample(self, f_now, label, all_f, mean_f, all_global_protos_keys):
        f_pos = np.array(all_f)[all_global_protos_keys == label.item()][0].to(self.device)
        f_neg = torch.cat(list(np.array(all_f)[all_global_protos_keys != label.item()])).to(self.device)
        L_contra = self.compute_L_contra(f_now, f_pos, f_neg)

        mean_f_pos = np.array(mean_f)[all_global_protos_keys == label.item()][0].to(self.device)
        mean_f_pos = mean_f_pos.view(1, -1)

        loss_mse = nn.MSELoss()
        L_consis = loss_mse(f_now, mean_f_pos)

        L_MP_cls = L_contra + L_consis
        return L_MP_cls

    def compute_L_MP(self, f_now, labels, all_f, mean_f, all_global_protos_keys):
        assert labels.shape[0] == f_now.shape[0]

        unique_labels = labels.unique()
        total_loss = 0.0

        for cls in unique_labels:
            cls_indices = (labels == cls)
            f_now_cls = f_now[cls_indices]

            cls_item = cls.item()
            cls_indices_global = torch.tensor(all_global_protos_keys == cls_item).to(self.device)

            if cls_indices_global.any():
                f_pos = torch.cat([all_f[i] for i in torch.where(cls_indices_global)[0]]).to(self.device)
                f_neg = torch.cat([all_f[i] for i in torch.where(~cls_indices_global)[0]]).to(self.device)

                L_contra = self.compute_L_contra(f_now_cls, f_pos, f_neg)

                mean_f_pos = torch.stack([mean_f[i] for i in torch.where(cls_indices_global)[0]]).to(self.device)
                mean_f_pos = mean_f_pos.mean(dim=0, keepdim=True)

                loss_mse = nn.MSELoss()
                L_consis = loss_mse(f_now_cls, mean_f_pos.expand_as(f_now_cls))

                L_MP_cls = L_contra + L_consis
                total_loss += L_MP_cls
            else:
                continue

        final_loss = total_loss / unique_labels.numel()
        return final_loss

    def compute_L_contra(self, f_now, f_pos, f_neg):
        f_proto = torch.cat((f_pos, f_neg), dim=0)

        l = torch.cosine_similarity(f_now.unsqueeze(1), f_proto.unsqueeze(0), dim=2)
        l = l / self.infoNCET

        l = torch.clamp(l, max=20.0)

        exp_l = torch.exp(l)
        exp_l = torch.clamp(exp_l, max=1e6)

        pos_mask = torch.cat([torch.ones(f_pos.size(0)), torch.zeros(f_neg.size(0))]).to(self.device)
        pos_l = exp_l * pos_mask

        eps = 1e-9
        sum_pos_l = pos_l.sum(1) + eps
        sum_exp_l = exp_l.sum(1) + eps

        contra_loss = -torch.log(sum_pos_l / sum_exp_l)

        return contra_loss.mean()

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = total_clients[:self.online_num]
        self.online_clients = online_clients
        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i], self.optimizers[i])
        self.global_protos = self.proto_aggregation(self.local_protos)
        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader, optimizer):
        net = net.to(self.device)
        optimizer = optimizer
        criterion = JointLoss()
        criterion.to(self.device)

        iterator = tqdm(range(self.local_epoch))
        for iter in iterator:
            agg_protos_layer = {}
            for batch_idx, batch in enumerate(train_loader):
                images, labels = batch["image"], batch["label"]
                optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs, encoder, decoder, final = net(images, return_feature=True)
                decoder.append(final)

                encoder_list_updated = []
                if len(self.selected_layers) > 0 and max(self.selected_layers) < len(encoder):
                    selected_encoder_indices = [i for i in self.selected_layers if i < len(encoder)]

                    if len(selected_encoder_indices) > 1:
                        deepest_idx = max(selected_encoder_indices)
                        f_deepest = encoder[deepest_idx]

                        for idx in sorted(selected_encoder_indices):
                            if idx < deepest_idx:
                                f = encoder[idx]
                                f_deepest_upsampled = F.interpolate(f_deepest, size=f.shape[2:], mode='nearest')
                                f_concat = torch.cat((f, f_deepest_upsampled), dim=1)
                                if idx < len(net.encoder_fusions):
                                    f_fused = net.encoder_fusions[idx](f_concat)
                                    encoder_list_updated.append(f_fused)
                    elif len(selected_encoder_indices) == 1:
                        encoder_list_updated.append(encoder[selected_encoder_indices[0]])

                if len(encoder_list_updated) > 0:
                    _, _, H, W = encoder_list_updated[0].size()
                    encoder_list_updated = [F.interpolate(f, size=(H, W), mode='nearest') for f in encoder_list_updated]
                    encoder_f = torch.cat(encoder_list_updated, dim=1)
                else:
                    encoder_f = None
                encoder_list_updated.clear()

                decoder_list_updated = []
                if len(self.selected_layers) > 0:
                    selected_decoder_indices = [3 - i for i in self.selected_layers if i < 4]

                    if len(selected_decoder_indices) > 1:
                        deepest_idx = min(selected_decoder_indices)
                        f_deepest = decoder[deepest_idx]

                        for idx in sorted(selected_decoder_indices):
                            if idx > deepest_idx:
                                f = decoder[idx]
                                f_deepest_upsampled = F.interpolate(f_deepest, size=f.shape[2:], mode='nearest')
                                f_concat = torch.cat((f, f_deepest_upsampled), dim=1)
                                fusion_idx = idx - 1
                                if fusion_idx < len(net.decoder_fusions):
                                    f_fused = net.decoder_fusions[fusion_idx](f_concat)
                                    decoder_list_updated.append(f_fused)
                    elif len(selected_decoder_indices) == 1:
                        decoder_list_updated.append(decoder[selected_decoder_indices[0]])

                if len(decoder_list_updated) > 0:
                    _, _, H, W = decoder_list_updated[0].size()
                    decoder_list_updated = [F.interpolate(f, size=(H, W), mode='nearest') for f in decoder_list_updated]
                    decoder_f = torch.cat(decoder_list_updated, dim=1)
                else:
                    decoder_f = None
                decoder_list_updated.clear()

                f_list_updated = []
                if encoder_f is not None:
                    f_list_updated.append(encoder_f)
                if decoder_f is not None:
                    f_list_updated.append(decoder_f)

                lossCE = criterion(outputs, labels)

                loss_MP = 0.0
                for layer_index, f in enumerate(f_list_updated):
                    batch_size, channels, height, width = f.size()
                    f_flat = f.permute(0, 2, 3, 1).contiguous().view(-1, channels)

                    labels_downsampled = F.interpolate(labels.float(), size=(f.shape[2], f.shape[3]), mode='nearest').squeeze(1).long()
                    labels_flat = labels_downsampled.view(-1)

                    assert f_flat.size(0) == labels_flat.size(0)

                    if layer_index in self.global_protos:
                         all_global_protos_keys = np.array(list(self.global_protos[layer_index].keys()))
                         all_f = []
                         mean_f = []
                         for protos_key in all_global_protos_keys:
                             temp_f = self.global_protos[layer_index][protos_key]
                             temp_f = torch.cat(temp_f, dim=0).to(self.device)
                             all_f.append(temp_f.cpu())
                             mean_f.append(torch.mean(temp_f, dim=0).cpu())
                         all_f = [item.detach() for item in all_f]
                         mean_f = [item.detach() for item in mean_f]

                         valid_indices = labels_flat != 0
                         labels_valid = labels_flat[valid_indices]
                         f_valid = f_flat[valid_indices]
                         if labels_valid.numel() > 0:
                             loss_MP_layer = self.compute_L_MP(f_valid, labels_valid, all_f, mean_f, all_global_protos_keys)
                             loss_MP += loss_MP_layer

                    if iter == self.local_epoch - 1:
                        with torch.no_grad():
                            for cls in torch.unique(labels_flat):
                                idx = labels_flat == cls
                                features_cls = f_flat[idx]
                                if layer_index not in agg_protos_layer:
                                    agg_protos_layer[layer_index] = {}
                                if cls.item() in agg_protos_layer[layer_index]:
                                        agg_protos_layer[layer_index][cls.item()].append(features_cls.mean(dim=0))
                                else:
                                    agg_protos_layer[layer_index][cls.item()] = [features_cls.mean(dim=0)]
                if len(f_list_updated) > 0:
                    loss_MP /= len(f_list_updated)
                f_list_updated.clear()
                loss = lossCE + loss_MP
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,L_MP = %0.3f" % (index, lossCE, loss_MP)
                optimizer.step()
                self.writer.add_scalar(f'Client_{index}/Loss_CE', lossCE.item(), self.global_steps[index])
                self.writer.add_scalar(f'Client_{index}/Loss_MP', loss_MP, self.global_steps[index])
                self.writer.add_scalar(f'Client_{index}/Total_Loss', loss.item(), self.global_steps[index])

                self.global_steps[index] += 1
                del f_list_updated
                torch.cuda.empty_cache()

        agg_protos = {}
        for layer_index, layer_protos in agg_protos_layer.items():
                agg_protos[layer_index] = agg_func(layer_protos)
        self.local_protos[index] = agg_protos

import random
from torch import nn
import torch
import torch.nn.functional as F


class KDloss(nn.Module):

    def __init__(self, lambda_x=0.1):
        super(KDloss, self).__init__()
        self.lambda_x = lambda_x

    def inter_fd(self, f_s, f_t):
        s_C, t_C, s_H, t_H = f_s.shape[1], f_t.shape[1], f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))

        idx_s = random.sample(range(s_C), min(s_C, t_C))
        idx_t = random.sample(range(t_C), min(s_C, t_C))

        inter_fd_loss = F.mse_loss(f_s[:, idx_s, :, :], f_t[:, idx_t, :, :].detach())
        return inter_fd_loss

    def intra_fd(self, f_s):
        sorted_s, indices_s = torch.sort(F.normalize(f_s, p=2, dim=(2, 3)).mean([0, 2, 3]), dim=0, descending=True)
        f_s = torch.index_select(f_s, 1, indices_s)
        intra_fd_loss = F.mse_loss(f_s[:, 0:f_s.shape[1] // 2, :, :], f_s[:, f_s.shape[1] // 2: f_s.shape[1], :, :])
        return intra_fd_loss

    def forward(self, feature, feature_decoder, final_up):
        f1_0 = feature[0]
        f2_0 = feature[1]
        f3_0 = feature[2]
        f4_0 = feature[3]

        f1_d_0 = feature_decoder[0]
        f2_d_0 = feature_decoder[1]
        f3_d_0 = feature_decoder[2]

        final_layer = final_up

        loss = (self.intra_fd(f1_0) + self.intra_fd(f2_0) + self.intra_fd(f3_0) + self.intra_fd(f4_0)) / 4
        loss += (self.intra_fd(f1_d_0) + self.intra_fd(f2_d_0) + self.intra_fd(f3_d_0)) / 3

        loss += (self.inter_fd(f1_d_0, final_layer) + self.inter_fd(f2_d_0, final_layer) + self.inter_fd(f3_d_0, final_layer)
                   + self.inter_fd(f1_0, final_layer) + self.inter_fd(f2_0, final_layer) + self.inter_fd(f3_0, final_layer) + self.inter_fd(f4_0, final_layer)) / 7

        loss = loss * self.lambda_x
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        all_dice = 0
        gt = gt.squeeze(dim=1)
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):
            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred == i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt == i] = 1

            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)

            union = each_pred.view(batch_size, -1).sum(1) + each_gt.view(batch_size, -1).sum(1)
            dice = (2. * intersection) / (union + 1e-5)

            all_dice += torch.mean(dice)

        return all_dice * 1.0 / num_class

    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred, dim=1)

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]

        bg = torch.zeros_like(gt)
        bg[gt == 0] = 1
        label1 = torch.zeros_like(gt)
        label1[gt == 1] = 1
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1
        label = torch.cat([bg, label1, label2], dim=1)

        loss = 0
        smooth = 1e-5

        for i in range(num_class):
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...])
            z_sum = torch.sum(sigmoid_pred[:, i, ...])
            y_sum = torch.sum(label[:, i, ...])
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss


class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.dice = DiceLoss()

    def forward(self, pred, gt):
        ce = self.ce(pred, gt.squeeze(axis=1).long())
        ce = ce.mean()
        return (ce + self.dice(pred, gt)) / 2

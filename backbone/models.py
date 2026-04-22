import math
import sys
import os
from typing import List
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.dac import FSR
from torch.nn.functional import avg_pool2d


class UNet(nn.Module):
    def __init__(self, input_shape=None, in_channels=3, out_channels=2, init_features=32):
        super(UNet, self).__init__()

        self.entropy_hist = Entropy_Hist(0.1)
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        self.encoder_fusions = nn.ModuleList()
        self.decoder_fusions = nn.ModuleList()
        self.decoder_fusions.append(nn.Conv2d(192 * 2, 256, kernel_size=1))
        self.decoder_fusions.append(nn.Conv2d(160 * 2, 256, kernel_size=1))
        self.decoder_fusions.append(nn.Conv2d(144 * 2, 256, kernel_size=1))
        self.encoder_fusions.append(nn.Conv2d(288, 256, kernel_size=1))
        self.encoder_fusions.append(nn.Conv2d(320, 256, kernel_size=1))
        self.encoder_fusions.append(nn.Conv2d(384, 256, kernel_size=1))

    def forward(self, x, return_feature=False, return_entropy_hist=False):
        encoder_features = []
        decoder_features = []
        enc1 = self.encoder1(x)
        encoder_features.append(enc1)
        enc2 = self.encoder2(self.pool1(enc1))
        encoder_features.append(enc2)
        enc3 = self.encoder3(self.pool2(enc2))
        encoder_features.append(enc3)
        enc4 = self.encoder4(self.pool3(enc3))
        encoder_features.append(enc4)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        decoder_features.append(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        decoder_features.append(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        decoder_features.append(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        decoder_features.append(dec1)
        final = decoder_features[-1]
        decoder_features = decoder_features[:-1]
        dec1 = self.conv(dec1)
        if return_feature and return_entropy_hist:
            return dec1, encoder_features, decoder_features, final, self.entropy_hist(final)[1]
        elif return_feature:
            return dec1, encoder_features, decoder_features, final
        elif return_entropy_hist:
            return dec1, self.entropy_hist(final)[1]
        else:
            return dec1

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "_conv1", nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False)),
                    (name + "_bn1", nn.BatchNorm2d(num_features=features, affine=False, track_running_stats=False)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (name + "_conv2", nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)),
                    (name + "_bn2", nn.BatchNorm2d(num_features=features, affine=False, track_running_stats=False)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class UNet_FSR(nn.Module):
    def __init__(self, input_shape=None, in_channels=3, out_channels=2, init_features=32):
        super(UNet_FSR, self).__init__()

        self.amp_norm = None
        self.entropy_hist = Entropy_Hist(0.1)
        features = init_features
        self.encoder1 = UNet_FSR._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet_FSR._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fsr2 = FSR(features * 2)
        self.encoder3 = UNet_FSR._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fsr3 = FSR(features * 4)
        self.encoder4 = UNet_FSR._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fsr4 = FSR(features * 8)

        self.bottleneck = UNet_FSR._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet_FSR._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet_FSR._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet_FSR._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet_FSR._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        self.encoder_fusions = nn.ModuleList()
        self.decoder_fusions = nn.ModuleList()
        self.decoder_fusions.append(nn.Conv2d(192 * 2, 256, kernel_size=1))
        self.decoder_fusions.append(nn.Conv2d(160 * 2, 256, kernel_size=1))
        self.decoder_fusions.append(nn.Conv2d(144 * 2, 256, kernel_size=1))
        self.encoder_fusions.append(nn.Conv2d(288, 256, kernel_size=1))
        self.encoder_fusions.append(nn.Conv2d(320, 256, kernel_size=1))
        self.encoder_fusions.append(nn.Conv2d(384, 256, kernel_size=1))

    def forward(self, x, return_feature=False, return_entropy_hist=False):
        if self.amp_norm is not None:
            x = self.amp_norm(x)
        encoder_features = []
        decoder_features = []
        enc1 = self.encoder1(x)
        encoder_features.append(enc1)
        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = self.fsr2(enc2)
        encoder_features.append(enc2)
        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = self.fsr3(enc3)
        encoder_features.append(enc3)
        enc4 = self.encoder4(self.pool3(enc3))
        enc4 = self.fsr4(enc4)
        encoder_features.append(enc4)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        decoder_features.append(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        decoder_features.append(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        decoder_features.append(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        decoder_features.append(dec1)
        final = decoder_features[-1]
        decoder_features = decoder_features[:-1]
        dec1 = self.conv(dec1)
        if return_feature and return_entropy_hist:
            return dec1, encoder_features, decoder_features, final, self.entropy_hist(final)[1]
        elif return_feature:
            return dec1, encoder_features, decoder_features, final
        elif return_entropy_hist:
            return dec1, self.entropy_hist(final)[1]
        else:
            return dec1

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "_conv1", nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False)),
                    (name + "_bn1", nn.BatchNorm2d(num_features=features, affine=False, track_running_stats=False)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (name + "_conv2", nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)),
                    (name + "_bn2", nn.BatchNorm2d(num_features=features, affine=False, track_running_stats=False)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class Entropy_Hist(nn.Module):
    def __init__(self, ratio, win_w=3, win_h=3):
        super(Entropy_Hist, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.ratio = ratio

    def calcIJ_new(self, img_patch):
        total_p = img_patch.shape[-1] * img_patch.shape[-2]
        if total_p % 2 != 0:
            tem = torch.flatten(img_patch, start_dim=-2, end_dim=-1)
            center_p = tem[:, :, :, int(total_p / 2)]
            mean_p = (torch.sum(tem, dim=-1) - center_p) / (total_p - 1)
            if torch.is_tensor(img_patch):
                return center_p * 100 + mean_p
            else:
                return (center_p, mean_p)
        else:
            print("modify patch size")

    def histc_fork(self, ij):
        BINS = 256
        B, C = ij.shape
        N = 16
        BB = B // N
        min_elem = ij.min()
        max_elem = ij.max()
        ij = ij.view(N, BB, C)

        def f(x):
            with torch.no_grad():
                res = []
                for e in x:
                    res.append(torch.histc(e, bins=BINS, min=min_elem, max=max_elem))
                return res

        futures: List[torch.jit.Future[torch.Tensor]] = []

        for i in range(N):
            futures.append(torch.jit.fork(f, ij[i]))

        results = []
        for future in futures:
            results += torch.jit.wait(future)
        with torch.no_grad():
            out = torch.stack(results)
        return out

    def forward(self, img):
        with torch.no_grad():
            B, C, H, W = img.shape
            ext_x = int(self.win_w / 2)
            ext_y = int(self.win_h / 2)

            new_width = ext_x + W + ext_x
            new_height = ext_y + H + ext_y

            nn_Unfold = nn.Unfold(kernel_size=(self.win_w, self.win_h), dilation=1, padding=ext_x, stride=1)
            x = nn_Unfold(img)
            x = x.view(B, C, 3, 3, -1).permute(0, 1, 4, 2, 3)
            ij = self.calcIJ_new(x).reshape(B * C, -1)

            fij_packed = self.histc_fork(ij)
            p = fij_packed / (new_width * new_height)
            h_tem = -p * torch.log(torch.clamp(p, min=1e-40)) / math.log(2)

            a = torch.sum(h_tem, dim=1)
            H = a.reshape(B, C)

            _, index = torch.topk(H, int(self.ratio * C), dim=1)
        selected = []
        for i in range(img.shape[0]):
            selected.append(torch.index_select(img[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)

        sample_entropy = H.mean(dim=1)

        return selected, sample_entropy

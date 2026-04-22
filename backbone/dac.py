import numpy as np
import torch.nn as nn
import copy
from torch import Tensor
import torch
import math
import torch.nn.functional as F
import random


class FSR(nn.Module):
    def __init__(self, channels, channel_first=True):
        super().__init__()
        self.channels = channels
        self.channel_first = channel_first
        if channel_first:
            self.weight = nn.Parameter(torch.ones(channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
            self.h = nn.Parameter(torch.zeros(2, 1, 1))
        else:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
            self.h = nn.Parameter(torch.zeros(2))
        self.T = 0.1

    def forward(self, x):
        weight = F.softmax(self.h / self.T, dim=0)

        if not self.channel_first:
            x = x.permute(0, 3, 1, 2)
        norm = F.instance_norm(x)
        phase, amp_ori = decompose(x, 'all')
        norm_phase, norm_amp = decompose(norm, 'all')
        amp = norm_amp * weight[0] + amp_ori * weight[1]
        x = compose(phase, amp)
        if not self.channel_first:
            x = x.permute(0, 2, 3, 1)

        return x * self.weight + self.bias


def replace_denormals(x, threshold=1e-5):
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y


def decompose(x, mode='all'):
    fft_im = torch.view_as_real(torch.fft.fft2(x, norm='backward'))
    if mode == 'all' or mode == 'amp':
        fft_amp = fft_im.pow(2).sum(dim=-1, keepdim=False)
        fft_amp = torch.sqrt(replace_denormals(fft_amp))
    else:
        fft_amp = None

    if mode == 'all' or mode == 'phase':
        fft_pha = torch.atan2(fft_im[..., 1], replace_denormals(fft_im[..., 0]))
    else:
        fft_pha = None
    return fft_pha, fft_amp


def compose(phase, amp):
    x = torch.stack([torch.cos(phase) * amp, torch.sin(phase) * amp], dim=-1)
    x = x / math.sqrt(x.shape[2] * x.shape[3])
    x = torch.view_as_complex(x)
    return torch.fft.irfft2(x, s=x.shape[2:], norm='ortho')

import sys

import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import numpy as np


# 支持空洞率
class TDRConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, in_ratio=0.5, out_ratio=0.5, exp_times=2,
                 reduction=16, base_g=1, padding=1, dilation=1, bias=False):
        super(TDRConv, self).__init__()
        # self.kernel_size = (kernel_size,kernel_size)
        self.out_channels = out_channels
        self.stride = stride
        # split ratio 输入特征的拆分比例
        self.in_ratio = in_ratio
        # 是否需要 通道匹配（特征拓展模块的输出  需要与 特征提取模块的输出 进行add融合 若通道个数不一致 需要进行匹配）
        self.need_match = False
        # the base branch output channels
        base_out_channels = int(math.ceil(out_channels * out_ratio))
        # main_out_channels = int(math.ceil(out_channels * out_ratio))
        # the diversity branch output channels
        diversity_out_channels = out_channels - base_out_channels
        # print(diversity_out_channels, out_channels, base_out_channels)
        # detail_out_channels = out_channels - main_out_channels
        exp_out_channels = diversity_out_channels * exp_times
        # the main part channels
        self.main_in = int(math.ceil(in_channels * in_ratio))
        # the expansion part channels
        exp_in = in_channels - self.main_in
        # the diversity branch  channels
        diversity_in = self.main_in + exp_out_channels
        base_groups = base_g if base_out_channels % base_g == 0 and self.main_in % base_g == 0 else 1

        if dilation == 1:
            padding = kernel_size // 2
        else:
            padding = dilation
        # self.base_branch = nn.Conv2d(in_channels=self.main_in, out_channels=base_out_channels, kernel_size=kernel_size,
        #                              stride=stride, padding=kernel_size // 2,groups=base_groups, bias=False)
        self.base_branch = nn.Conv2d(in_channels=self.main_in, out_channels=base_out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=base_groups, bias=bias, dilation=dilation)
        # print(exp_out_channels)
        if exp_out_channels == 0:
            exp_out_channels = out_channels
        # part_exp
        if exp_in != 0:
            self.expand_operation = nn.Conv2d(in_channels=exp_in, out_channels=exp_out_channels, kernel_size=1,
                                              stride=1, padding=0, bias=False)
        else:
            self.expand_operation = None
            diversity_in = self.main_in
        if exp_out_channels != self.out_channels:
            self.need_match = True
            # print(exp_out_channels, self.out_channels, base_out_channels)
            self.match_branch = nn.Conv2d(in_channels=exp_out_channels, out_channels=self.out_channels, kernel_size=1,
                                          stride=1, padding=0, groups=base_out_channels, bias=False)

        # diversity_groups = self.main_in
        # if diversity_out_channels % self.main_in != 0:
        #     diversity_groups = int(math.ceil(diversity_groups / 2))
        # 修改diversity branch分组的计算!!! 分组数为 多样性分支的输出通道 与 主要部分的通道数的最大公约数
        diversity_groups = math.gcd(diversity_out_channels, self.main_in)
        if diversity_out_channels != 0:
            # self.diversity_branch = nn.Conv2d(in_channels=diversity_in, out_channels=diversity_out_channels,
            #                                   kernel_size=kernel_size,
            #                                   stride=1, padding=kernel_size // 2, groups=diversity_groups, bias=False)
            self.diversity_branch = nn.Conv2d(in_channels=diversity_in, out_channels=diversity_out_channels,
                                              kernel_size=kernel_size,
                                              stride=1, padding=padding, groups=diversity_groups, bias=bias,
                                              dilation=dilation)
        else:
            self.diversity_branch = None
        self.bn1 = nn.BatchNorm2d(diversity_in)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.avgpool_s2_diversity = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool_s2_expand = nn.AvgPool2d(kernel_size=2, stride=2)
        # SE
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        se_out = out_channels
        self.fc = nn.Sequential(
            nn.Linear(se_out, max(2, se_out // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(2, se_out // reduction), se_out),
            nn.Sigmoid()
        )
        #
        self.alpha = nn.Parameter(torch.ones(in_channels))

    def forward(self, x):
        # 将输入拆分为x_m,x_e 分别进行处理
        x_m = x[:, :self.main_in, :, :]
        x_e = x[:, self.main_in:, :, :]
        # base对x_m进行特征提取
        y_sc = self.base_branch(x_m)
        # 步长为2 则对两个拆分输入x_m,x_e进行下采样
        if self.stride == 2:
            x_m = self.avgpool_s2_diversity(x_m)
            x_e = self.avgpool_s2_expand(x_e)
        # x_e进行拓展操作->y_e
        if self.expand_operation is not None:
            y_e = self.expand_operation(x_e)
        else:
            y_e = 0
        # diversity对x_m及拓展后的y_e进行特征提取
        if self.diversity_branch is not None:
            x_gwc = torch.cat([x_m, y_e], dim=1)
            x_gwc = self.bn1(x_gwc)
            y_gwc = self.diversity_branch(x_gwc)
            y_m = torch.cat([y_sc, y_gwc], dim=1)
        else:
            y_m = y_sc
        y_m = self.bn2(y_m)
        # se
        b, c, _, _ = y_m.size()
        w = self.avg_pool(y_m).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        # print(torch.max(w), torch.min(w))
        y_m = y_m * w

        if self.need_match:
            y_e = self.match_branch(y_e)
        # add op
        y = y_m + y_e
        return y[:, :self.out_channels, :, :]



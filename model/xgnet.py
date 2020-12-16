import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from base import BaseModel
from model.blocks import *


class XGNet(BaseModel):
    def __init__(self, block, heads, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(XGNet, self).__init__()
        block = eval(block)
        layers = [3, 4, 6, 3]
        self.inplanes = 64
        self.heads = eval(heads)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # pixel
        self.decoder4_p = self._make_deconv_layer(UpBlock, 512 * block.expansion, 256, stride=2)
        self.decoder3_p = self._make_deconv_layer(UpBlock, 256, 128, stride=2)
        self.decoder2_p = self._make_deconv_layer(UpBlock, 128, 64, stride=2)
        self.decoder1_p = self._make_deconv_layer(UpBlock, 64, 32, stride=2)
        self.decoder0_p = self._make_deconv_layer(UpBlock, 32, 16, stride=2)

        self.outfea_p = nn.Conv2d(16, 10, kernel_size=3, padding=1)
        self.out_p = nn.Conv2d(10, 1, kernel_size=1)

        # edge
        self.edge_0_1 = nn.Conv2d(64, 8, 3, padding=1)
        self.edge_0_2 = nn.Conv2d(8, 1, 4, padding=2)  # todo padding
        self.edge_1_1 = nn.Conv2d(256, 8, 3, padding=1)
        self.edge_1_2 = nn.Conv2d(8, 1, 4, padding=2)  # todo padding
        self.edge_2_1 = nn.Conv2d(512, 8, 3, padding=1)
        self.edge_2_2 = nn.Conv2d(8, 1, 4, padding=2)  # todo padding
        self.edge_3_1 = nn.Conv2d(1024, 8, 3, padding=1)
        self.edge_3_2 = nn.Conv2d(8, 1, 4, padding=2)  # todo padding

        # region
        self.region_0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=8, stride=8, padding=4),  # todo padding
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.region_1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=8, stride=8, padding=4),  # todo padding
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.region_2 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=8, stride=8, padding=4),  # todo padding
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.region_3 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=8, stride=8, padding=4),  # todo padding
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)
        self.fuse_e = nn.Conv2d(4, 1, kernel_size=1)
        self.fuse_r = nn.Conv2d(4, 1, kernel_size=1)

        self.fuse_se = SELayer(20, 9)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, block, inplanes, planes, stride=1):
        layers = block(inplanes, planes, stride=stride)
        return layers

    def forward(self, x):
        # Encoder
        e0 = self.conv1(x)
        e0 = self.bn1(e0)
        e0 = self.relu(e0)

        e1 = self.maxpool(e0)
        e1 = self.layer1(e1)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        # seg
        d4 = self.decoder4_p(e4) + e3
        d3 = self.decoder3_p(d4) + e2
        d2 = self.decoder2_p(d3) + e1
        d1 = self.decoder1_p(d2) + e0
        fea_p = self.decoder0_p(d1)


        # edge


        #

        out = self.deconv(d1)

        return out


class UpBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(UpBlock, self).__init__()
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def _up_sample(self, x, factor=2):
        return F.interpolate(x, scale_factor=factor, mode='nearest')

    def forward(self, x):
        x = self._up_sample(x, self.stride)
        out = self.conv(x)
        return out


class squeeze_excitation_layer(nn.Module):
    def __init__(self, inplanes, planes):
        super(squeeze_excitation_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Linear(inplanes, planes),
            nn.ReLU(inplace=True),
            nn.Linear(planes, planes),
            nn.Sigmoid())

    def forward(self, x):
        u = self.ca(self.avg_pool(x))
        ca_out = x * u

        return ca_out


if __name__ == '__main__':
    model = XGNet('Bottleneck')
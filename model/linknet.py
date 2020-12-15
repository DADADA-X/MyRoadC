import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from base import BaseModel
from model.blocks import *


class LinkNet(BaseModel):
    def __init__(self, block, heads, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(LinkNet, self).__init__()
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

        # deconv layers
        self.decoder4 = self._make_deconv_layer(DecoderBlock, 512 * block.expansion, 256 * block.expansion, stride=2)
        self.decoder3 = self._make_deconv_layer(DecoderBlock, 256 * block.expansion, 128 * block.expansion, stride=2)
        self.decoder2 = self._make_deconv_layer(DecoderBlock, 128 * block.expansion, 64 * block.expansion, stride=2)
        self.decoder1 = self._make_deconv_layer(DecoderBlock, 64 * block.expansion, 64)

        num_class = self.heads[0]
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,  # TODO intermediate channel
                               output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, num_class, kernel_size=2, stride=2)
        )

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
        rows = x.size()[2]
        cols = x.size()[3]

        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        e1 = self.maxpool(x)
        e1 = self.layer1(e1)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.deconv(d1)

        return out


class DLinkNet(BaseModel):
    def __init__(self, block, heads, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(DLinkNet, self).__init__()
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

        self.dblock = Dblock(512 * block.expansion)

        # deconv layers
        self.decoder4 = self._make_deconv_layer(DecoderBlock, 512 * block.expansion, 256 * block.expansion, stride=2)
        self.decoder3 = self._make_deconv_layer(DecoderBlock, 256 * block.expansion, 128 * block.expansion, stride=2)
        self.decoder2 = self._make_deconv_layer(DecoderBlock, 128 * block.expansion, 64 * block.expansion, stride=2)
        self.decoder1 = self._make_deconv_layer(DecoderBlock, 64 * block.expansion, 64)

        num_class = self.heads[0]
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,  # TODO intermediate channel
                               output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, num_class, kernel_size=2, stride=2)
        )

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
        rows = x.size()[2]
        cols = x.size()[3]

        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        e1 = self.maxpool(x)
        e1 = self.layer1(e1)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.deconv(d1)

        return out


class MHLinkNet(BaseModel):
    def __init__(self, block, heads, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(MHLinkNet, self).__init__()
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

        # mask deconv layers
        self.decoder4 = self._make_deconv_layer(DecoderBlock, 512 * block.expansion, 256 * block.expansion, stride=2)
        self.decoder3 = self._make_deconv_layer(DecoderBlock, 256 * block.expansion, 128 * block.expansion, stride=2)
        self.decoder2 = self._make_deconv_layer(DecoderBlock, 128 * block.expansion, 64 * block.expansion, stride=2)
        self.decoder1 = self._make_deconv_layer(DecoderBlock, 64 * block.expansion, 64)

        for i, num_classes in enumerate(self.heads):
            deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,  # TODO intermediate channel
                                   output_padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
            )
            self.__setattr__(str(i), deconv)

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
        rows = x.size()[2]
        cols = x.size()[3]
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        e1 = self.maxpool(x)
        e1 = self.layer1(e1)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        # Decoder
        d4 = self.decoder4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
            ] + e3
        d3 = self.decoder3(d4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
            ] + e2
        d2 = self.decoder2(d3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
            ] + e1
        d1 = self.decoder1(d2)

        if len(self.heads) == 1:
            out = self.__getattr__("0")(d1)
        else:
            out = []
            for i in range(len(self.heads)):
                out.append(self.__getattr__(str(i))(d1))

        return out


class MBLinkNet(BaseModel):
    def __init__(self, block, heads, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(MBLinkNet, self).__init__()
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

        # mask deconv layers
        self.decoder_m4 = self._make_deconv_layer(DecoderBlock, 512 * block.expansion, 256 * block.expansion, stride=2)
        self.decoder_m3 = self._make_deconv_layer(DecoderBlock, 256 * block.expansion, 128 * block.expansion, stride=2)
        self.decoder_m2 = self._make_deconv_layer(DecoderBlock, 128 * block.expansion, 64 * block.expansion, stride=2)
        self.decoder_m1 = self._make_deconv_layer(DecoderBlock, 64 * block.expansion, 64)

        self.deconv_m = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,  # TODO intermediate channel
                               output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, self.heads[0], kernel_size=2, stride=2)
        )

        # connection deconv layers
        self.decoder_c4 = self._make_deconv_layer(DecoderBlock, 512 * block.expansion, 256 * block.expansion, stride=2)
        self.decoder_c3 = self._make_deconv_layer(DecoderBlock, 256 * block.expansion, 128 * block.expansion, stride=2)
        self.decoder_c2 = self._make_deconv_layer(DecoderBlock, 128 * block.expansion, 64 * block.expansion, stride=2)
        self.decoder_c1 = self._make_deconv_layer(DecoderBlock, 64 * block.expansion, 64)

        self.deconv_c = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,  # TODO intermediate channel
                               output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, self.heads[1], kernel_size=2, stride=2)
        )

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
        rows = x.size()[2]
        cols = x.size()[3]
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        e1 = self.maxpool(x)
        e1 = self.layer1(e1)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        # Decoder
        d_m4 = self.decoder_m4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
            ] + e3
        d_m3 = self.decoder_m3(d_m4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
            ] + e2
        d_m2 = self.decoder_m2(d_m3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
            ] + e1
        d_m1 = self.decoder_m1(d_m2)
        out_m = self.deconv_m(d_m1)

        d_c4 = self.decoder_c4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
            ] + e3
        d_c3 = self.decoder_c3(d_c4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
            ] + e2
        d_c2 = self.decoder_c2(d_c3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
            ] + e1
        d_c1 = self.decoder_c1(d_c2)
        out_c = self.deconv_c(d_c1)

        return out_m, out_c


class SideLinkNet(BaseModel):
    def __init__(self, block, heads, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(SideLinkNet, self).__init__()
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

        # deconv layers
        self.decoder4 = self._make_deconv_layer(DecoderBlock, 512 * block.expansion, 256 * block.expansion, stride=2)
        self.decoder3 = self._make_deconv_layer(DecoderBlock, 256 * block.expansion, 128 * block.expansion, stride=2)
        self.decoder2 = self._make_deconv_layer(DecoderBlock, 128 * block.expansion, 64 * block.expansion, stride=2)
        self.decoder1 = self._make_deconv_layer(DecoderBlock, 64 * block.expansion, 64)

        # feature transform layer
        self.side4 = FeatureTrans(block, 256 * block.expansion)
        self.side3 = FeatureTrans(block, 128 * block.expansion)
        self.side2 = FeatureTrans(block, 64 * block.expansion)

        self.sidefusion = SideInfoFusion(64)

        self.maskout = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, self.heads[0], kernel_size=2, stride=2)
        )
        self.connout = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, self.heads[1], kernel_size=2, stride=2)
        )

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
        rows = x.size()[2]
        cols = x.size()[3]

        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        e1 = self.maxpool(x)
        e1 = self.layer1(e1)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        # Decoder
        d4 = self.decoder4(e4)[
                :, :, : int(math.ceil(rows / 16.0)), : int(math.ceil(cols / 16.0))
            ] + e3
        d3 = self.decoder3(d4)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
            ] + e2
        d2 = self.decoder2(d3)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
            ] + e1
        d1 = self.decoder1(d2)

        s4 = self.side4(d4)
        s3 = self.side3(d3)
        s2 = self.side2(d2)
        s = self.sidefusion(s2, s3, s4, rows, cols)

        mask_out = self.maskout(torch.cat([d1, s], dim=1))
        conn_out = self.connout(s)

        return mask_out, conn_out


class MBLinkNet3(BaseModel):
    def __init__(self, block, heads, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(MBLinkNet3, self).__init__()
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

        # mask deconv layers
        self.decoder_m4 = self._make_deconv_layer(DecoderBlock, 512 * block.expansion, 256 * block.expansion, stride=2)
        self.decoder_m3 = self._make_deconv_layer(DecoderBlock, 256 * block.expansion, 128 * block.expansion, stride=2)
        self.decoder_m2 = self._make_deconv_layer(DecoderBlock, 128 * block.expansion, 64 * block.expansion, stride=2)
        self.decoder_m1 = self._make_deconv_layer(DecoderBlock, 64 * block.expansion, 64)

        self.deconv_m = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,  # TODO intermediate channel
                               output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, self.heads[0], kernel_size=2, stride=2)
        )

        # connection deconv layers
        self.decoder_c4 = self._make_deconv_layer(DecoderBlock, 512 * block.expansion, 256 * block.expansion, stride=2)
        self.decoder_c3 = self._make_deconv_layer(DecoderBlock, 256 * block.expansion, 128 * block.expansion, stride=2)
        self.decoder_c2 = self._make_deconv_layer(DecoderBlock, 128 * block.expansion, 64 * block.expansion, stride=2)
        self.decoder_c1 = self._make_deconv_layer(DecoderBlock, 64 * block.expansion, 64)

        self.deconv_c = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,  # TODO intermediate channel
                               output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, self.heads[1], kernel_size=2, stride=2)
        )

        # connection deconv layers
        self.decoder_p4 = self._make_deconv_layer(DecoderBlock, 512 * block.expansion, 256 * block.expansion, stride=2)
        self.decoder_p3 = self._make_deconv_layer(DecoderBlock, 256 * block.expansion, 128 * block.expansion, stride=2)
        self.decoder_p2 = self._make_deconv_layer(DecoderBlock, 128 * block.expansion, 64 * block.expansion, stride=2)
        self.decoder_p1 = self._make_deconv_layer(DecoderBlock, 64 * block.expansion, 64)

        self.deconv_p = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,  # TODO intermediate channel
                               output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, self.heads[2], kernel_size=2, stride=2)
        )

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        e1 = self.maxpool(x)
        e1 = self.layer1(e1)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        # Decoder
        d_m4 = self.decoder_m4(e4) + e3
        d_m3 = self.decoder_m3(d_m4) + e2
        d_m2 = self.decoder_m2(d_m3) + e1
        d_m1 = self.decoder_m1(d_m2)
        out_m = self.deconv_m(d_m1)

        d_c4 = self.decoder_c4(e4) + e3
        d_c3 = self.decoder_c3(d_c4) + e2
        d_c2 = self.decoder_c2(d_c3) + e1
        d_c1 = self.decoder_c1(d_c2)
        out_c = self.deconv_c(d_c1)

        d_p4 = self.decoder_p4(e4) + e3
        d_p3 = self.decoder_p3(d_p4) + e2
        d_p2 = self.decoder_p2(d_p3) + e1
        d_p1 = self.decoder_p1(d_p2)
        out_p = self.deconv_c(d_p1)

        return out_m, out_c, out_p


class FeatureTrans(nn.Module):
    """
    n -> n transform + n -> 64 compose
    """
    def __init__(self, block, inplanes):
        super(FeatureTrans, self).__init__()
        self.res1 = block(inplanes, inplanes)
        self.res2 = block(inplanes, inplanes)
        self.res3 = block(inplanes, inplanes)
        # feature compression
        self.fc = conv1x1(inplanes, 64)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        out = self.fc(x)
        return out


class SideInfoFusion(nn.Module):
    def __init__(self, inplanes):
        super(SideInfoFusion, self).__init__()
        self.layer_block1 = nn.Conv2d(inplanes, inplanes, 3, padding=1)
        self.layer_block2 = nn.Conv2d(inplanes, inplanes, 3, padding=1)
        self.layer_block3 = nn.Conv2d(inplanes, inplanes, 3, padding=1)

    def _up_sample(self, x, factor=2):
        return F.interpolate(x, scale_factor=factor, mode='nearest')

    def forward(self, x1, x2, x3, rows, cols):
        out_3 = self.layer_block3(x3)
        out_2 = self.layer_block2(self._up_sample(out_3)[
                :, :, : int(math.ceil(rows / 8.0)), : int(math.ceil(cols / 8.0))
            ] + x2)
        out_1 = self.layer_block1(self._up_sample(out_2)[
                :, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))
            ] + x1)
        return out_1


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        d1 = self.relu(self.dilate1(x))
        d2 = self.relu(self.dilate1(d1))
        d3 = self.relu(self.dilate1(d2))
        d4 = self.relu(self.dilate1(d3))
        d5 = self.relu(self.dilate1(d4))
        out = d1 + d2 + d3 + d4 + d5
        return out


import cv2
import time
from torchvision import transforms
from utils import weights_init
if __name__ == '__main__':
    seed = 1234
    if seed is None:
        seed = np.random.randint(1, 10000)
    else:
        seed = int(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # ----------- !!! ------------ #
    model = DLinkNet("BasicBlock", "[2]")
    weights_init(model, seed=seed)
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            count += 1
    print(model)
    print(count)
    image = cv2.imread('/home/data/xyj/spacenet/train/images/RGB-PanSharpen_AOI_2_Vegas_img853_18.tif')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy((image/255.0).transpose((2, 0, 1)).astype(np.float32))
    mean = [0.334, 0.329, 0.326]
    std = [0.161, 0.153, 0.144]
    normalize = transforms.Normalize(mean, std)
    image = normalize(image).unsqueeze(0)
    s = time.time()
    out = model.forward(image)
    e = time.time()
    print(e-s)
    print(out.shape)
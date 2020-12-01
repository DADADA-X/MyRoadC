import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from base import BaseModel


class LinkNet(BaseModel):
    def __init__(self, heads, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(LinkNet, self).__init__()
        block = SCBasicBlock
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


class MTLLinkNet(BaseModel):
    def __init__(self, heads, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(MTLLinkNet, self).__init__()
        block = SCBasicBlock
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

        return out_m, out_c

# todo just for test
class MTLLinkNet3(BaseModel):
    def __init__(self, heads, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(MTLLinkNet3, self).__init__()
        block = SCBasicBlock
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


# todo just for test
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SideLinkNet(BaseModel):
    def __init__(self, heads, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(SideLinkNet, self).__init__()
        block = SCBasicBlock
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
        self.side4 = FeatureTrans(256 * block.expansion)
        self.side3 = FeatureTrans(128 * block.expansion)
        self.side2 = FeatureTrans(64 * block.expansion)

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

        s4 = self.side4(d4)
        s3 = self.side3(d3)
        s2 = self.side2(d2)
        s = self.sidefusion(s2, s3, s4)

        mask_out = self.maskout(torch.cat([d1, s], dim=1))
        conn_out = self.connout(s)

        return mask_out, conn_out


# ======================================================== #
#                        Utilities                         #
# ======================================================== #


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SCBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SCBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # self.se = SCLayer(planes, ratio=16)
        self.se = SCLayer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.Sequential(
            conv1x1(inplanes, inplanes // 4),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inplanes // 4, inplanes // 4, kernel_size=3, stride=stride, padding=1,
                               output_padding=stride - 1, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True),
            conv1x1(inplanes // 4, planes),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.deconv(x)
        return out


class SCLayer(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SCLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        u = self.ca(self.avg_pool(x))
        ca_out = x * u

        v = torch.mean(ca_out, dim=1).unsqueeze(1).sigmoid()
        sa_out = ca_out * v

        return sa_out


class FeatureTrans(nn.Module):
    """
    n -> n transform + n -> 64 compose
    """
    def __init__(self, inplanes):
        super(FeatureTrans, self).__init__()
        self.res1 = SCBasicBlock(inplanes, inplanes)
        self.res2 = SCBasicBlock(inplanes, inplanes)
        self.res3 = SCBasicBlock(inplanes, inplanes)
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

    def forward(self, x1, x2, x3):
        out_3 = self.layer_block3(x3)
        out_2 = self.layer_block2(self._up_sample(out_3) + x2)
        out_1 = self.layer_block1(self._up_sample(out_2) + x1)
        return out_1


if __name__ == '__main__':
    net = MTLLinkNet("[2, 5]")
    x = torch.randn(1, 3, 256, 256)
    print(net)
    print(net(x)[0].shape, net(x)[1].shape)

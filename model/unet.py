import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class UNet(BaseModel):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.mp = nn.MaxPool2d(kernel_size=2)

        self.decoder4 = self.deconv_block(512, 256)
        self.decoder3 = nn.Sequential(self.conv_block(512, 256), self.deconv_block(256, 128))
        self.decoder2 = nn.Sequential(self.conv_block(256, 128), self.deconv_block(128, 64))

        self.out = nn.Sequential(self.conv_block(128, 64), nn.Conv2d(64, 2, 1))

    def conv_block(self, in_channels, out_channels, batch_norm=True):
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True))

    def sc_conv_block(self, in_channels, out_channels, batch_norm=True):
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                SCLayer(out_channels))
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True))

    def se_conv_block(self, in_channels, out_channels, batch_norm=True):
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                SELayer(out_channels))
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True))

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        rows = x.size()[2]

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.mp(e1))
        e3 = self.encoder3(self.mp(e2))
        e4 = self.encoder4(self.mp(e3))

        e1 = self.convert1(e1)
        e2 = self.convert2(e2)
        e3 = self.convert3(e3)
        e4 = self.convert4(e4)

        if rows % 8:
            out = F.pad(self.decoder4(e4), [0, 1, 0, 1])
        else:
            out = self.decoder4(e4)
        out = self.decoder3(torch.cat([out, e3], dim=1))
        out = self.decoder2(torch.cat([out, e2], dim=1))
        out = self.out(torch.cat([out, e1], dim=1))

        return out


class MHUNet(BaseModel):
    def __init__(self):
        super(MHUNet, self).__init__()
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.mp = nn.MaxPool2d(kernel_size=2)

        self.decoder4 = self.deconv_block(512, 256)
        self.decoder3 = nn.Sequential(self.conv_block(512, 256), self.deconv_block(256, 128))
        self.decoder2 = nn.Sequential(self.conv_block(256, 128), self.deconv_block(128, 64))

        self.out_m = nn.Sequential(self.conv_block(128, 64), nn.Conv2d(64, 2, 1))
        self.out_c = nn.Sequential(self.conv_block(128, 64), nn.Conv2d(64, 5, 1))

    def conv_block(self, in_channels, out_channels, batch_norm=True):
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True))

    def sc_conv_block(self, in_channels, out_channels, batch_norm=True):
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                SCLayer(out_channels))
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True))

    def se_conv_block(self, in_channels, out_channels, batch_norm=True):
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                SELayer(out_channels))
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True))

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        rows = x.size()[2]

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.mp(e1))
        e3 = self.encoder3(self.mp(e2))
        e4 = self.encoder4(self.mp(e3))

        if rows % 8:
            out = F.pad(self.decoder4(e4), [0, 1, 0, 1])
        else:
            out = self.decoder4(e4)
        out = self.decoder3(torch.cat([out, e3], dim=1))
        out = self.decoder2(torch.cat([out, e2], dim=1))

        out_m = self.out_m(torch.cat([out, e1], dim=1))
        out_c = self.out_c(torch.cat([out, e1], dim=1))

        return out_m, out_c


# ------------------ util ------------------ #


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SEBasicBlock, self).__init__()
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
        self.se = SELayer(planes)

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


class SELayer(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        u = self.ca(self.avg_pool(x))
        ca_out = x * u

        return ca_out


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


import cv2
import time
import numpy as np
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

    model = UNet()
    weights_init(model, seed=seed)
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            count += 1
    print(model)
    print(count)
    image = cv2.imread('/home/data/xyj/spacenet/train/images/RGB-PanSharpen_AOI_2_Vegas_img853_18.tif')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy((image / 255.0).transpose((2, 0, 1)).astype(np.float32))
    mean = [0.334, 0.329, 0.326]
    std = [0.161, 0.153, 0.144]
    normalize = transforms.Normalize(mean, std)
    image = normalize(image).unsqueeze(0)
    image = torch.randn(1, 3, 1300, 1300)
    s = time.time()
    model.forward(image)
    e = time.time()
    print(e - s)
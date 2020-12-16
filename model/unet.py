import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from model.blocks import *


class UNet(BaseModel):
    def __init__(self, block):
        super(UNet, self).__init__()
        if block == "BasicBlock":
            self.encoder1 = self.conv_block(3, 64)
            self.encoder2 = self.conv_block(64, 128)
            self.encoder3 = self.conv_block(128, 256)
            self.encoder4 = self.conv_block(256, 512)
        elif block == "SEBasicBlock":
            self.encoder1 = self.se_conv_block(3, 64)
            self.encoder2 = self.se_conv_block(64, 128)
            self.encoder3 = self.se_conv_block(128, 256)
            self.encoder4 = self.se_conv_block(256, 512)
        elif block == "GABasicBlock":
            self.encoder1 = self.ga_conv_block(3, 64)
            self.encoder2 = self.ga_conv_block(64, 128)
            self.encoder3 = self.ga_conv_block(128, 256)
            self.encoder4 = self.ga_conv_block(256, 512)

        self.mp = nn.MaxPool2d(kernel_size=2)

        self.decoder4 = self.deconv_block(512, 256)
        self.decoder3 = nn.Sequential(self.conv_block(512, 256), self.deconv_block(256, 128))
        self.decoder2 = nn.Sequential(self.conv_block(256, 128), self.deconv_block(128, 64))

        self.out = nn.Sequential(self.conv_block(128, 64), nn.Conv2d(64, 2, 1))

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def se_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SELayer(out_channels))

    def ga_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            GALayer(out_channels))

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.mp(e1))
        e3 = self.encoder3(self.mp(e2))
        e4 = self.encoder4(self.mp(e3))

        out = self.decoder4(e4)
        out = self.decoder3(torch.cat([out, e3], dim=1))
        out = self.decoder2(torch.cat([out, e2], dim=1))
        out = self.out(torch.cat([out, e1], dim=1))

        return out


class MHUNet(BaseModel):
    def __init__(self, block):
        super(MHUNet, self).__init__()
        if block == "BasicBlock":
            self.encoder1 = self.conv_block(3, 64)
            self.encoder2 = self.conv_block(64, 128)
            self.encoder3 = self.conv_block(128, 256)
            self.encoder4 = self.conv_block(256, 512)
        elif block == "SEBasicBlock":
            self.encoder1 = self.se_conv_block(3, 64)
            self.encoder2 = self.se_conv_block(64, 128)
            self.encoder3 = self.se_conv_block(128, 256)
            self.encoder4 = self.se_conv_block(256, 512)
        elif block == "GABasicBlock":
            self.encoder1 = self.ga_conv_block(3, 64)
            self.encoder2 = self.ga_conv_block(64, 128)
            self.encoder3 = self.ga_conv_block(128, 256)
            self.encoder4 = self.ga_conv_block(256, 512)

        self.mp = nn.MaxPool2d(kernel_size=2)

        self.decoder4 = self.deconv_block(512, 256)
        self.decoder3 = nn.Sequential(self.conv_block(512, 256), self.deconv_block(256, 128))
        self.decoder2 = nn.Sequential(self.conv_block(256, 128), self.deconv_block(128, 64))

        self.out_m = nn.Sequential(self.conv_block(128, 64), nn.Conv2d(64, 2, 1))
        self.out_c = nn.Sequential(self.conv_block(128, 64), nn.Conv2d(64, 5, 1))

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def se_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SELayer(out_channels))

    def ga_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            GALayer(out_channels))

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.mp(e1))
        e3 = self.encoder3(self.mp(e2))
        e4 = self.encoder4(self.mp(e3))

        out = self.decoder4(e4)
        out = self.decoder3(torch.cat([out, e3], dim=1))
        out = self.decoder2(torch.cat([out, e2], dim=1))

        out_m = self.out_m(torch.cat([out, e1], dim=1))
        out_c = self.out_c(torch.cat([out, e1], dim=1))

        return out_m, out_c


class MBUNet(BaseModel):
    def __init__(self, block):
        super(MBUNet, self).__init__()
        if block == "BasicBlock":
            self.encoder1 = self.conv_block(3, 64)
            self.encoder2 = self.conv_block(64, 128)
            self.encoder3 = self.conv_block(128, 256)
            self.encoder4 = self.conv_block(256, 512)
        elif block == "SEBasicBlock":
            self.encoder1 = self.se_conv_block(3, 64)
            self.encoder2 = self.se_conv_block(64, 128)
            self.encoder3 = self.se_conv_block(128, 256)
            self.encoder4 = self.se_conv_block(256, 512)
        elif block == "GABasicBlock":
            self.encoder1 = self.ga_conv_block(3, 64)
            self.encoder2 = self.ga_conv_block(64, 128)
            self.encoder3 = self.ga_conv_block(128, 256)
            self.encoder4 = self.ga_conv_block(256, 512)

        self.mp = nn.MaxPool2d(kernel_size=2)

        self.decoder4_m = self.deconv_block(512, 256)
        self.decoder3_m = nn.Sequential(self.conv_block(512, 256), self.deconv_block(256, 128))
        self.decoder2_m = nn.Sequential(self.conv_block(256, 128), self.deconv_block(128, 64))
        self.out_m = nn.Sequential(self.conv_block(128, 64), nn.Conv2d(64, 2, 1))

        self.decoder4_c = self.deconv_block(512, 256)
        self.decoder3_c = nn.Sequential(self.conv_block(512, 256), self.deconv_block(256, 128))
        self.decoder2_c = nn.Sequential(self.conv_block(256, 128), self.deconv_block(128, 64))
        self.out_c = nn.Sequential(self.conv_block(128, 64), nn.Conv2d(64, 5, 1))

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def se_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SELayer(out_channels))

    def ga_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            GALayer(out_channels))

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.mp(e1))
        e3 = self.encoder3(self.mp(e2))
        e4 = self.encoder4(self.mp(e3))

        d4_m = self.decoder4_m(e4)
        d3_m = self.decoder3_m(torch.cat([d4_m, e3], dim=1))
        d2_m = self.decoder2_m(torch.cat([d3_m, e2], dim=1))
        out_m = self.out_m(torch.cat([d2_m, e1], dim=1))

        d4_c = self.decoder4_m(e4)
        d3_c = self.decoder3_m(torch.cat([d4_c, e3], dim=1))
        d2_c = self.decoder2_m(torch.cat([d3_c, e2], dim=1))
        out_c = self.out_c(torch.cat([d2_c, e1], dim=1))

        return out_m, out_c


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

    model = MBUNet('BasicBlock')
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
    image = torch.randn(1, 3, 512, 512)
    s = time.time()
    model.forward(image)
    e = time.time()
    print(e - s)

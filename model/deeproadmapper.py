import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from base import BaseModel
from model.blocks import *


class RoadCNN(BaseModel):
    def __init__(self):
        super(RoadCNN, self).__init__()
        self.layer1 = nn.Sequential(conv3x3(3, 128, 2), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(conv3x3(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(conv3x3(128, 128, 2), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(conv3x3(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(conv3x3(128, 256, 2), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(conv3x3(256, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.layer7 = nn.Sequential(conv3x3(256, 512, 2), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.layer8 = nn.Sequential(conv3x3(512, 512, 2), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.layer9 = nn.Sequential(conv3x3(512, 512, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.layer10 = nn.Sequential(conv3x3(512, 512, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.layer11 = nn.Sequential(conv3x3(512, 512, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        self.layer12 = nn.Sequential(deconv3x3(512, 512, 2), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.layer13 = nn.Sequential(conv3x3(1024, 512, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.layer14 = nn.Sequential(deconv3x3(512, 256, 2), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.layer15 = nn.Sequential(conv3x3(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.layer16 = nn.Sequential(deconv3x3(256, 128, 2), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.layer17 = nn.Sequential(conv3x3(256, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.layer18 = nn.Sequential(deconv3x3(128, 128, 2), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.layer19 = nn.Sequential(deconv3x3(256, 128, 2), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.layer20 = nn.Sequential(conv3x3(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.pre_outputs = conv3x3(128, 2, 1)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)
        e6 = self.layer6(e5)
        e7 = self.layer7(e6)
        e8 = self.layer8(e7)
        e9 = self.layer9(e8)
        e10 = self.layer10(e9)
        e11 = self.layer11(e10)

        d12 = self.layer12(e11)
        d13 = self.layer13(torch.cat([e7, d12], dim=1))
        d14 = self.layer14(d13)
        d15 = self.layer15(torch.cat([e6, d14], dim=1))
        d16 = self.layer16(d15)
        d17 = self.layer17(torch.cat([e4, d16], dim=1))
        d18 = self.layer18(d17)
        d19 = self.layer19(torch.cat([e2, d18], dim=1))
        d20 = self.layer20(d19)

        out = self.pre_outputs(d20)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def deconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, output_padding=1)


if __name__ == '__main__':
    model = RoadCNN()
    image = torch.randn(1, 3, 512, 512)
    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
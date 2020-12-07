import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

__all__ = ['StackHourglass', 'MTLStackHourglass']


# class StackHourglass(BaseModel):
#     def __init__(self, block, heads, depth, num_stacks, num_blocks):
#         super(StackHourglass, self).__init__()
#         self.inplanes = 64  # keep updating
#         self.num_feats = 128
#         self.block = eval(block)
#         self.heads = heads
#         self.num_classes = sum(heads.values())
#         self.num_stacks = num_stacks
#
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self._make_residual(self.block, self.inplanes, 1)  # self.inplanes as plane in the function
#         self.layer2 = self._make_residual(self.block, self.inplanes, 1)
#         self.layer3 = self._make_residual(self.block, self.num_feats, 1)
#         self.maxpool = nn.MaxPool2d(2, stride=2)
#
#         # build hourglass modules
#         ch = self.num_feats * self.block.expansion  # 256
#         hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
#         for i in range(self.num_stacks):  # number of stacked-hourglass
#             hg.append(Hourglass(self.block, num_blocks, self.num_feats, depth))
#             res.append(self._make_residual(self.block, self.num_feats, num_blocks))
#             fc.append(self._make_fc(ch, ch))
#             score.append(MultitaskHead(self.heads, ch))
#             if i < num_stacks - 1:
#                 fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
#                 score_.append(nn.Conv2d(self.num_classes, ch, kernel_size=1))
#         self.hg = nn.ModuleList(hg)
#         self.res = nn.ModuleList(res)
#         self.fc = nn.ModuleList(fc)
#         self.score = nn.ModuleList(score)
#         self.fc_ = nn.ModuleList(fc_)
#         self.score_ = nn.ModuleList(score_)
#
#         for head in self.heads:
#             num_classes = self.heads[head]
#             finalcls = nn.Sequential(
#                 DecoderBlock(ch, 64),
#                 nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,  # TODO intermediate channel
#                                    output_padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Conv2d(32, 32, kernel_size=3, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
#             )
#             self.__setattr__(head, finalcls)
#
#     def _make_residual(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(  # TODO dwonsample wo BN??
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion)
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def _make_fc(self, inplanes, planes):
#         conv = nn.Conv2d(inplanes, planes, kernel_size=1)
#         bn = nn.BatchNorm2d(planes)
#         return nn.Sequential(conv, bn, self.relu)
#
#     def forward(self, x):
#         out = []
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.layer1(x)
#         x = self.maxpool(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#
#         for i in range(self.num_stacks):
#             y = self.hg[i](x)
#             y = self.res[i](y)
#             y = self.fc[i](y)
#             score = self.score[i](y)
#             if len(self.heads) == 1:
#                 out.append(score)
#             else:
#                 out.append(torch.chunk(score, self.num_classes, dim=1))
#             if i < self.num_stacks - 1:
#                 fc_ = self.fc_[i](y)
#                 score_ = self.score_[i](score)
#                 x = x + fc_ + score_
#
#         if len(self.heads) == 1:
#             f_out = self.__getattr__(list(self.heads.keys())[0])(y)
#         else:
#             f_out = []
#             for head in self.heads:
#                 f_out.append(self.__getattr__(head)(y))
#
#         return out[::-1], f_out  # outputs

class StackHourglass(BaseModel):
    def __init__(self, block, heads, depth, num_stacks, num_blocks):
        super(StackHourglass, self).__init__()
        self.inplanes = 64  # keep updating
        self.num_feats = 128
        self.block = eval(block)
        self.heads = heads
        self.num_classes = sum(heads.values())
        self.num_stacks = num_stacks

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(self.block, self.inplanes, 1)  # self.inplanes as plane in the function
        self.layer2 = self._make_residual(self.block, self.inplanes, 1)
        self.layer3 = self._make_residual(self.block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats * self.block.expansion  # 256
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(self.num_stacks):  # number of stacked-hourglass
            hg.append(Hourglass(self.block, num_blocks, self.num_feats, depth))
            res.append(self._make_residual(self.block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(MultitaskHead(self.heads, ch))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(self.num_classes, ch, kernel_size=1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

        for head in self.heads:
            num_classes = self.heads[head]
            finalcls = nn.Sequential(
                DecoderBlock(ch, 64),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,  # TODO intermediate channel
                                   output_padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
            )
            self.__setattr__(head, finalcls)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(  # TODO dwonsample wo BN??
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, planes):
        conv = nn.Conv2d(inplanes, planes, kernel_size=1)
        bn = nn.BatchNorm2d(planes)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        out1 = []
        out2 = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            if len(self.heads) == 1:
                out.append(score)
            else:
                out1.append(score[:, :2, :, :])
                out2.append(score[:, 2:, :, :])
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        if len(self.heads) == 1:
            f_out = self.__getattr__(list(self.heads.keys())[0])(y)
        else:
            f_out = self.__getattr__('seg')(y)

        return [out1[::-1], f_out], out2[::-1]  # outputs


class MTLStackHourglass(BaseModel):
    def __init__(self, block, heads, depth, num_stacks, num_blocks):
        super(MTLStackHourglass, self).__init__()
        self.inplanes = 64  # keep updating
        self.num_feats = 128
        self.block = eval(block)
        self.heads = eval(heads)
        self.num_stacks = num_stacks

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(self.block, self.inplanes, 1)  # self.inplanes as plane in the function
        self.layer2 = self._make_residual(self.block, self.inplanes, 1)
        self.layer3 = self._make_residual(self.block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats * self.block.expansion  # 256
        hg = []
        res_1, fc_1, score_1, fc_1_, score_1_ = [], [], [], [], []
        res_2, fc_2, score_2, fc_2_, score_2_ = [], [], [], [], []
        for i in range(self.num_stacks):  # number of stacked-hourglass
            hg.append(MTLHourglass(self.block, num_blocks, self.num_feats, depth))

            res_1.append(self._make_residual(self.block, self.num_feats, num_blocks))
            res_2.append(self._make_residual(self.block, self.num_feats, num_blocks))

            fc_1.append(self._make_fc(ch, ch))
            fc_2.append(self._make_fc(ch, ch))

            score_1.append(nn.Sequential(
                nn.Conv2d(ch, ch // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 4, self.heads[0], kernel_size=1)))
            score_2.append(nn.Sequential(
                nn.Conv2d(ch, ch // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 4, self.heads[1], kernel_size=1)))

            if i < num_stacks - 1:
                fc_1_.append(nn.Conv2d(ch, ch, kernel_size=1))
                fc_2_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_1_.append(nn.Conv2d(self.heads[0], ch, kernel_size=1))
                score_2_.append(nn.Conv2d(self.heads[1], ch, kernel_size=1))

        self.hg = nn.ModuleList(hg)

        self.res_1 = nn.ModuleList(res_1)
        self.fc_1 = nn.ModuleList(fc_1)
        self.score_1 = nn.ModuleList(score_1)
        self.fc_1_ = nn.ModuleList(fc_1_)
        self.score_1_ = nn.ModuleList(score_1_)

        self.res_2 = nn.ModuleList(res_2)
        self.fc_2 = nn.ModuleList(fc_2)
        self.score_2 = nn.ModuleList(score_2)
        self.fc_2_ = nn.ModuleList(fc_2_)
        self.score_2_ = nn.ModuleList(score_2_)

        self.finalcls_1 = self._make_head(ch, self.heads[0])
        self.finalcls_2 = self._make_head(ch, self.heads[1])

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(  # TODO dwonsample wo BN??
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, planes):
        conv = nn.Conv2d(inplanes, planes, kernel_size=1)
        bn = nn.BatchNorm2d(planes)
        return nn.Sequential(conv, bn, self.relu)

    def _make_head(self, inplanes, planes):
        return nn.Sequential(
            # DecoderBlock(inplanes, 64, stride=2),
            # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(32, 32, kernel_size=3),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(32, planes, kernel_size=2, padding=1)
            DecoderBlock(inplanes, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, planes, kernel_size=2, stride=2)
        )

    def forward(self, x):
        out_1 = []
        out_2 = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y1, y2 = self.hg[i](x)
            y1, y2 = self.res_1[i](y1), self.res_2[i](y2)
            y1, y2 = self.fc_1[i](y1), self.fc_2[i](y2)
            score_1, score_2 = self.score_1[i](y1), self.score_2[i](y2)

            out_1.append(score_1)
            out_2.append(score_2)

            if i < self.num_stacks - 1:
                fc_1_, fc_2_ = self.fc_1_[i](y1), self.fc_2_[i](y2)
                score_1_, score_2_ = self.score_1_[i](score_1), self.score_2_[i](score_2)
                x = x + fc_1_ + score_1_ + fc_2_ + score_2_

        out_1.append(self.finalcls_1(y1))
        out_2.append(self.finalcls_2(y2))

        return out_1, out_2


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


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth  # hourglass order
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = x  # n order, starting from the outermost todo cancel up1 (self.hg[n - 1][0](x))
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class MTLHourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(MTLHourglass, self).__init__()
        self.depth = depth  # hourglass order
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(4):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)  # n order, starting from the outermost
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2_1, low2_2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2_1 = self.hg[n - 1][4](low1)
            low2_2 = self.hg[n - 1][5](low1)
        low3_1 = self.hg[n - 1][2](low2_1)
        low3_2 = self.hg[n - 1][3](low2_2)
        up2_1 = F.interpolate(low3_1, scale_factor=2)
        up2_2 = F.interpolate(low3_2, scale_factor=2)
        out_1 = up1 + up2_1
        out_2 = up1 + up2_2
        return out_1, out_2

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


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
            nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inplanes // 4, inplanes // 4, kernel_size=3, stride=stride, padding=1,
                               output_padding=stride - 1, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 4, planes, kernel_size=1, bias=False),
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


class MultitaskHead(nn.Module):
    def __init__(self, heads, inplanes):
        super(MultitaskHead, self).__init__()
        self.heads = heads
        heads_conv = []
        for head in self.heads:
            num_classes = self.heads[head]
            heads_conv.append(
                nn.Sequential(
                    nn.Conv2d(inplanes, inplanes // 4, kernel_size=3, padding=1),  # TODO head's intermediate chennel
                    nn.ReLU(inplace=True),
                    nn.Conv2d(inplanes // 4, num_classes, kernel_size=1)))
        self.heads_conv = nn.ModuleList(heads_conv)

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads_conv], dim=1)


if __name__ == '__main__':
    model = StackHourglass("SCBasicBlock", {"seg": 2, "conn": 5}, depth=3, num_stacks=2, num_blocks=3)
    input = torch.randn(1, 3, 512, 512)
    out = model(input)
    print(model)
    print(out[1][2].shape)

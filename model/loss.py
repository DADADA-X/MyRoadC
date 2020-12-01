import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import utils, functional as F


def soft_iou_loss(pred, gt):
    b, c, h, w = pred.shape
    pred_ = F.softmax(pred, dim=1)
    gt_ = to_one_hot_var(gt, c).float()

    intersection = pred_ * gt_
    intersection = intersection.view(b, c, -1).sum(2)  # b, c, h, w => b, c
    union = pred_ + gt_ - (pred_ * gt_)
    union = union.view(b, c, -1).sum(2)
    loss = intersection / (union + 1e-8)
    return -torch.mean(loss)


def to_one_hot_var(tensor, num_class):
    n, c, h, w = tensor.shape
    one_hot = tensor.new(n, num_class, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.to(torch.int64), 1)
    return one_hot


def balanced_ce_loss(pred, gt):
    b, c, h, w = pred.shape
    pred_ = F.log_softmax(pred, dim=1)
    gt_ = gt.squeeze().long()
    if len(gt_.shape) < 3:
        gt_ = gt_.unsqueeze(0)

    weight = torch.ones(c).cuda()  # todo weight
    nll_loss = nn.NLLLoss(weight=weight, ignore_index=255)
    return nll_loss(pred_, gt_)


def mse_loss(pred, gt):
    criterion = torch.nn.MSELoss()
    pred_ = pred.sigmoid()
    return criterion(pred_, gt)
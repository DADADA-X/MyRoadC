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

    weight = []
    num_total = b * h * w
    for i in range(c):
        num = torch.eq(gt, i).float().sum()
        p = num / num_total
        weight.append(1 / torch.log(1.02 + p))
    weight = torch.Tensor(weight).cuda()
    # weight = torch.ones(c).cuda()  # todo weight
    nll_loss = nn.NLLLoss(weight=weight, ignore_index=255)
    return nll_loss(pred_, gt_)


def mse_loss(pred, gt):
    criterion = torch.nn.MSELoss()
    pred_ = pred.sigmoid()
    return criterion(pred_, gt)


def focal_loss(pred, gt, step):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    num_pos = pos_inds.float().sum()
    num_neg = neg_inds.float().sum()

    annealing_step = 20  # TODO annealing function.

    zeta = 0.5 * (1. + np.cos(np.pi * step / annealing_step)) if step <= annealing_step else 0. # Cosine
    # zeta = np.maximum(1 - step / annealing_step, 0)  # Linear

    neg_weights = torch.pow(1 - gt[neg_inds], 4)
    c = 1.02
    p_pos = num_pos / (num_pos + num_neg)
    pos_weight = 1 / torch.log(c + p_pos)  # TODO pos weight need to be fixed.

    loss = 0
    pred.sigmoid_()
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_pred = pos_pred.clamp(min=0.0001, max=1)
    neg_pred = neg_pred.clamp(min=0, max=0.9999)

    pos_loss = torch.log(pos_pred) * (torch.pow(1 - pos_pred, 2) + zeta * (1 - torch.pow(1 - pos_pred, 2)))
    neg_loss = torch.log(1 - neg_pred) * (torch.pow(neg_pred, 2) + zeta * (1 - torch.pow(neg_pred, 2))) * neg_weights

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss
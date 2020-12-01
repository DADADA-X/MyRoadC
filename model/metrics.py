import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def mIoU(pred, gt):
    "mean IoU of multi-class seg"
    with torch.no_grad():
        b, c, h, w = pred.shape
        pred_ = pred.cpu().numpy()
        pred_ = np.argmax(pred_, axis=1)[:, np.newaxis, :, :]
        gt_ = gt.cpu().numpy()
        confusion_matrix = _generate_matrix(gt_.astype(np.int8), pred_.astype(np.int8), num_class=c)
        miou = _Class_IOU(confusion_matrix)
    return miou.mean()


def rIoU(pred, gt):
    "road segmentation IoU"
    with torch.no_grad():
        b, c, h, w = pred.shape
        pred_ = pred.cpu().numpy()
        pred_ = np.argmax(pred_, axis=1)[:, np.newaxis, :, :]
        gt_ = gt.cpu().numpy()
        confusion_matrix = _generate_matrix(gt_.astype(np.int8), pred_.astype(np.int8), num_class=c)
        miou = _Class_IOU(confusion_matrix)
    return miou[1]


def MSE(pred, gt):
    with torch.no_grad():
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()
        MSE = np.sqrt(((pred - gt) ** 2).sum() / gt.size)
    return MSE


# ======================================================== #
#                        Utilities                         #
# ======================================================== #


def _generate_matrix(gt_image, pre_image, num_class):
    mask = (gt_image >= 0) & (gt_image < num_class)
    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix


def _Class_IOU(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    MIoU = intersection / (union + 1e-8)
    MIoU[np.where(union==0)] = 1
    return MIoU
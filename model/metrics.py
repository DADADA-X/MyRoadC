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


def relaxed_IoU(pred, gt):
    with torch.no_grad():
        b, c, h, w = pred.shape
        if c == 1:
            pred_ = pred.sigmoid().cpu().numpy()
        else:
            pred_ = torch.argmax(pred, dim=1).cpu().numpy()[:, np.newaxis, :, :]
        gt_ = gt.cpu().numpy()
        kernel = np.ones((4, 4))
        dilated_gt = []
        for i in range(b):
            tmp_gt = cv2.dilate(gt_[i][0], kernel)
            dilated_gt.append(tmp_gt)
        dilated_gt = np.array(dilated_gt)[:, np.newaxis, :, :]
        intersection = np.logical_and(dilated_gt, pred_)
        union = np.logical_or(gt_, pred_)
        if union.sum() == 0:
            iou_score = 1
        else:
            iou_score = intersection.sum() / union.sum()
    return iou_score


def MSE(pred, gt):
    with torch.no_grad():
        pred = pred.sigmoid().cpu().numpy()
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


if __name__ == '__main__':
    inputs = torch.randn(1, 2, 650, 650)
    target = torch.randint(0, 2, (1, 1, 650, 650)).float()
    print(relaxed_IoU(inputs, target))
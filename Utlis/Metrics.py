# -*- coding: utf-8 -*-
# @File：Metrics.py
# @Time：2025/3/4 15:01
# @Author：yangchen0372
import numpy as np
import torch


class BCDMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self,preds,masks):
        for pred,mask in zip(preds,masks):
            self.compute_confusion_matrix(pred.flatten(),mask.flatten())

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def compute_confusion_matrix(self, pred, mask):
        valid_mask = (mask >= 0) & (mask < self.num_classes)
        hist =np.bincount(
            self.num_classes * mask[valid_mask].astype(int) + pred[valid_mask].astype(int),
            minlength=self.num_classes**2
        ).reshape(self.num_classes,self.num_classes)
        self.confusion_matrix+=hist

    def cm2F1(self):
        hist = self.confusion_matrix
        tp = hist[1, 1]
        fn = hist[1, 0]
        fp = hist[0, 1]
        tn = hist[0, 0]
        # recall
        recall = tp / (tp + fn + np.finfo(np.float32).eps)
        # precision
        precision = tp / (tp + fp + np.finfo(np.float32).eps)
        # F1 score
        f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
        return f1 * 100

    def cm2score(self):
        hist = self.confusion_matrix
        tp = hist[1, 1]
        fn = hist[1, 0]
        fp = hist[0, 1]
        tn = hist[0, 0]
        # acc
        oa = (tp + tn) / (tp + fn + fp + tn + np.finfo(np.float32).eps)
        # recall
        recall = tp / (tp + fn + np.finfo(np.float32).eps)
        # precision
        precision = tp / (tp + fp + np.finfo(np.float32).eps)
        # F1 score
        f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
        # IoU
        iou = tp / (tp + fp + fn + np.finfo(np.float32).eps)
        # pre
        pre = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (tp + fp + tn + fn) ** 2
        # kappa
        kappa = (oa - pre) / (1 - pre)
        #
        kappa *= 100
        iou *= 100
        f1 *= 100
        oa *= 100
        recall *= 100
        precision *= 100
        pre *= 100
        #
        score_dict = {
            'Kappa': float(round(kappa, 2)), 'F1': float(round(f1, 2)),
            'precision': float(round(precision, 2)), 'recall': float(round(recall, 2)),
            'OA': float(round(oa, 2)), 'IoU': float(round(iou, 2)),
        }
        return score_dict


# bce_metrics = BCDMetrics(num_classes=2)
# input = torch.randn(50,50).sigmoid()
# pred_class =  torch.where(input > 0.5, torch.ones_like(input), torch.zeros_like(input)).float()
# label = torch.randint(0,2,(50,50))
# bce_metrics.update(
#                         preds=pred_class.cpu().numpy(),
#                         masks=label.detach().cpu().numpy()
#                     )
# print(bce_metrics.cm2score())
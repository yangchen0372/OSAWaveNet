# -*- coding: utf-8 -*-
# @File：Losses.py
# @Time：2025/3/5 09:12
# @Author：yangchen0372
# 该脚本的loss主要是针对是bce方式
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        '''
        :param inputs:经过sigmoid处理的预测值
        :param targets:仅包含包含类索引
        '''
        bce = F.binary_cross_entropy(inputs, targets)  # 等同于nn.BCELoss
        return bce

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, eps=1e-8):
        '''
        :param inputs:经过sigmoid处理的预测值
        :param targets:仅包含包含类索引
        '''
        inter = (inputs * targets).sum()
        dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
        return 1 - dice

class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, targets, eps=1e-8):
        '''
        :param inputs:经过sigmoid处理的预测值
        :param targets:仅包含包含类索引
        '''
        inter = (inputs * targets).sum() # 交集
        union = (inputs + targets).sum() # 并集(但是重叠地方为2)
        iou = (inter + eps) / (union - inter + eps) # 交集/真实的并集
        return 1 - iou

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, eps=1e-8):
        '''
        :param inputs:经过sigmoid处理的预测值
        :param targets:仅包含包含类索引
        '''
        #BCE Loss
        bce = F.binary_cross_entropy(inputs, targets)  # 等同于nn.BCELoss
        # Dice Loss
        inter = (inputs * targets).sum()
        dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
        dice = 1 - dice
        return bce + dice

# debug 添加新的loss时 在这里留下样例
# input = torch.randn(2,1,3,2).sigmoid()
# print(input)
# target = torch.randint(0,2,(2,1,3,2)).float()
# print(target)
# bceloss = BCEloss()
# diceloss = DiceLoss()
# iouloss = IoULoss()
# dicebceloss = DiceBCELoss()
# print('BCEloss:',bceloss(input,target).item())
# print('DiceLoss:',diceloss(input,target).item())
# print('IoULoss:',iouloss(input,target).item())
# print('DiceBCELoss:',dicebceloss(input,target).item())
# print('BCEloss+DiceLoss:',bceloss(input,target).item()+diceloss(input,target).item())
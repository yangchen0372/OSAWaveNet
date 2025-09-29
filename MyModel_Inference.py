# -*- coding: utf-8 -*-
# @File：ChangeFormer_LEVIR256_Train.py
# @Time：2025/3/4 13:03
# @Author：yangchen0372
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from Dataset_Helper.LEVIR256 import LEVIR256
from Dataset_Helper.HRCUS256 import HRCUS256
from Dataset_Helper.SYSU256 import SYSU256
from tqdm import tqdm
from Utlis.Tools import Logger
from Utlis.Metrics import BCDMetrics
from Utlis.Losses import DiceBCELoss
from Model.OSAWaveNet.MyModelv4 import MyModel
import numpy as np
import cv2

if __name__ == '__main__':


    # 训练超参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_backbone_path = r'./Pretrained_model/resnet18-5c106cde.pth'   # 项目相对路径
    model = MyModel(out_channels=1, pretrained_backbone_path=pretrained_backbone_path).to(device) # 模型声明

    criter = DiceBCELoss()  # 损失函数
    bce_metrics = BCDMetrics(num_classes=2)


    # LEVIR256
    state_dict = torch.load(r'./LEVIR256.pt',map_location=device,weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    TEST_DATASET = LEVIR256(flag='test')
    TEST_DATALOADER = DataLoader(dataset=TEST_DATASET,batch_size=1,shuffle=True,num_workers=8)

    # SYSU256
    state_dict = torch.load(r'./SYSU256.pt',map_location=device,weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    TEST_DATASET = SYSU256(flag='test')
    TEST_DATALOADER = DataLoader(dataset=TEST_DATASET,batch_size=1,shuffle=True,num_workers=8)

    # HRCUS256
    state_dict = torch.load(r'./HRCUS256.pt',map_location=device,weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    TEST_DATASET = HRCUS256(flag='test')
    TEST_DATALOADER = DataLoader(dataset=TEST_DATASET,batch_size=1,shuffle=True,num_workers=8)

    # test
    print('==> Testing model..')
    model.eval()
    with torch.no_grad():
        for iter, batch in enumerate(TEST_DATALOADER,start=1):
            # 1.读取数据
            time1, time2, label, file_name = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3]
            # 2.前向传播（预测）
            pred = model(time1, time2)
            # 3.计算损失
            loss = criter(pred, label)
            # 4.计算指标
            pred_class = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred)).float()
            bce_metrics.update(
                preds=pred_class.cpu().numpy(),
                masks=label.detach().cpu().numpy()
            )
            print('\rEpoch-Test {}\{}'.format(iter, len(TEST_DATALOADER)), end='')
            pred_class *= 255
            pred_class = pred_class.cpu().numpy().astype(np.uint8).squeeze(axis=(0)).transpose(1, 2, 0).astype(np.uint8)
            # cv2.imwrite(os.path.join(save_path, file_name[0]), pred_class, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # cv2.imwrite(os.path.join(save_path,'out.png'), pred_class, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 用于骨干网络特征图显示

    epoch_metrics = bce_metrics.cm2score()
    print('test metrics: ' + str(epoch_metrics))



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
from Dataset_Helper.SYSU256 import SYSU256
from tqdm import tqdm
from Utlis.Tools import Logger
from Utlis.Metrics import BCDMetrics
from Utlis.Losses import DiceBCELoss
from Model.OSAWaveNet.MyModelv4 import MyModel

if __name__ == '__main__':
    # 各类路径
    PROJECT_HOME = os.getcwd()  # 根目录路径
    PROJECT_RUN_TIME = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) # 项目运行时间
    LOG_DIR = os.path.join(PROJECT_HOME, 'TrainLog', 'EXPMODEL_SYSU256', PROJECT_RUN_TIME) # 日志保存路径

    # 创建日志文件、设置随机种子
    os.makedirs(LOG_DIR,exist_ok=True)
    logger = Logger(log_dir=LOG_DIR)
    logger.setRandomSeed(seed=None)
    logger.write(log_str='TIME : {}'.format(PROJECT_RUN_TIME))

    # 训练超参数
    batch_szie = 16  # 一次输入几张图像
    num_workers = 8  # 有关数据读取的快慢
    train_epoch = 100
    learning_rate = 1e-4 # origin paper
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_backbone_path = r'./Pretrained_model/resnet18-5c106cde.pth'   # 项目相对路径
    # pretrained_backbone_path = r'./Pretrained_model/mobilenet_v2-b0353104.pth'   # 项目相对路径
    model = MyModel(out_channels=1, pretrained_backbone_path=pretrained_backbone_path).to(device) # 模型声明
    criter = DiceBCELoss()  # 损失函数
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
    bce_metrics = BCDMetrics(num_classes=2)

    # 数据集
    print('==> Preparing data..')
    TRAIN_DATASET = SYSU256(flag='train')
    TEST_DATASET = SYSU256(flag='test')
    TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET,batch_size=batch_szie,shuffle=True,num_workers=num_workers)
    TEST_DATALOADER = DataLoader(dataset=TEST_DATASET,batch_size=1,shuffle=False,num_workers=num_workers)

    # 训练开始
    print('==> Training and Testing model..')
    best_iou = 0
    for current_epoch in range(1, train_epoch + 1):

        # Train One Epoch
        epoch_start_time = time.time()
        bce_metrics.reset()

        # train
        model.train()
        train_loss = 0
        for iter, batch in enumerate(TRAIN_DATALOADER, start=1):
            # 1.读取数据
            time1, time2, label, file_name = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3]
            # 2.前向传播（预测）
            pred = model(time1, time2)
            # 3.计算损失
            loss = criter(pred, label)
            # 4.反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 5.计算指标
            train_loss += loss.item()
            print('\r[{}] Epoch-Train {}\{} -> current loss:{} his_best_iou:{}'.format(current_epoch, iter, len(TRAIN_DATALOADER), loss.item(), best_iou), end='')
        # Finished One Train Epoch
        torch.cuda.synchronize()

        # test
        if current_epoch % 1 == 0 or current_epoch == train_epoch:
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
                    print('\r[{}] Epoch-Test {}\{} his_best_iou:{}'.format(current_epoch, iter, len(TEST_DATALOADER),best_iou), end='')
                epoch_metrics = bce_metrics.cm2score()
                if epoch_metrics['IoU'] > best_iou:
                    best_iou = epoch_metrics['IoU']
                    logger.save_checkpoint(model, 'EXPMODEL_SYSU256_best.pt')
                logger.write(
                    log_str='test metrics: ' + str(bce_metrics.cm2score())
                )
                logger.save_checkpoint(model, 'EXPMODEL_SYSU256_{}.pt'.format(current_epoch))


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF


class VisualTool:
    def __init__(self):
        pass

    @staticmethod
    def getTopK_features(x, topK=10):
        # x.shape: [B,C,H,W]
        per_batch_channel_mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)  # [B,1,1,1] 每个样本通道特征均值
        per_batch_channel_std = torch.mean(torch.pow(x - per_batch_channel_mean, 2), dim=(2, 3))  # [B,1,1,1] 每个样本通道特征方差
        x_channel_sample_index = torch.topk(per_batch_channel_std, dim=1, k=topK)[1]
        x_sample = x[torch.arange(x_channel_sample_index.shape[0]).unsqueeze(1), x_channel_sample_index, :, :]
        return x_sample

    @staticmethod
    def visualize_channel_feature(features, normalize='min_max', numCols=8, resize=(256,256),get_TopK=False):

        # init
        # getTopK_features
        if get_TopK:
            features = VisualTool.getTopK_features(features, topK=numCols*2)
        #
        features = F.interpolate(features.detach().clone(), size=resize, mode='nearest')
        b, c, h, w = features.shape
        numRows = int((c*b) / numCols)       # 输入通道应当可以被整除
        eps = 1e-8

        # # 对比
        # features1 = features.reshape(b*c, 1, h, w)
        # grid = make_grid(features1,normalize=True)
        # if not isinstance(grid, list):
        #     imgs = [grid]
        # fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        # for i, img in enumerate(imgs):
        #     img = img.detach()
        #     img = TF.to_pil_image(img)
        #     axs[0, i].imshow(np.asarray(img))
        #     axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        # plt.show()
        # # 对比

        # normalize
        if normalize == 'min_max':  # 将不同批次的特征值归一化到0-1
            for i in range(b):
                min_val = features[i].min()
                max_val = features[i].max()
                features[i,:,:,:] = (features[i,:,:,:] - min_val) / (max_val - min_val + eps)

        # 拼接
        imgs = []   # 用于存放每个通道的特征图
        gap = 5     # 控制行列间距
        col_gap = np.ones((h, gap), np.uint8)  # 列间距图像
        row_gap = np.ones((gap, w * numCols + gap * (numCols - 1)), np.uint8)  # 行间距图像
        features = features.reshape(b * c, h, w)
        for i in range(b*c):
            imgs.append(features[i].cpu().numpy())
        img_rows = []
        for i in range(numRows):
            img_row = []
            for j in range(numCols):
                index = i * numCols + j  # 获取通道图像索引号
                img = imgs[index]
                img_row.append(img)
                img_row.append(col_gap)  # 插入列间距图像
            img_row = np.hstack(img_row[:-1])  # 拼凑一行图像
            img_rows.append(img_row)
            img_rows.append(row_gap)  # 插入行间距图像
        features = np.vstack(img_rows[:-1])  # 将每一行图像,按垂直方向拼接
        plt.figure(figsize=(14, 7))
        plt.axis('off')
        plt.imshow(features)
        # plt.imshow(features,cmap='gray')
        plt.show()






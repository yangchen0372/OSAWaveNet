# -*- coding: utf-8 -*-
# @File：LEVIR256.py
# @Time：2025/2/25 17:22
# @Author：yangchen0372

import os
import cv2
import random
import pandas
from glob import glob
import pandas as pd
from torch.utils.data import Dataset
from Dataset_Helper.LEVIR256_CFG import LEVIR256_DATASET_CFG
from Utlis.Tools import Tools

class LEVIR256(Dataset):
    def __init__(self,flag='train'):
        '''
        LEVIR-CD数据集期待的目录为:
        dataset_root/
            A/
            B/
            label/
            list/
        :param root:数据集根目录路径
        :param flag:指定train、val或test
        '''
        data_root = os.path.join(LEVIR256_DATASET_CFG['data_root'],flag)
        self.time1_image_list = glob(os.path.join(data_root,'time1','*.png'))
        self.time2_image_list = glob(os.path.join(data_root,'time2','*.png'))
        self.label_image_list = glob(os.path.join(data_root,'label','*.png'))

        self.FLAG = flag
        self.DATA_AUGMENT = LEVIR256_DATASET_CFG['transform'][flag]
        self.DATA_RESIZE = LEVIR256_DATASET_CFG['transform']['resize']
        self.DATA_NORMALIZE = LEVIR256_DATASET_CFG['transform']['normalize']
        self.DATA_TO_TENSOR = LEVIR256_DATASET_CFG['transform']['to_tensor']

    def __getitem__(self, index):
        # image path
        time1_image_path = self.time1_image_list[index]
        time2_image_path = self.time2_image_list[index]
        label_image_path = self.label_image_list[index]
        file_name = os.path.basename(label_image_path)

        # read image
        time1 = cv2.imread(time1_image_path,cv2.IMREAD_UNCHANGED)
        time1 = cv2.cvtColor(time1, cv2.COLOR_BGR2RGB)
        time2 = cv2.imread(time2_image_path,cv2.IMREAD_UNCHANGED)
        time2 = cv2.cvtColor(time2, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_image_path,cv2.IMREAD_UNCHANGED)
        # Tools.visualize_image(time1)
        # Tools.visualize_image(time2)
        # Tools.visualize_image(label)

        # data resize
        if self.DATA_RESIZE is not None:
            transformed = self.DATA_RESIZE(image=time1,image1=time2,mask=label)
            time1, time2, label = transformed['image'], transformed['image1'], transformed['mask']
            # Tools.visualize_image(time1)
            # Tools.visualize_image(time2)
            # Tools.visualize_image(label)

        # data augmentation only trainset
        if self.DATA_AUGMENT is not None and self.FLAG == 'train':
            transformed = self.DATA_AUGMENT(image=time1,image1=time2,mask=label)
            time1, time2, label = transformed['image'], transformed['image1'], transformed['mask']
            # Tools.visualize_image(time1)
            # Tools.visualize_image(time2)
            # Tools.visualize_image(label)

        # normalization
        if self.DATA_NORMALIZE is not None:
            transformed = self.DATA_NORMALIZE(image=time1,image1=time2)
            time1, time2 = transformed['image'], transformed['image1']
            # Tools.visualize_image(time1)
            # Tools.visualize_image(time2)
            # Tools.visualize_image(label)

        # to tensor
        if self.DATA_TO_TENSOR is not None:
            label = label[:,:,None]
            transformed = self.DATA_TO_TENSOR(image=time1,image1=time2,mask=label)
            time1, time2, label = transformed['image'], transformed['image1'], transformed['mask']
            label = label.float().div(255)  # 二值化标签0、1

        return time1, time2, label, file_name

    def __len__(self):
        return len(self.label_image_list)

# if __name__ == '__main__':
#     import random
#     from torch.utils.data import DataLoader
#     dataset = LEVIR256(flag='train')
#     TRAIN_DATALOADER = DataLoader(dataset=dataset,batch_size=16,shuffle=True,num_workers=8)
#     # dataset[random.randint(0,len(dataset))]
#     for time1, time2, label, file_name in TRAIN_DATALOADER:
#         print(time1.shape)
#         print(time2.shape)
#         print(label.shape)
#         print(file_name)
#         break
# -*- coding: utf-8 -*-
# @File：Tools.py
# @Time：2025/2/26 09:50
# @Author：yangchen0372
import os
import glob
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
class Logger():
    def __init__(self, log_dir):
        '''
        :param log_dir: 日志输出目录, 该目录用于存放本次训练的文本记录与tensorboard记录
        '''
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_path = os.path.join(log_dir, 'log.txt')

    def write(self, log_str, is_print=True):
        '''
        :param log_str: 日志文本
        :param is_print: 是否在控制台打印该文本
        '''
        f = open(self.log_path, 'a', encoding='utf-8')
        if isinstance(log_str, str):
            f.write(str(log_str) + '\n')
            if is_print:
                print(log_str)
        elif isinstance(log_str, list):
            for line_str in log_str:
                f.write(line_str + '\n')
                if is_print:
                    print(line_str)
        f.close()

    def save_checkpoint(self, model ,file_name):
        torch.save(
            model.state_dict(),
            str(os.path.join(self.log_dir, file_name))
        )

    def setRandomSeed(self, seed=1241211017, is_print=True):
        if seed == None:
            seed = random.randint(1, 4294967295)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        log_str = 'seed : {}'.format(seed)
        self.write(log_str,is_print=is_print)
        return seed

class Tools:
    @staticmethod
    def visualize_image(image):
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(image)
        plt.show()

    @staticmethod
    def compute_mean_std(match_str):
        """Compute the mean and std of dataset images.

        Parameter:
            dataset_name(str): name of the specified dataset.

        Return:
            means(list): means in three channel(RGB) of images in :obj:`images_dir`
            stds(list): stds in three channel(RGB) of images in :obj:`images_dir`
        """

        image_list = glob.glob(match_str)
        num_image = len(image_list)
        # calculate means and std
        means = [0, 0, 0]
        stds = [0, 0, 0]


        if not num_image:
            raise RuntimeError(f'No input file found in {match_str}, make sure you put your images there')
        for index in tqdm(range(num_image)):
            img_file = image_list[index]
            img = cv2.imread(img_file,cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img)
            img = img.astype(np.float32) / 255.
            for i in range(3):
                means[i] += img[:, :, i].mean()
                stds[i] += img[:, :, i].std()

        means = np.asarray(means) / num_image
        stds = np.asarray(stds) / num_image

        print("normMean = {}".format(means))
        print("normStd = {}".format(stds))

        return means, stds

# import os
# import shutil
# import pandas as pd
# from tqdm import tqdm
# file_list = pd.read_csv(r'D:\Dataset_Root\Change Detection\LEVIR256\list\test.txt',sep=' ',header=None)[0].to_list()
# for file in tqdm(file_list):
#     imageA_path = os.path.join(r'D:\Dataset_Root\Change Detection\LEVIR256\A',file)
#     imageB_path = os.path.join(r'D:\Dataset_Root\Change Detection\LEVIR256\B',file)
#     label_path = os.path.join(r'D:\Dataset_Root\Change Detection\LEVIR256\label',file)
#     shutil.move(imageA_path,r'D:\Dataset_Root\Change Detection\LEVIR256\test\A')
#     shutil.move(imageB_path,r'D:\Dataset_Root\Change Detection\LEVIR256\test\B')
#     shutil.move(label_path,r'D:\Dataset_Root\Change Detection\LEVIR256\test\gt')


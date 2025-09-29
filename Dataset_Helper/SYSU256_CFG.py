# -*- coding: utf-8 -*-
# @File：LEVIR256_CFG.py
# @Time：2025/2/26 08:36
# @Author：yangchen0372
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1' # 关闭自动更新  应当放在任何导入albumentations库前
import albumentations as A
SYSU_DATASET_CFG = {
    'data_root': os.path.abspath(r'.\Dataset_Demo\SYSU-CD'),    # for 5070ti laptop
    'classes_name':['unchanged','change'],
    'classes_mask':[          0,     255],  # 需要对label整除255
    'transform': {
        'train': A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(45, p=0.5),
        ], additional_targets={'image1':'image'} ),
        'val': None,
        'test': None,
        'resize':A.Compose([
            A.Resize(height=256, width=256),
        ], additional_targets={'image1':'image'} ),
        'normalize': A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ], additional_targets={'image1':'image'} ),
        'to_tensor': A.Compose([
            A.pytorch.transforms.ToTensorV2(transpose_mask=True,p=1.0),
        ], additional_targets={'image1':'image'} ),
    }
 }
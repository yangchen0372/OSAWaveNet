# # -*- coding: utf-8 -*-
# # @File：Split_Whole_image.py
# # @Time：2025/3/4 10:20
# # @Author：yangchen0372
# import os
# from multiprocessing.util import spawnv_passfds
#
# import cv2
#
# root = r'D:\Dataset_Root\temp\WHUCD'
# time1 = os.path.join(root, '2012_test.tif')
# time2 = os.path.join(root, '2016_test.tif')
# label = os.path.join(root, 'change_label_test.tif')
# time1 = cv2.imread(time1, cv2.IMREAD_UNCHANGED)
# time2 = cv2.imread(time2, cv2.IMREAD_UNCHANGED)
# label = cv2.imread(label, cv2.IMREAD_UNCHANGED)
# patch_size = 512
# h, w = label.shape[0] // patch_size, label.shape[1] // patch_size
# print('patch_h:', h, ' patch_w:', w)
#
# # 切分
# for i in range(h):
#     for j in range(w):
#         new_time1 = time1[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size,:]
#         new_time2 = time2[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size,:]
#         new_label = label[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
#         cv2.imwrite(os.path.join(r'D:\Dataset_Root\Change Detection\WHU-CD\test\time1',str(i)+'_'+str(j)+'.png'),new_time1,[cv2.IMWRITE_PNG_COMPRESSION,0])
#         cv2.imwrite(os.path.join(r'D:\Dataset_Root\Change Detection\WHU-CD\test\time2',str(i)+'_'+str(j)+'.png'),new_time2,[cv2.IMWRITE_PNG_COMPRESSION,0])
#         cv2.imwrite(os.path.join(r'D:\Dataset_Root\Change Detection\WHU-CD\test\label',str(i)+'_'+str(j)+'.png'),new_label,[cv2.IMWRITE_PNG_COMPRESSION,0])
#
# # 画网格线
# for i in range(h):
#     for j in range(w):
#         time1[i * patch_size:(i + 1) * patch_size, j * patch_size, :] = 255
#         time1[i * patch_size:(i + 1) * patch_size, (j + 1) * patch_size, :] = 255
#         time1[        i*patch_size, j * patch_size:(j + 1) * patch_size, :] = 255
#         time1[(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :] = 255
#
#         time2[i * patch_size:(i + 1) * patch_size, j * patch_size, :] = 255
#         time2[i * patch_size:(i + 1) * patch_size, (j + 1) * patch_size, :] = 255
#         time2[        i*patch_size, j * patch_size:(j + 1) * patch_size, :] = 255
#         time2[(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :] = 255
#
#         label[i * patch_size:(i + 1) * patch_size, j * patch_size] = 255
#         label[i * patch_size:(i + 1) * patch_size, (j + 1) * patch_size] = 255
#         label[        i*patch_size, j * patch_size:(j + 1) * patch_size] = 255
#         label[(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = 255
#
# cv2.imwrite(os.path.join(r'D:\Dataset_Root\Change Detection\WHU-CD\test\time1_demo.png'),time1,[cv2.IMWRITE_PNG_COMPRESSION,9])
# cv2.imwrite(os.path.join(r'D:\Dataset_Root\Change Detection\WHU-CD\test\time2_demo.png'),time2,[cv2.IMWRITE_PNG_COMPRESSION,9])
# cv2.imwrite(os.path.join(r'D:\Dataset_Root\Change Detection\WHU-CD\test\label_demo.png'),label,[cv2.IMWRITE_PNG_COMPRESSION,9])

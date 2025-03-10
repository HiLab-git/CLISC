import cv2
import os
import numpy as np
import SimpleITK as sitk
from copy import deepcopy
from tqdm import tqdm



base_path = '/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/CLIP_label'
input_path = os.path.join(base_path, 'train')
box_path = os.path.join(base_path, 'bounding_box')
output_path = os.path.join(base_path, 'aug_train')

def hidden_cam_mask(img, hidden_list, dilation):
    for x1, y1, x2, y2 in hidden_list:
        assert x1 <= x2, 'x axis bounding box error!'
        assert y1 <= y2, 'y axis bounding box error!'
        x1 = max(0, x1 - dilation)
        y1 = max(0, y1 - dilation)
        x2 = min(img.shape[0], x2 + dilation)
        y2 = min(img.shape[1], y2 + dilation)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)


def hidden_mask(img, hidden_list, min_x, max_x, min_y, max_y, dilation):
    min_y = max(min_y - dilation, 0)
    min_x = max(min_x - dilation, 0)
    max_y = min(img.shape[1], max_y + dilation)
    max_x = min(img.shape[0], max_x + dilation)
    d_x = max_x - min_x
    d_y = max_y - min_y
    for m1, m2, m3, m4 in hidden_list:
        assert m1 <= m3, 'x axis bounding box error!'
        assert m2 <= m4, 'y axis bounding box error!'
        x1 = int(min_x + d_x * m1)
        x2 = int(min_x + d_x * m3)
        y1 = int(min_y + d_y * m2)
        y2 = int(min_y + d_y * m4)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

def hidden_back(img, min_x, max_x, min_y, max_y):
    min_y = max(min_y - dilation, 0)
    min_x = max(min_x - dilation, 0)
    max_y = min(img.shape[1], max_y + dilation)
    max_x = min(img.shape[0], max_x + dilation)
    cv2.rectangle(img, (0, 0), (min_x, min_y), (0, 0, 0), -1)
    cv2.rectangle(img, (0, max_y), (min_x, max_y), (0, 0, 0), -1)
    cv2.rectangle(img, (max_x, 0), (max_x, min_y), (0, 0, 0), -1)
    cv2.rectangle(img, (max_x, max_y), (img.shape[0], img.shape[1]), (0, 0, 0), -1)
    

# file_path = '/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/CLIP_label/bounding_box/positive/BraTS20_Training_006_slice_68_label_1.txt'
# with open(file_path, 'r') as file:
#     lines = list(map(int, file.readline().split()))

# min_y, max_y, min_x, max_x = lines
# print(lines)
phase = ['positive', 'negative']
# /media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/CLIP_label/train/positive/BraTS20_Training_001_slice_33_label_1.tiff
dilation = 10
for phe in phase:
    os.makedirs(os.path.join(output_path, phe), exist_ok=True)
    for file in tqdm(os.listdir(os.path.join(input_path, phe))):
        img = cv2.imread(os.path.join(input_path, phe, file))
        bounding_box_path = os.path.join(box_path, phe, file.replace('.tiff', '.txt'))
        cam_path = bounding_box_path.replace('.txt', '.nii.gz')
 
        if phe == 'positive' and os.path.exists(bounding_box_path):
            cv2.imwrite(os.path.join(output_path, phe, file.replace('.tiff', '_fresh.tiff')), img)
            with open(bounding_box_path, 'r') as txt:
                    line = list(map(int, txt.readline().split()))
            min_x, max_x, min_y, max_y = line
            cam = sitk.GetArrayFromImage(sitk.ReadImage(cam_path))
            x_ratio = 224 / cam.shape[0]
            y_ratio = 224 / cam.shape[1]
            min_x = int(min_x * x_ratio)
            max_x = int(max_x * x_ratio)
            min_y = int(min_y * y_ratio)
            max_y = int(max_y * y_ratio)
            cam = cv2.resize(cam, (224, 224))

            """ method1: 将boundingbox内按均值划分，高位的部分直接遮挡 """
            # threshold = cam[min_x : max_x + 1, min_y : max_y + 1].mean()
            # for i in range(min_x, max_x + 1):
            #     for j in range(min_y, max_y + 1):
            #         if cam[i, j] > threshold: img[i, j] = 0
            
            """ method2: 将boundingbox内的区域分为10段，分别高位的几段直接遮挡 """
            min_one = cam[min_x : max_x + 1, min_y : max_y + 1].min()
            max_one = cam[min_x : max_x + 1, min_y : max_y + 1].max()
            d = max_one - min_one
            k = 8
            threshold = min_one + k / 10 * d
            for i in range(min_x, max_x + 1):
                for j in range(min_y, max_y + 1):
                    if cam[i, j] > threshold: img[i, j] = 0
            
            """ method3: 找一个最大区域 """
            # for i in range(16):
            #     for j in range(16):
            #         pos_x1 = int(min_x + i / 16 * d_x)
            #         pos_x2 = int(min_x + (i + 1) / 16 * d_x)
            #         pos_y1 = int(min_y + j / 16 * d_y)
            #         pos_y2 = int(min_y + (j + 1) / 16 * d_y)
            #         tmp = cam[pos_x1 : pos_x2 + 1, pos_y1 : pos_y2 + 1]
            #         evaluation = tmp.mean()
            #         if evaluation > max_highlight: 
            #             max_highlight = evaluation
            #             hidden_pos = [[pos_x1, pos_y1, pos_x2, pos_y2]]
            # hidden_cam_mask(img, hidden_pos, dilation)


        cv2.imwrite(os.path.join(output_path, phe, file), img)
            
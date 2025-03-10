import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from medpy import metric
from time import time as tt
print('-' * 20)
base_path = "/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/cam/test_layercam_l3"
label_folder = "/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/volume_pre/label/test"
input_path = os.path.join(base_path, 'cam')
output_path = os.path.join(base_path, "binarized_cam")

os.makedirs(output_path, exist_ok=True)

# thresh = 0.27
files = os.listdir(input_path)
# print("calculating")

# alls = []
# scores = []
# for filename in tqdm(files):
#     cam = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_path, filename)))[:, :, :, 0]
#     label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_folder, filename)))
#     out_path = os.path.join(output_path, filename)
#     # if cam.sum(): # 若cam未捕捉到信息则直接不要
#     # t = tt()
#     if cam.sum():
#         cam = (cam - cam.min()) / (cam.max() - cam.min())
#         # """ Method 1: 用cam的均值筛选掉一部分 """
#         # if cam.mean() > 0.025:
#         #     binarized_cam = np.zeros_like(cam)
#         #     binarized_cam[cam > thresh] = 1
#         #     scores.append(metric.binary.dc(binarized_cam, label))
#         #     alls.append([cam.mean(), metric.binary.dc(binarized_cam, label)])
#         # print(tt() - t)
        
#         # """ Method 2: 用cam有效部分的均值筛选掉一部分(效果不好) """
#         # if (cam[cam > 0]).mean() > 0:
#         #     binarized_cam = np.zeros_like(cam)
#         #     binarized_cam[cam > thresh] = 1
            
#         #     alls.append([cam[cam > 0].mean(), binarized_cam, os.path.join(output_path, filename), os.path.join(label_folder, filename)])

            
#         """ Method 3: 不对cam做筛选 """
#         binarized_cam = np.zeros_like(cam)
#         binarized_cam[cam > thresh] = 1
#         scores.append(metric.binary.dc(binarized_cam, label))
#         alls.append([cam.mean(), metric.binary.dc(binarized_cam, label), metric.binary.hd95(binarized_cam, label)])
#         sitk.WriteImage(sitk.GetImageFromArray(binarized_cam), out_path)
#         # print(tt() - t)
        
#         # """ Method 4: 将所有cam按照cam的均值部分排序，并取较高的一半 """
#         # binarized_cam = np.zeros_like(cam)
#         # binarized_cam[cam > thresh] = 1

#         # alls.append([cam[cam > 0].mean(), binarized_cam, label, os.path.join(output_path, filename)])

# """ Method 4: 将所有cam按照cam的均值部分排序，并取较高的一半 """
# alls.sort()
# for a, b in alls:
#     print(a, b)
# scores = np.array(scores)
# print(len(scores), scores.mean(), scores.std(), scores.max(), scores.min())

""" find_best_thresh """
ratio = 0
print('-' * 20)
for i in range(35, 80):
    score = []
    score_hd95 = []
    for filename in tqdm(files):
        cam = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_path, filename)))[:, :, :, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        thresh = ratio + i * 0.01
        label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_folder, filename)))
        binarized_cam = np.zeros_like(cam)
        binarized_cam[cam > thresh] = 1
        score.append(metric.binary.dc(binarized_cam, label))
        score_hd95.append(metric.binary.hd95(binarized_cam, label))
    score = np.array(score)
    score_hd95 = np.array(score_hd95)
    print(f"{thresh:.2f}, {score.mean()}, {score.std()}, {score.max()}, {score.min()}")
    print(f"hd95mean: {score_hd95.mean()}, hd95std: {score_hd95.std()}")
        
import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from medpy import metric
from time import time as tt
import argparse
args = argparse.ArgumentParser()
args.add_argument('stage', type=str, default="raw", help="raw or enhanced")
args = args.parse_args()
if args.stage == "raw":
    base_path = "/media/ubuntu/maxiaochuan/CLISC/data_BraTS/cam/train_layercam_l3"
else:
    base_path = "/media/ubuntu/maxiaochuan/CLISC/data_BraTS/cam/train_layercam_l3_aug"

label_folder = "/media/ubuntu/maxiaochuan/CLISC/data_BraTS/volume_pre/label/train"
input_path = os.path.join(base_path, 'cam')
output_path = os.path.join(base_path, "binarized_cam")

os.makedirs(output_path, exist_ok=True)

thresh = 0.27
files = os.listdir(input_path)
print("calculating")

alls = []
scores = []
for filename in tqdm(files):
    cam = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_path, filename)))[:, :, :, 0]
    label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_folder, filename)))
    out_path = os.path.join(output_path, filename)
    # if cam.sum(): # 若cam未捕捉到信息则直接不要
    # t = tt()
    if cam.sum():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        binarized_cam = np.zeros_like(cam)
        binarized_cam[cam > thresh] = 1
        scores.append(metric.binary.dc(binarized_cam, label))
        alls.append([cam.mean(), metric.binary.dc(binarized_cam, label), metric.binary.hd95(binarized_cam, label)])
        sitk.WriteImage(sitk.GetImageFromArray(binarized_cam), out_path)

""" find_best_thresh_with_valid """
# ratio = 0
# print('-' * 20)
# for i in range(35, 80):
#     score = []
#     score_hd95 = []
#     for filename in tqdm(files):
#         cam = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_path, filename)))[:, :, :, 0]
#         cam = (cam - cam.min()) / (cam.max() - cam.min())
#         thresh = ratio + i * 0.01
#         label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_folder, filename)))
#         binarized_cam = np.zeros_like(cam)
#         binarized_cam[cam > thresh] = 1
#         score.append(metric.binary.dc(binarized_cam, label))
#         score_hd95.append(metric.binary.hd95(binarized_cam, label))
#     score = np.array(score)
#     score_hd95 = np.array(score_hd95)
#     print(f"{thresh:.2f}, {score.mean()}, {score.std()}, {score.max()}, {score.min()}")
#     print(f"hd95mean: {score_hd95.mean()}, hd95std: {score_hd95.std()}")
        
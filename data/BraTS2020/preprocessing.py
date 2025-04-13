import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
import SimpleITK as sitk
import glob
import os
import random
import csv

from tqdm import tqdm

def brain_bbox(data, gt):
    mask = (data != 0)
    brain_voxels = np.where(mask != 0)
    mask = (data != 0).astype(np.uint8)
    brain_voxels_mask = np.where(mask != 0)
    new_mask = np.zeros_like(data, dtype=np.uint8)
    new_mask[brain_voxels_mask] = 1

    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))
    data_bboxed = data[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    gt_bboxed = gt[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    mask_bboxed = new_mask[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]

    return data_bboxed, gt_bboxed, mask_bboxed


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())


splits_dir = "./data/BraTS2020/splits"
os.makedirs(splits_dir, exist_ok=True)

image_dir = "./data/BraTS2020/image"
label_dir = "./data/BraTS2020/label"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

all_flair = glob.glob("./data/BraTS2020/raw_data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*_flair.nii")
random.seed(42)
random.shuffle(all_flair)

total_samples = len(all_flair)
train_idx = int(total_samples * 0.7)
valid_idx = int(total_samples * 0.1) + train_idx

train_files = all_flair[:train_idx]
valid_files = all_flair[train_idx:valid_idx]
test_files = all_flair[valid_idx:]

def save_split_csv(file_list, csv_path):
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_pth", "mask_pth"])
        for p in file_list:
            uid = os.path.basename(p).replace("_flair.nii", ".nii.gz")
            image_pth = os.path.join(image_dir, uid)
            mask_pth = os.path.join(label_dir, uid)
            writer.writerow([os.path.abspath(image_pth), os.path.abspath(mask_pth)])

for split, file_list in [('train', train_files), ('valid', valid_files), ('test', test_files)]:
    print(f"Processing {split} set: {len(file_list)} files")
    for p in tqdm(file_list):
        data = sitk.GetArrayFromImage(sitk.ReadImage(p))
        lab = sitk.GetArrayFromImage(sitk.ReadImage(p.replace("flair", "seg")))
        img, lab, mask = brain_bbox(data, lab)
        img = MedicalImageDeal(img, percent=0.999).valid_img
        lab[lab > 0] = 1
        uid = os.path.basename(p).replace("_flair.nii", ".nii.gz")
        
        sitk.WriteImage(sitk.GetImageFromArray(img), os.path.join(image_dir, uid))
        sitk.WriteImage(sitk.GetImageFromArray(lab), os.path.join(label_dir, uid))

    save_split_csv(file_list, os.path.abspath(os.path.join(splits_dir, f"{split}.csv")))
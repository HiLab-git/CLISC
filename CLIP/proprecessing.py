import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
import nibabel as nib
import SimpleITK as sitk
import glob
import os
import random

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

    # return data_bboxed, gt_bboxed
    return data_bboxed, gt_bboxed, mask_bboxed


def volume_bounding_box(data, gt, expend=0, status="train"):
    data, gt = brain_bbox(data, gt)
    print(data.shape)
    mask = (gt != 0)
    brain_voxels = np.where(mask != 0)
    z, x, y = data.shape
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))

    minZidx_jitterd = max(minZidx - expend, 0)
    maxZidx_jitterd = min(maxZidx + expend, z)
    minXidx_jitterd = max(minXidx - expend, 0)
    maxXidx_jitterd = min(maxXidx + expend, x)
    minYidx_jitterd = max(minYidx - expend, 0)
    maxYidx_jitterd = min(maxYidx + expend, y)

    data_bboxed = data[minZidx_jitterd:maxZidx_jitterd,
                       minXidx_jitterd:maxXidx_jitterd, minYidx_jitterd:maxYidx_jitterd]
    print([minZidx, maxZidx, minXidx, maxXidx, minYidx, maxYidx])
    print([minZidx_jitterd, maxZidx_jitterd,
           minXidx_jitterd, maxXidx_jitterd, minYidx_jitterd, maxYidx_jitterd])

    if status == "train":
        gt_bboxed = np.zeros_like(data_bboxed, dtype=np.uint8)
        gt_bboxed[expend:maxZidx_jitterd-expend, expend:maxXidx_jitterd -
                  expend, expend:maxYidx_jitterd - expend] = 1
        return data_bboxed, gt_bboxed

    if status == "test":
        gt_bboxed = gt[minZidx_jitterd:maxZidx_jitterd,
                       minXidx_jitterd:maxXidx_jitterd, minYidx_jitterd:maxYidx_jitterd]
        return data_bboxed, gt_bboxed


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out = out.astype(np.float32)
    return out


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


base_dir = "/media/ubuntu/maxiaochuan/CLISC/data_BraTS/volume_pre"
for split in ['train', 'valid', 'test']:
    os.makedirs(f"{base_dir}/image/{split}", exist_ok=True)
    os.makedirs(f"{base_dir}/label/{split}", exist_ok=True)

all_flair = glob.glob("/media/ubuntu/maxiaochuan/data/CLISC/data_BraTS/BraTS2020_TrainingData/*/*_flair.nii")
random.seed(42)  
random.shuffle(all_flair)

total_samples = len(all_flair)
train_idx = int(total_samples * 0.7)
valid_idx = int(total_samples * 0.1) + train_idx

# 划分数据集
train_files = all_flair[:train_idx]
valid_files = all_flair[train_idx:valid_idx]
test_files = all_flair[valid_idx:]

for split, file_list in [('train', train_files), ('valid', valid_files), ('test', test_files)]:
    print(f"Processing {split} set: {len(file_list)} files")
    for p in file_list:
        data = sitk.GetArrayFromImage(sitk.ReadImage(p))
        lab = sitk.GetArrayFromImage(sitk.ReadImage(p.replace("flair", "seg")))
        img, lab, mask = brain_bbox(data, lab)
        img = MedicalImageDeal(img, percent=0.999).valid_img
        lab[lab > 0] = 1
        uid = p.split("/")[-1]
        
        sitk.WriteImage(sitk.GetImageFromArray(img), 
                       f"{base_dir}/image/{split}/{uid}")
        sitk.WriteImage(sitk.GetImageFromArray(mask), 
                       f"{base_dir}/label/{split}/{uid}")
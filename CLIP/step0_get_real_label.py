# 使用噪声标签（clip得到的标签）训练cls_res50.py，使用真实标签计算准确率resnet_preprocess.py

import os
import torch
from tqdm import tqdm
import glob
from PIL import Image
import SimpleITK as sitk
import numpy as np
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

root_dir = '/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/volume_pre/image/valid' # 用验证集val调整训练参数，用含噪声的train来训练后续的resnet
root_dir_label = '/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/volume_pre/label/valid'
save_path = '/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/CLIP_label/gt'
filenames = glob.glob(root_dir + "/*.nii.gz") 
print(len(filenames))

label_pred, label_gt = [], []
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, "positive"), exist_ok=True)
os.makedirs(os.path.join(save_path, "negative"), exist_ok=True)

for idx, filename in enumerate(tqdm(filenames, ncols=70)):
    image_3d = sitk.GetArrayFromImage(sitk.ReadImage(filename))
    name = os.path.basename(filename).split('.')[0]  # 移除文件扩展名
    label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_dir_label, name)))

    positive_save_path = os.path.join(save_path, 'positive')
    negative_save_path = os.path.join(save_path, 'negative')

    for i in range(image_3d.shape[0]):
        image_2d = image_3d[i, :, :]
        if label[i,:,:].sum():
            label_cls = 1
        else:
            label_cls = 0

        image_2d_normalized = (image_2d - np.min(image_2d)) / (np.max(image_2d) - np.min(image_2d)) * 255
        image_2d_normalized = image_2d_normalized.astype(np.uint8)
        image_pil = Image.fromarray(image_2d_normalized).convert('L')
        image_resized = image_pil.resize([224,224])

        # 决定保存到哪个子文件夹
        if label_cls == 1:
            save_folder = positive_save_path
        else:
            save_folder = negative_save_path

        # 构建保存图像的文件名，包含切片索引和标签
        save_filename = f"{name}_slice_{i}_label_{label_cls}.tiff"
        image_resized.save(os.path.join(save_folder, save_filename))
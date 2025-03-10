from torchvision import models
import torch.nn as nn
import SimpleITK as sitk
import numpy as np
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import os

label_path = '/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/volume_pre/label/valid/BraTS20_Training_297.nii.gz'
cam_raw_path = label_path.replace('volume_pre/label/valid', 'cam/valid_layercam_l3/cam')

cam_largest_path = cam_raw_path.replace('3/cam', '3/cam_lcnt_3d_l3')
image_path = label_path.replace('label', 'image')
sam_path = '/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/volume_pre/pseudo/valid/BraTS20_Training_297.nii.gz'


cam_raw = sitk.GetArrayFromImage(sitk.ReadImage(cam_raw_path))
cam_largest = sitk.GetArrayFromImage(sitk.ReadImage(cam_largest_path))
image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
sam = sitk.GetArrayFromImage(sitk.ReadImage(sam_path))

save_folder = os.path.join('/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/sam/viscam', f'{label_path[-10:-7]}')
print(save_folder)

os.makedirs(save_folder, exist_ok=True)

for i in range(cam_raw.shape[0]):
    image_2d = image[i, :, :]
    
    image_2d_norm = (image_2d - image_2d.min()) / (image_2d.max() - image_2d.min())
    image_2d_norm = np.concatenate([np.expand_dims(image_2d_norm, 2), np.expand_dims(image_2d_norm, 2), np.expand_dims(image_2d_norm, 2)], 2)
    
    cam_raw_slice = cam_raw[i, :, :]
    if cam_raw_slice.sum():
        cam_raw_slice = (cam_raw_slice - cam_raw_slice.min()) / (cam_raw_slice.max() - cam_raw_slice.min())
    cam_largest_slice = cam_largest[i, :, :]
    label_slice = label[i, :, :]

    sam_slice = sam[i, :, :]
    show_sam = show_cam_on_image(image_2d_norm, sam_slice, use_rgb=False)
    
    show_cam_raw = show_cam_on_image(image_2d_norm, cam_raw_slice, use_rgb=False)
    show_cam_largest = show_cam_on_image(image_2d_norm, cam_largest_slice, use_rgb=False)
    show_label = show_cam_on_image(image_2d_norm, label_slice, use_rgb=False)
    
    fig = plt.figure(figsize=[20, 100], frameon=False)
    ax = fig.add_subplot(1, 4, 1)
    ax.axis("off")
    sv_vis = show_cam_raw[:, :, :: -1]
    ax.title.set_text('cam')
    ax.imshow(sv_vis)

    ax = fig.add_subplot(1, 4, 2)
    ax.axis("off")
    sv_vis_label = show_cam_largest[:, :, :: -1]
    ax.title.set_text('after largest')
    ax.imshow(sv_vis_label)
    
    ax = fig.add_subplot(1, 4, 3)
    ax.axis("off")
    sv_vis_label = show_sam[:, :, :: -1]
    ax.title.set_text('sam')
    ax.imshow(sv_vis_label)
    
    
    ax = fig.add_subplot(1, 4, 4)
    ax.axis("off")
    sv_vis_label = show_label[:, :, :: -1]
    ax.title.set_text('label')
    ax.imshow(sv_vis_label)
    # fig.subplots_adjust(hspace=0, wspace=0)
    
    fig.savefig(f'{save_folder}/test{i + 1}.png', bbox_inches='tight', pad_inches=0)
    plt.close()


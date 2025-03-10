import SimpleITK as sitk
import numpy as np
import torch
from segment_anything.utils.transforms import ResizeLongestSide
import os
import cv2
import re
from tqdm import tqdm
from pytorch_grad_cam.utils.image import show_cam_on_image

def find_bounding_box_2d(mask):
    assert mask.ndim == 2, "The mask must be a 2D numpy array."

    ones_indices = np.where(mask == 1)

    if not ones_indices[0].size:
        return None, None, None, None

    min_y = np.min(ones_indices[0])
    max_y = np.max(ones_indices[0])
    min_x = np.min(ones_indices[1])
    max_x = np.max(ones_indices[1])
    
    
    return min_y, max_y, min_x, max_x

input_path = '/media/ubuntu/maxiaochuan/CLISC/data_BraTS/CLIP_label/train'
cam_path = '/media/ubuntu/maxiaochuan/CLISC/data_BraTS/cam/train_layercam_l3/cam_lcnt_3d_l3'
save_path = '/media/ubuntu/maxiaochuan/CLISC/data_BraTS/CLIP_label/bounding_box'
phase = ['positive', 'negative']
os.makedirs(os.path.join(input_path.replace('train', 'bounding_box'), 'positive'), exist_ok=True)
os.makedirs(os.path.join(input_path.replace('train', 'bounding_box'), 'negative'), exist_ok=True)

for phe in phase:
    dirs = os.path.join(input_path, phe)
    files = os.listdir(dirs)
    for file in tqdm(files):
        pos = re.finditer('_', file)
        pos = [p.start() for p in pos]
        pos_id = pos[1] + 1
        pos_slice = (pos[3] + 1, pos[4])
        idx = file[pos_id]
        slice = int(file[pos_slice[0] : pos_slice[1]])
        if phe == 'positive':
            cam_slice_path = os.path.join(cam_path, file.replace(f'_slice_{slice}_label_1.tiff', '.nii.gz'))
        else:
            cam_slice_path = os.path.join(cam_path, file.replace(f'_slice_{slice}_label_0.tiff', '.nii.gz'))
        if not os.path.exists(cam_slice_path): 
            continue
        
        cam_save_path = cam_slice_path.replace('cam_lcnt_3d_l3', 'cam')
        
        cam = sitk.GetArrayFromImage(sitk.ReadImage(cam_slice_path))[slice, :, :]
        cam_save = sitk.GetArrayFromImage(sitk.ReadImage(cam_save_path))[slice, :, :, 0]
        min_y, max_y, min_x, max_x = find_bounding_box_2d(cam)
        if min_y == None: continue
        txt_path = os.path.join(input_path.replace('train', 'bounding_box'), phe, file.replace('.tiff', '.txt'))
        if not os.path.exists(txt_path):
            os.mknod(txt_path)
        with open(txt_path, 'w') as file:
            file.write(f"{min_y} {max_y} {min_x} {max_x}")
        cam_save = sitk.GetImageFromArray(cam_save)
        sitk.WriteImage(cam_save, txt_path.replace('.txt', '.nii.gz'))
        
        
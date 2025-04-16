import time
import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
from segment_anything.utils.transforms import ResizeLongestSide
import os
import cv2

import sys
from segment_anything import sam_model_registry, SamPredictor

import matplotlib.pyplot as plt
from medpy import metric


## 辅助函数
def find_bounding_box(mask):
    assert mask.ndim == 3, "The mask must be a 3D numpy array."
    ones_indices = np.where(mask == 1)
    if not ones_indices[0].size:
        return None
    min_z = np.min(ones_indices[0])
    max_z = np.max(ones_indices[0])
    min_y = np.min(ones_indices[1])
    max_y = np.max(ones_indices[1])
    min_x = np.min(ones_indices[2])
    max_x = np.max(ones_indices[2])

    return min_z, max_z, min_y, max_y, min_x, max_x

def find_bounding_box_2d(mask):
    assert mask.ndim == 2, "The mask must be a 2D numpy array."

    ones_indices = np.where(mask == 1)

    if not ones_indices[0].size:
        return None

    min_y = np.min(ones_indices[0])
    max_y = np.max(ones_indices[0])
    min_x = np.min(ones_indices[1])
    max_x = np.max(ones_indices[1])
    
    
    return min_y, max_y, min_x, max_x

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=4))  

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 


def save_top_ratios(all_score):
    all_score.sort()
    all_score.reverse()
    ratios = [0.6, 0.7, 0.8, 0.9]

    # Save files for each ratio
    for ratio in ratios:
        top_count = int(len(all_score) * ratio)
        top_files = all_score[:top_count]
        top_files = [[b, c] for a, b, c in top_files]

        # Create a DataFrame with two columns: image_pth and mask_pth
        df = pd.DataFrame(top_files, columns=["image_pth", "mask_pth"])

        # Save to CSV
        output_file = os.path.join("../data/BraTS2020/splits", f"top_{int(ratio * 100)}_percent.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved top {int(ratio * 100)}% to {output_file}")

## 实例化sam model
if __name__ == "__main__":
    sam_checkpoint = "../checkpoint/sam_vit_b_01ec64.pth" 
    model_type = "vit_b"

    # sam_checkpoint = "../checkpoint/sam_vit_h_4b8939.pth"
    # model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.cuda()
    predictor = SamPredictor(sam)

    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    root_dir = "../data/BraTS2020"
    UNet_label_dir = os.path.join(root_dir, "UNet_label")


    alls = len(os.listdir(UNet_label_dir))

    all_score = []
    for s, filename in enumerate(os.listdir(UNet_label_dir)):
        if filename.endswith('.nii.gz'):
            UNet_label_path = os.path.join(UNet_label_dir, filename) 
            image_path = os.path.join(root_dir, 'image', filename)
            label_path = os.path.join(root_dir, 'label', filename)

            UNet_label = sitk.GetArrayFromImage(sitk.ReadImage(UNet_label_path))

            if UNet_label.sum() == 0:
                continue

            img_array = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
            label_3d = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
            
            
            mask_3d = (img_array != 0).astype(np.uint8)

            pos_idx = []
            for i in range(len(UNet_label)):
                if UNet_label[i, :, :].sum() != 0:
                    pos_idx.append(i)


            foreground_points = {}
            for slice_index in pos_idx:
                cam_slice = UNet_label[slice_index, :, :]
                img_slice = img_array[slice_index, :, :]
                img_slice_normalized = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-6)
                masked_img = cam_slice * img_slice_normalized

                if masked_img.sum() == 0:
                    continue


                hist, bins = np.histogram(masked_img, bins=10, range=[np.min(masked_img), np.max(masked_img)])            
                high_intensity_threshold = bins[-5]
                foreground_indices = np.where(masked_img >= high_intensity_threshold)

                mid = int(len(foreground_indices[0]) / 2)
                third_1 = int(len(foreground_indices[0]) / 3)
                third_2 = int(len(foreground_indices[0]) / 3 * 2)

                foreground_points[slice_index] = [[foreground_indices[1][0], foreground_indices[0][0]],
                                                [foreground_indices[1][-1], foreground_indices[0][-1]],
                                                [foreground_indices[1][mid], foreground_indices[0][mid]],
                                                [foreground_indices[1][third_1], foreground_indices[0][third_1]],
                                                [foreground_indices[1][third_2], foreground_indices[0][third_2]],]
        
            ##  sam
            all_masks = []
            d = 0 # box dilation

            for i in range(len(pos_idx)):
                # score = 0
                current_batch = pos_idx[i]
                
                if current_batch not in foreground_points:
                    all_masks.append(np.zeros((img_slice.shape[0], img_slice.shape[1]), dtype=np.uint8))
                    continue
                
                input_img = img_array[current_batch, :, :]
                cam_img = UNet_label[current_batch]
                
                
                min_y, max_y, min_x, max_x = find_bounding_box_2d(cam_img)

                minXid = max(0, min_x - d)
                minYid = max(0, min_y - d)
                maxXid = min(input_img.shape[1], max_x + d)
                maxYid = min(input_img.shape[0], max_y + d)

                
                input_box = np.array([minXid, minYid, maxXid, maxYid])

                input_point = []
                input_label = []
                

               
                for fore in foreground_points[current_batch]:
                    input_point.append(fore)
                    input_label.append(1)
                
                
                # add two background points
                background_points = [
                    [minXid, minYid],
                    [minXid, maxYid],
                    [maxXid, minYid],
                    [maxXid, maxYid],
                ]
                for back in background_points:
                    input_point.append(back)
                    input_label.append(0)
                
                
                image_2d_normalized = (input_img - np.min(input_img)) / (np.max(input_img) - np.min(input_img)) * 255
                image_2d_normalized = image_2d_normalized.astype(np.uint8)

                
                image_2d_normalized = cv2.equalizeHist(image_2d_normalized)
                test = np.expand_dims(image_2d_normalized, axis=2)
                test = np.repeat(test, 3, axis=2)

                predictor.set_image(test)

                input_point = np.array(input_point)
                input_label = np.array(input_label)
                with torch.no_grad():
                    masks, _, _ = predictor.predict(
                        box=input_box,
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=False,
                    )
                
                all_masks.append(masks[0])


            if len(all_masks) > 0:
                all_masks_array = np.stack(all_masks, axis=0)
                sam_bzd = all_masks_array.astype(np.uint8)
            else:
                sam_bzd = np.zeros((len(pos_idx), img_array.shape[1], img_array.shape[2]), dtype=np.uint8)

            total_length = img_array.shape[0]
            complete_sam_bzd = np.zeros((total_length, sam_bzd.shape[1], sam_bzd.shape[2]), dtype=np.uint8)
            for i, idx in enumerate(pos_idx):
                complete_sam_bzd[idx] = sam_bzd[i]

            complete_sam_bzd *= mask_3d
            sam_dice = metric.binary.dc(complete_sam_bzd, label_3d)
            UNet_label_dice = metric.binary.dc(UNet_label, label_3d)
            similarity = metric.binary.dc(complete_sam_bzd, UNet_label)
            print(f'{s + 1} / {alls}, {filename}, UNet_label_dice: {UNet_label_dice}, sam_dice: {sam_dice}, similarity: {similarity}')
            all_score.append([similarity, os.path.join("../data/BraTS2020/image", filename), os.path.join("../data/BraTS2020/label", filename)])

    
    save_top_ratios(all_score)
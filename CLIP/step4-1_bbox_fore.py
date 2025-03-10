# mask不一样时,异常值检测

import time
import SimpleITK as sitk
import numpy as np
import torch
from segment_anything.utils.transforms import ResizeLongestSide
import os
import cv2

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

import matplotlib.pyplot as plt
from medpy import metric
import argparse


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

def find_bgpoints_2d(mask, min_y, max_y, min_x, max_x):
    zeros = np.where(mask[min_y : max_y + 1, min_x : max_x + 1] == 0)
    if len(zeros[0]) == 0: return None
    y_len = len(zeros[0])
    x_len = len(zeros[1])
    bgpoint1_y = zeros[0][y_len // 3] + min_y
    bgpoint1_x = zeros[1][x_len // 3] + min_x
    bgpoint2_y = zeros[0][y_len // 3 * 2] + min_y
    bgpoint2_x = zeros[1][x_len // 3 * 2] + min_x
    bgpoint_list = [[bgpoint1_x, bgpoint1_y], [bgpoint2_x, bgpoint2_y]]
    return bgpoint_list

def plot_fig(tensor_img):
    if tensor_img.size(0) == 1:
        tensor_img = tensor_img.squeeze(0)
    plt.imshow(tensor_img.cpu().numpy(), cmap='gray')
    plt.show()

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.unsqueeze(0).contiguous()


## 实例化sam model
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="cuda number")
args = parser.parse_args()
device = args.device
sam_checkpoint = "/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/segment-anything/sam_vit_b_01ec64.pth" 
model_type = "vit_b"

# sam_checkpoint = "/media/ubuntu//maxiaochuan/CLIP_SAM_zero_shot_segmentation/segment-anything/sam_vit_h_4b8939.pth"
# model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

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

def box_iou_xyxy(box1, box2):
    # 获取box1左上角和右下角的坐标
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    # 计算box1的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    # 获取box2左上角和右下角的坐标
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算box2的面积
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)
    
    # 计算相交矩形框的坐标
    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    # 计算相交矩形行的高度、宽度、面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w
    # 计算相并面积
    union = s1 + s2 - intersection
    # 计算交并比
    iou = intersection / union
    return iou

 


f_path = '/media/ubuntu//maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS'
cam_folder = f_path + '/cam/train_layercam_l3/cam_lcnt_3d_l3'
bbx_cam_folder = cam_folder


sam_dice_set = []
cam_dice_set = []
# scores = []
alls = len(os.listdir(cam_folder))
os.makedirs(f_path + '/volume_pre/pseudo/train/bare', exist_ok=True)
for s, cam_file in enumerate(os.listdir(cam_folder)):
    if cam_file.endswith('.nii.gz'):
        # 构建CAM和标签的文件路径
        cam_nii_path = os.path.join(cam_folder, cam_file) # 找到cam文件
        bbx_cam_nii_path = os.path.join(bbx_cam_folder, cam_file) # 同上
        # label_file = cam_file.replace('_CAM', '')
        # file = label_file.replace('_binary', '')
        # label_path = f_path + '/volume_pre/label/train/' + file
        # input_img_path = f_path + '/volume_pre/image/train/' + file

        # output_path = f_path + '/train/train_sam/bb_2d/' + file
        # out_point_fig_dir = f_path + '/train/train_sam/box_2d/' + file.split('.')[0] + '/'



        label_path = f_path + '/volume_pre/label/train/' + cam_file
        input_img_path = f_path + '/volume_pre/image/train/' + cam_file

        output_path = f_path + '/volume_pre/pseudo/train/bare/' + cam_file
        out_point_fig_dir = f_path + '/sam/testing/box/' + cam_file.split('.')[0] + '/'
        
        
        mask_file = cam_file.replace('.nii.gz', '_flair.nii')
        mask_path = '/media/ubuntu//maxiaochuan/CLIP_SAM_zero_shot_segmentation/data/BraTS2020_preprocess/mask/' + mask_file

        if not os.path.exists(out_point_fig_dir):
            os.makedirs(out_point_fig_dir, exist_ok=True)

        ##  加载cam,图像,label

        cam_array = sitk.GetArrayFromImage(sitk.ReadImage(cam_nii_path))
        bbx_cam_array = sitk.GetArrayFromImage(sitk.ReadImage(bbx_cam_nii_path))

        if cam_array.sum() == 0:
            continue

        img = sitk.ReadImage(input_img_path)
        img_array = sitk.GetArrayFromImage(img)
        label_3d = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        mask_3d = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

        pos_idx = []
        for i in range(len(cam_array)):
            if cam_array[i, :, :].sum() != 0:
                pos_idx.append(i)

        ## if 3d bbox
        # min_z, max_z, min_y, max_y, min_x, max_x = find_bounding_box(cam_array)
        # slice_set = np.array(range(min_z, max_z))
        # input_box = np.array([min_x, min_y, max_x, max_y])

        ## fore 
        foreground_points = {}
        for slice_index in pos_idx:
            cam_slice = cam_array[slice_index, :, :]
            img_slice = img_array[slice_index, :, :]
            img_slice_normalized = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))
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
            input_img = img_array[current_batch, :, :]
            cam_img = bbx_cam_array[current_batch]
            if cam_img.sum() == 0:
                continue
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
            # bgpoints_list = find_bgpoints_2d(cam_img, minYid, maxYid, minXid, maxXid)
            # if bgpoints_list != None:
            #     for back in bgpoints_list:
            #         input_point.append(back)
            #         input_label.append(0)
            
            
            image_2d_normalized = (input_img - np.min(input_img)) / (np.max(input_img) - np.min(input_img)) * 255
            image_2d_normalized = image_2d_normalized.astype(np.uint8)

            # 对比度限制的自适应直方图均衡
            # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
            # clahe_image = clahe.apply(image_2d_normalized)
            
            image_2d_normalized = cv2.equalizeHist(image_2d_normalized)
            test = np.expand_dims(image_2d_normalized, axis=2)
            test = np.repeat(test, 3, axis=2)

            predictor.set_image(test)

            input_point = np.array(input_point)
            input_label = np.array(input_label)

            masks, _, _ = predictor.predict(
                box=input_box,
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
            
            all_masks.append(masks[0])

            plt.figure(figsize=(10,10))
            plt.imshow(input_img, cmap='gray')
            show_mask(masks[0], plt.gca())
            show_box(input_box, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.axis('off')
            point_save = out_point_fig_dir + str(current_batch) +'.jpg'
            plt.savefig(point_save, bbox_inches='tight', pad_inches=0)
            plt.close()

            # box1 = [minXid, minYid, maxXid, maxYid]
            # b1, b2, a1, a2 = find_bounding_box_2d(masks[0])
            # box2 = [a1, b1, a2, b2]
            # score += box_iou_xyxy(box1, box2)
            

        if len(all_masks) > 0:
            all_masks_array = np.stack(all_masks, axis=0)
            sam_bzd = all_masks_array.astype(np.uint8)
        else:
            sam_bzd = np.zeros((len(pos_idx), img_array.shape[1], img_array.shape[2]), dtype=np.uint8)

        ## 补齐sam的分割
        total_length = img_array.shape[0]
        complete_sam_bzd = np.zeros((total_length, sam_bzd.shape[1], sam_bzd.shape[2]), dtype=np.uint8)
        for i, idx in enumerate(pos_idx):
            complete_sam_bzd[idx] = sam_bzd[i]

        ## sam去除非脑区
        complete_sam_bzd *= mask_3d
        sam_save = sitk.GetImageFromArray(complete_sam_bzd)
        # sitk.WriteImage(sam_save, output_path)
        ## 对比原始cam
        sam_dice = metric.binary.dc(complete_sam_bzd, label_3d)
        cam_dice = metric.binary.dc(cam_array, label_3d)
        print(f'{s + 1} / {alls}, {cam_file}, cam_dice: {cam_dice}, sam_dice: {sam_dice}')
        sam_dice_set.append(sam_dice)
        cam_dice_set.append(cam_dice)
        # scores.append([score, cam_dice, sam_dice])
        

sam_dice_set = np.array(sam_dice_set) 
cam_dice_set = np.array(cam_dice_set)
print('finished')
print(f'cam_dice:{np.mean(cam_dice_set):.4f} cam_std:{np.std(cam_dice_set):.4f} sam_dice:{np.mean(sam_dice_set):.4f} sam_std:{np.std(sam_dice_set):.4f}')
# 关闭文件

# scores.sort()
# for score, cam_dice, sam_dice in scores:
#     print(score, '--', cam_dice, '--', sam_dice)
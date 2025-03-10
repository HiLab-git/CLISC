# optimal threshold (slice-level and volume-level)

# find the threshold to generate initial mask using cam
# two ways can be selected to generate mask:
# 1) true_label_one: if the true label of sample is one, the cam is used to generate mask using the given threshold
# 2) pred_label_one: if the pred label of sample is one, the cam is used to generate mask using the given threshold

# run 'generate_cam.py' before using this file

# python step3_find_best_threshold.py --method "gradcam_gce" --region 3

import os
import sys
import logging
import datetime
import argparse
import numpy as np
import pandas as pd
import medpy.metric as metric
import SimpleITK as sitk
from skimage import measure

def largestConnectComponent(binaryimg, ratio=1):
    label_image, num = measure.label(binaryimg, background=0, return_num=True)
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    if num > 1:
        for region in measure.regionprops(label_image):
            if region.area < areas[-1] / ratio:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    label_image = label_image.astype(np.int8)
    label_image[np.where(label_image > 0)] = 1

    return label_image


def largestConnectComponent_3d(binaryimg, ratio=1):
    label_image, num = measure.label(binaryimg, background=0, return_num=True)
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    if num > 1:
        for region in measure.regionprops(label_image):
            if region.area < areas[-1] / ratio:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1], coordinates[2]] = 0
    label_image = label_image.astype(np.int8)
    label_image[np.where(label_image > 0)] = 1

    return label_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="finding threshold for cam or fused cam"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default='/media/ubuntu/data3/xuzhen/data_BraTS/volume_pre/label/valid',
    )
    parser.add_argument("--postprocessing", type=float, default=1) # 默认找最大联通
    parser.add_argument("--regions", type=int, default=1)
    parser.add_argument("--best_thresh", type=float, default=0)
    
    parser.add_argument(
        "--mask_path", 
        type=str, 
        default="/media/ubuntu/data3/xuzhen/data/BraTS2020_preprocess/mask",
    )
    parser.add_argument(
        "--save_cam_dir", 
        default="/media/ubuntu/data3/xuzhen/data_BraTS/res50_val_cam",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
        choices=[
            "gradcam",
            "gradcam_gce",
            "gradcam_elt",
            "gradcam_l3",
            "scorecam",
            "ablationcam",
            "layercam",
            "layercam_l3",
            "layercam_l3_l4",
            "layercam_l3_aug",
            "layercam_l3_aug2",  #adaptive thresh
            "layercam_l3_l4_aug",
        ],

    )
    sys.path.append(os.getcwd())
    args = parser.parse_args()

    cam_dir = os.path.join(
        args.save_cam_dir + "/",
        args.method + "/cam",
        )
    save_name_log = "_GridSearch.log"    
    print(cam_dir)

    save_name_log = os.path.join(os.path.dirname(cam_dir), save_name_log)
    logging.basicConfig(
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H-%M-%S",
        filename=save_name_log,
        level=logging.DEBUG,
    )
    logging.debug(args)

    """load data"""
    cam_folder = cam_dir
    label_folder = args.root_path
    mask_folder = args.mask_path

    if args.best_thresh:
        best_thresh = args.best_thresh
    else:
        best_Dice = 0.4
        
        for step in range(4,10): # layercam l3需要比较小的阈值(4,10),0.02 有l4信息的：(4,16),0.05
            thresh = step * 0.05
            Dice_all_metric = []
            
            for cam_file in os.listdir(cam_folder):
                if cam_file.endswith('.nii.gz'):
                    # 构建CAM和标签的文件路径
                    cam_path = os.path.join(cam_folder, cam_file)
                    label_file = cam_file.replace('_CAM', '')
                    label_path = os.path.join(label_folder, label_file)
                    # mask_pth = os.path.join(mask_folder, label_file)

                    mask_file = label_file.replace('.nii.gz', '_flair.nii')
                    mask_pth = os.path.join(mask_folder, mask_file)

                    # 读取CAM和标签图像
                    cam_3d = sitk.GetArrayFromImage(sitk.ReadImage(cam_path))[:,:,:,0]
                    label_3d = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
                    mask_3d = sitk.GetArrayFromImage(sitk.ReadImage(mask_pth))

                    # 应用mask去除非脑区
                    cam_3d *= mask_3d
                    label_3d *= mask_3d
                
                    mask = np.zeros_like(cam_3d, dtype=label_3d.dtype)
                    mask[cam_3d > thresh * 255] = 1
                    if args.postprocessing:  # 最大联通找肿瘤区域
                        mask = largestConnectComponent_3d(mask, ratio=args.regions)

                    dice = metric.binary.dc(mask, label_3d)
                    Dice_all_metric.append(dice)

            Dice_all_metric = np.array(Dice_all_metric)

            if Dice_all_metric.mean() > best_Dice:
                best_thresh = thresh
                best_Dice = Dice_all_metric.mean()
                logging.debug(
                    "threshold: {:.4f}, best dice_mean: {:.4f}, dice_std: {:.4f}, dice_max: {:.4f}\n".format(
                        thresh, Dice_all_metric.mean(), Dice_all_metric.std(), Dice_all_metric.max()
                    )
                )
                print(
                    "threshold: {:.4f}, best dice_mean: {:.4f}, dice_std: {:.4f}, dice_max: {:.4f}\n".format(
                        thresh, Dice_all_metric.mean(), Dice_all_metric.std(), Dice_all_metric.max()
                    )
                )
            else:
                logging.debug(
                    "threshold: {:.4f}, dice_mean: {:.4f}, dice_std: {:.4f}, dice_max: {:.4f}\n".format(
                        thresh, Dice_all_metric.mean(), Dice_all_metric.std(), Dice_all_metric.max()
                    )
                )
                print(
                    "threshold: {:.4f}, dice_mean: {:.4f}, dice_std: {:.4f}, dice_max: {:.4f}\n".format(
                        thresh, Dice_all_metric.mean(), Dice_all_metric.std(), Dice_all_metric.max()
                    )
                )
    
    print("best threshold", best_thresh)

    Dice_all_metric = []
    name_set = []
    for cam_file in os.listdir(cam_folder):
        if cam_file.endswith('.nii.gz'):
            cam_path = os.path.join(cam_folder, cam_file)
            label_file = cam_file.replace('_CAM', '')
            label_path = os.path.join(label_folder, label_file)

            mask_file = label_file.replace('.nii.gz', '_flair.nii')
            mask_pth = os.path.join(mask_folder, mask_file)

            cam_3d = sitk.GetArrayFromImage(sitk.ReadImage(cam_path))[:,:,:,0]
            label_3d = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
            mask_3d = sitk.GetArrayFromImage(sitk.ReadImage(mask_pth))

            cam_3d *= mask_3d

            # 保存去掉非脑区的cam
            folder_path = cam_folder+'/masked_cam'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            output_path = os.path.join(folder_path, cam_file.replace('.nii', '_CAM_mask.nii')) 
            cam_masked = sitk.GetImageFromArray(cam_3d) 
            sitk.WriteImage(cam_masked, output_path)

            label_3d *= mask_3d
        
            mask = np.zeros_like(cam_3d, dtype=label_3d.dtype)
            # best_thresh = 0.5
            mask[cam_3d > best_thresh * 255] = 1
            if args.postprocessing: 
                mask = largestConnectComponent_3d(mask, ratio=args.regions)

            # 保存二值化后的cam
            folder_path = cam_folder+'/binarized_cam_3d'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            output_path = os.path.join(folder_path, cam_file.replace('.nii', '_CAM_binary.nii')) 
            cam_bzd = sitk.GetImageFromArray(mask) 
            sitk.WriteImage(cam_bzd, output_path)

            dice = metric.binary.dc(mask, label_3d)
            Dice_all_metric.append(dice)
            name_set.append(cam_file)

    Dice_all_metric = np.array(Dice_all_metric)
    logging.debug(
        "best threshold: {:.4f}, best dice_mean: {:.2f}, dice_std: {:.2f}, dice_max: {:.2f}\n".format(
            best_thresh,
            Dice_all_metric.mean() * 100.0,
            Dice_all_metric.std() * 100.0,
            Dice_all_metric.max() * 100.0,
        )
    )
    print(
        "best threshold: {:.4f}, best dice_mean: {:.2f}, dice_std: {:.2f}, dice_max: {:.2f}\n".format(
            best_thresh,
            Dice_all_metric.mean() * 100.0,
            Dice_all_metric.std() * 100.0,
            Dice_all_metric.max() * 100.0,
        )
    )
    # print(name_set[np.argmax(Dice_all_metric)])
    # median_idx = np.where(Dice_all_metric == np.median(Dice_all_metric))
    # print(name_set[median_idx[0]])
    
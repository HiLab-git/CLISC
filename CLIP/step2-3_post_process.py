import SimpleITK as sitk
import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import binary_opening, generate_binary_structure, iterate_structure
from skimage import measure
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm

label_folder = '/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/volume_pre/label/train'
cam_folder = '/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/cam/train_layercam_l3/binarized_cam'


def largestConnectComponent_3d(binaryimg, ratio=1):
    label_image, num = measure.label(binaryimg, background=0, return_num=True)
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    if num > 1:
        for region in measure.regionprops(label_image):
            if region.area < areas[-1] / ratio:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1], coordinates[2]] = 0
    label_image = label_image.astype(np.uint8)
    label_image[np.where(label_image > 0)] = 1

    return label_image

def largestConnectComponent_2d(binaryimg, ratio=1):
    label_image, num = measure.label(binaryimg, background=0, return_num=True)
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    if num > 1:
        for region in measure.regionprops(label_image):
            if region.area < areas[-1] / ratio:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    label_image = label_image.astype(np.uint8)
    label_image[np.where(label_image > 0)] = 1

    return label_image

dc_all_0 = []
dc_all_1 = []
dc_all_2 = []
alls = []
struct = generate_binary_structure(3, 3)  # 1:6 neighor, 2:18neighbor, 3:26neighbor
struct = iterate_structure(struct, iterations=1)  # dilate struct 
print("calculating.............")
filenames = os.listdir(cam_folder)
for tt, filename in enumerate(tqdm(filenames)):
    cam = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(cam_folder, filename)))
    label = sitk.GetArrayFromImage(sitk.ReadImage(f'{label_folder}/{filename}'))
    dc_0 = metric.binary.dc(cam, label)
    dc_all_0.append(dc_0)

    # morph 变形, 去除孤立的1
    cam_mor = binary_opening(cam, structure=struct, iterations=1) 
    dc_1 = metric.binary.dc(cam_mor, label)
    dc_all_1.append(dc_1)

    folder_path = cam_folder.replace('binarized_cam', 'cam_mor')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    output_path = os.path.join(folder_path, filename) 
    cam_mor = cam_mor.astype(np.uint8)
    cam_bzd = sitk.GetImageFromArray(cam_mor)
    sitk.WriteImage(cam_bzd, output_path)   

    folder_path = cam_folder.replace('binarized_cam', 'cam_lcnt_3d_l3')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mask = largestConnectComponent_3d(cam_mor, ratio=3)

    output_path = os.path.join(folder_path, filename) 
    cam_bzd = sitk.GetImageFromArray(mask)
    
    dc_2 = metric.binary.dc(mask, label)
    dc_all_2.append(dc_2)
    alls.append([mask.mean(), output_path, cam_bzd, dc_2])
    

print("cam score")
dc_all_0 = np.array(dc_all_0)
print(round(dc_all_0.mean(), 4), round(dc_all_0.std(), 4), round(dc_all_0.max(), 4), round(dc_all_0.min(), 4))

print("morphology score")
dc_all_1 = np.array(dc_all_1)
print(round(dc_all_1.mean(), 4), round(dc_all_1.std(), 4), round(dc_all_1.max(), 4), round(dc_all_1.min(), 4))

print("largestcnt score")
dc_all_2 = np.array(dc_all_2)
print(round(dc_all_2.mean(), 4), round(dc_all_2.std(), 4), round(dc_all_2.max(), 4), round(dc_all_2.min(), 4))

alls.sort()
save_alls = []
for i, (value, output_path, cam_save, dc) in enumerate(alls):
    print(value, dc)
    if i > len(alls) // 20:
        save_alls.append(dc)
        sitk.WriteImage(cam_save, output_path)
save_alls = np.array(save_alls)
print(save_alls.mean(), save_alls.std(), save_alls.max(), save_alls.min())
from html import parser
import os
import pandas as pd
import torch
import cv2
from tqdm import tqdm
import glob
from PIL import Image
import SimpleITK as sitk
import numpy as np
from torchvision import models, transforms
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from medpy.metric.binary import dc
from scipy.ndimage import binary_opening, generate_binary_structure, iterate_structure
from skimage import measure
import argparse

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



def get_cam(filenames, model, output_dir):
    for filename in tqdm((filenames), ncols=70):
        image_3d = sitk.GetArrayFromImage(sitk.ReadImage(filename))

        cam_3d = np.zeros([image_3d.shape[0], image_3d.shape[1], image_3d.shape[2], 3], dtype=np.uint8)
        cam_plus = np.zeros([image_3d.shape[0], image_3d.shape[1], image_3d.shape[2], 3], dtype=np.uint8)

        name = os.path.basename(filename)
        label = sitk.GetArrayFromImage(sitk.ReadImage(filename.replace("image", "label")))
        cal = 0
        no_cal = 0
        
        for i in range(image_3d.shape[0]):  
            image_2d = image_3d[i, :, :]  # 提取二维图像
            if label[i,:,:].sum():
                label_cls = 1
            else:
                label_cls = 0

            image_pil = Image.fromarray(image_2d)  # 转换为PIL图像

            image_2d_normalized = (image_pil - np.min(image_pil)) / (np.max(image_pil) - np.min(image_pil)) * 255
            image_2d_normalized = image_2d_normalized.astype(np.uint8)
            image_save = Image.fromarray(image_2d_normalized).convert('L')

            input_tensor = preprocess(image_save).unsqueeze(0).to(device)

            # 使用模型获取预测
            with torch.no_grad():
                input_tensor = input_tensor.repeat(1, 3, 1, 1)
                outputs = model(input_tensor) # [1, 2]
                _, preds = torch.max(outputs, dim=1)
                
                # probs = torch.nn.functional.softmax(outputs, dim=1)  # 计算softmax概率
                # preds = probs[:, 1] > threshold  # 如果正类的概率大于0.7，判断为正类

                pred_class = preds.cpu().numpy()[0] # 如果是1代表预测为有肿瘤的概率更大

            # 如果预测为有肿瘤的类别（类别1），则生成Grad-CAM
            if pred_class == 1 and label_cls == 1:
                cal = cal + 1
                targets = [ClassifierOutputTarget(1)]

                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]  # 224 224

                img = input_tensor.squeeze(0).permute(1,2,0).cpu()
                img = np.array(img) # 224 224 3

                visualization = show_cam_on_image(img, grayscale_cam, use_rgb=False)
                image_2d_norm = (image_2d - image_2d.min()) / (image_2d.max() - image_2d.min())
                image_2d_norm = np.concatenate([np.expand_dims(image_2d_norm, 2), np.expand_dims(image_2d_norm, 2), np.expand_dims(image_2d_norm, 2)], 2)
                visualization_label = show_cam_on_image(image_2d_norm, label[i,:,:], use_rgb=False)
                
                cam_im = np.uint8(grayscale_cam * 255)
                cam_im = np.stack((cam_im,) * 3, axis=-1)

                # 再resize回来
                cam_im = cv2.resize(cam_im, [image_3d.shape[2], image_3d.shape[1]])
                visualization = cv2.resize(visualization, [image_3d.shape[2], image_3d.shape[1]])
                # cv2.imwrite('label.png', visualization_label)
                # cv2.imwrite('cam_resize.png', visualization)

                fig = plt.figure(figsize=[10, 5], frameon=False)
                ax = fig.add_subplot(1, 2, 1)
                ax.axis("off")
                sv_vis = visualization[:, :, :: -1]
                ax.imshow(sv_vis)

                ax = fig.add_subplot(1, 2, 2)
                ax.axis("off")
                sv_vis_label = visualization_label[:, :, :: -1]
                ax.imshow(sv_vis_label)
                fig.subplots_adjust(hspace=0, wspace=0)

                save_name = os.path.basename(filename).split('.')[0]
                save_folder = output_dir + f'/vis_cam/{save_name}/'
                os.makedirs(save_folder, exist_ok=True)
                save_path = save_folder + f'{i}.png'
                fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()            

                cam_3d[i, :, :, :] = cam_im # [181, 131, 3]

            # 如果预测为没有肿瘤，则记为全黑图像
            else:
                no_cal = no_cal + 1
                img = input_tensor.squeeze(0).permute(1,2,0).cpu()
                img = np.array(img)
                cam_im = np.zeros((224, 224, 3), dtype=np.float32)

                cam_im = np.uint8(cam_im * 255)

                cam_im = cv2.resize(cam_im, [image_3d.shape[2], image_3d.shape[1]])
                visualization = cv2.resize(img, [image_3d.shape[2], image_3d.shape[1]])
                visualization = np.uint8(visualization * 255)

                cam_3d[i, :, :, :] = cam_im
                cam_plus[i, :, :, :] = visualization
        
        output_path = os.path.join(output_dir+'/cam', name) 
        cam_3d_sitk = sitk.GetImageFromArray(cam_3d) 
        
        sitk.WriteImage(cam_3d_sitk, output_path) 




def find_best_binary_threshold(cam_dir):
    files = os.listdir(os.path.join(cam_dir, "cam"))
    cam_files = [os.path.join(cam_dir, "cam", file) for file in files if file.endswith(".nii.gz")]
    label_files = [os.path.join("../data/BraTS2020", "label", file) for file in files if file.endswith(".nii.gz")]

    if len(cam_files) != len(label_files):
        raise ValueError("The number of CAM files and label files must be the same.")

    best_threshold = 0.0
    best_dice = 0.0

    # Iterate over thresholds from 0.2 to 0.8 with a step of 0.05
    for threshold in np.arange(0.2, 0.85, 0.05):
        dice_scores = []

        for cam_file, label_file in zip(cam_files, label_files):
            # Load CAM and label
            cam = sitk.GetArrayFromImage(sitk.ReadImage(cam_file))[..., 0]
            label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))

            # Binarize CAM using the current threshold
            cam_binary = (cam >= threshold * 255).astype(np.uint8)

            # Compute Dice coefficient
            if label.sum() > 0:  # Avoid division by zero for empty labels
                dice = dc(cam_binary, label)
                dice_scores.append(dice)

        # Compute the average Dice score for the current threshold
        avg_dice = np.mean(dice_scores) if dice_scores else 0.0

        # Update the best threshold if the current one is better
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_threshold = threshold
        print(f"threshold: {threshold:.2f}, Dice: {avg_dice:.4f}")

    print(f"Best threshold: {best_threshold}, Best Dice: {best_dice}")
    return best_threshold



def binarize_cam(cam_dir, threshold):
    cam_files = sorted(glob.glob(os.path.join(cam_dir, "cam", "*.nii.gz")))
    output_dir = os.path.join(cam_dir, "cam_post")
    os.makedirs(output_dir, exist_ok=True)

    # Define the structuring element for morphological operations
    struct = generate_binary_structure(3, 3)  # 3D structuring element with 26 neighbors
    struct = iterate_structure(struct, iterations=1)  # Dilate the structuring element

    for cam_file in tqdm(cam_files):
        # Load the CAM
        cam = sitk.GetArrayFromImage(sitk.ReadImage(cam_file))[..., 0]

        # Binarize the CAM using the threshold
        cam_binary = (cam >= threshold * 255).astype(np.uint8)

        # Apply morphological operations
        cam_mor = binary_opening(cam_binary, structure=struct).astype(np.uint8)

        # Extract the largest connected component
        cam_lcc = largestConnectComponent_3d(cam_mor, ratio=3)

        # Save the processed CAM
        cam_lcc_sitk = sitk.GetImageFromArray(cam_lcc)
        cam_lcc_sitk.CopyInformation(sitk.ReadImage(cam_file))  # Copy metadata from the original CAM
        output_path = os.path.join(output_dir, os.path.basename(cam_file))
        sitk.WriteImage(cam_lcc_sitk, output_path)

        tqdm.write(f"Processed and saved: {output_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--stage', type=str, default="raw", choices=["raw", "aug"], help="running stage")
    args = args.parse_args()
    stage = args.stage

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_csv_path = "../data/BraTS2020/splits/train.csv"
    valid_csv_path = "../data/BraTS2020/splits/valid.csv"
    train_filenames = pd.read_csv(train_csv_path)["image_pth"]
    valid_filenames = pd.read_csv(valid_csv_path)["image_pth"]
    model_path = f"./resnet50_model/{stage}/best_model.pth"
    output_dir = f"../data/BraTS2020/LayerCAM/{stage}"

    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    # Modify the classifier (fc) part of the model
    model.fc = nn.Sequential(
        nn.Flatten(), 
        nn.Linear(num_ftrs, 256), 
        nn.ReLU(), 
        nn.Dropout(0.5),  
        nn.Linear(256, 2) 
    )
    # model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    target_layers = [model.layer3]
    cam = LayerCAM(model=model, target_layers=target_layers)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_output_dir = os.path.join(output_dir, "train")
    valid_output_dir = os.path.join(output_dir, "valid")
    label_pred, label_gt = [], []
    
    os.makedirs(os.path.join(valid_output_dir, "cam"), exist_ok=True)
    os.makedirs(os.path.join(valid_output_dir, "vis_cam"), exist_ok=True)
    os.makedirs(os.path.join(train_output_dir, "cam"), exist_ok=True)
    os.makedirs(os.path.join(train_output_dir, "vis_cam"), exist_ok=True)

    get_cam(valid_filenames, model, valid_output_dir)
    threshold = find_best_binary_threshold(valid_output_dir)
    get_cam(train_filenames, model, train_output_dir)
    binarize_cam(train_output_dir, threshold)


    



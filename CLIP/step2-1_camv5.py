from html import parser
import os
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
import argparse

class EqualizeHist:
    """Apply OpenCV clahe equalize to the image."""
    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(1, 1))
            clahe_image = clahe.apply(img_gray)
            img_np = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(1, 1))
            clahe_image = clahe.apply(img_np)

        return Image.fromarray(img_np)

args = argparse.ArgumentParser()
args.add_argument('stage', type=str, default="raw", help="raw or enhanced")
args = args.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if args.stage == "raw":
    root_dir = '/media/ubuntu/maxiaochuan/CLISC/data_BraTS/volume_pre/image/train'
    output_dir = '/media/ubuntu/maxiaochuan/CLISC/data_BraTS/cam/train_layercam_l3'
    model_path = '/media/ubuntu/maxiaochuan/CLISC/CLIP/resnet_model/bare_res50_best_model.pth'
else:
    root_dir = '/media/ubuntu/maxiaochuan/CLISC/data_BraTS/volume_pre/image/train'
    root_dir_label = '/media/ubuntu/maxiaochuan/CLISC/data_BraTS/volume_pre/label/train'
    output_dir = '/media/ubuntu/maxiaochuan/CLISC/data_BraTS/cam/train_layercam_l3_aug'
    model_path = '/media/ubuntu/maxiaochuan/CLISC/CLIP/resnet_model/aug_res50_best_model.pth'
    
root_dir_label = '/media/ubuntu/maxiaochuan/CLISC/data_BraTS/volume_pre/label/train'
filenames = glob.glob(root_dir + "/*.nii.gz") 
print(len(filenames))

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
# exit()
cam = LayerCAM(model=model, target_layers=target_layers)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    # EqualizeHist(),
    transforms.ToTensor()
])

label_pred, label_gt = [], []
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "cam"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "vis_cam"), exist_ok=True)

for idx in tqdm(range(len(filenames)), ncols=70):
    filename = filenames[idx]
    image_3d = sitk.GetArrayFromImage(sitk.ReadImage(filename))

    cam_3d = np.zeros([image_3d.shape[0], image_3d.shape[1], image_3d.shape[2], 3], dtype=np.uint8)
    cam_plus = np.zeros([image_3d.shape[0], image_3d.shape[1], image_3d.shape[2], 3], dtype=np.uint8)

    name = os.path.basename(filename)
    label = sitk.GetArrayFromImage(sitk.ReadImage(root_dir_label + '/' + name.replace('T2w', 'dseg')))
    cal = 0
    no_cal = 0
    threshold = 0.7
    
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

    



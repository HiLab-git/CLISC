import os
import torch
import clip
from tqdm import tqdm
import glob
from PIL import Image
import SimpleITK as sitk
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import argparse


def classify_thresh(probs, threshold = 0.6):
    max_prob = np.max(probs)
    pred_cls = np.argmax(probs)

    if max_prob >= threshold:
        return pred_cls, True
    else:
        return -1, False

def pr_result(cm, label_gt, label_pred):
    print(cm)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    print(f'TN:{TN}, TP:{TP}, FN:{FN}, FP:{FP}')

    val_cls_acc = accuracy_score(label_gt, label_pred)
    val_cls_prec = precision_score(label_gt, label_pred, zero_division=0)
    val_cls_recall = recall_score(label_gt, label_pred, zero_division=0)
    val_cls_f1 = f1_score(label_gt, label_pred, zero_division=0)
    print(f'{val_cls_acc}|{val_cls_prec}|{val_cls_recall}|{val_cls_f1}')


def get_clip_label(filenames, model, preprocess, save_path):
    label_pred, label_gt = [], []
    prompt_normal = "an image of brain tissue showing typical signal intensity without any regions of abnormal intensity or suspicious mass"
    prompt_abnormal = "an image of brain tissue showing a tumor with uneven hyperintensity and irregular borders distinct from surroundings"


    text = clip.tokenize([prompt_normal, prompt_abnormal]).to(device)

    os.makedirs(os.path.join(save_path, "positive"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "negative"), exist_ok=True)

    for idx in tqdm(range(len(filenames)), ncols=70):  
        filename = filenames[idx]
        image_3d = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        label = sitk.GetArrayFromImage(sitk.ReadImage(filename.replace("image", "label")))
        name = os.path.basename(filename)

        slice_list = []
        for i in range(image_3d.shape[0]):  
            image_2d = image_3d[i, :, :]  
            
            label_cls = 1 if label[i, :, :].sum() else 0
            # print(label_cls)

            image_pil = Image.fromarray(image_2d)  
            image_preprocessed = preprocess(image_pil).unsqueeze(0).cuda()
            
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image_preprocessed, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                pred, is_certain = classify_thresh(probs, threshold=0.6)  

            if is_certain: 
                slice_list.append([i, pred, label_cls, image_2d])

            
        for i, (idx, pred, label_cls, image_2d) in enumerate(slice_list):
            around_label = []
            for j in range(max(0, i - 5), (min(len(slice_list), i + 6))):
                if j == i: continue
                tmp_idx = slice_list[j][0]
                if abs(idx - tmp_idx) > 6: continue
                around_label.append(slice_list[j][1])
            
            
            if len(around_label) > 8: 
                cnt_1 = sum(around_label)
                cnt_0 = len(around_label) - cnt_1

                target = 1 if cnt_1 > cnt_0 else 0
                pred = target
            
            label_pred.append(pred)
            label_gt.append(label_cls)
            
            positive_save_path = os.path.join(save_path, 'positive')
            negative_save_path = os.path.join(save_path, 'negative')

            if pred == 1:
                save_folder = positive_save_path
            else:
                save_folder = negative_save_path

            name = name.split('.')[0]
            image_2d_normalized = (image_2d - np.min(image_2d)) / (np.max(image_2d) - np.min(image_2d)) * 255
            image_2d_normalized = image_2d_normalized.astype(np.uint8)
            image_save = Image.fromarray(image_2d_normalized).convert('L')
            image_resized = image_save.resize([224,224])
            save_filename = f"{name}_slice_{idx}_label_{pred}.png"
            image_resized.save(os.path.join(save_folder, save_filename))
        

    cm = confusion_matrix(label_gt, label_pred)
    pr_result(cm, label_gt, label_pred)


def get_real_label(filenames, save_path):
    os.makedirs(os.path.join(save_path, "positive"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "negative"), exist_ok=True)

    for idx, filename in enumerate(tqdm(filenames, ncols=70)):
        image_3d = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        name = os.path.basename(filename).split('.')[0]  # 移除文件扩展名
        label = sitk.GetArrayFromImage(sitk.ReadImage(filename.replace("image", "label")))

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    train_csv_path = "../data/BraTS2020/splits/train.csv"
    valid_csv_path = "../data/BraTS2020/splits/valid.csv"
    train_csv = pd.read_csv(train_csv_path)
    valid_csv = pd.read_csv(valid_csv_path)

    train_save_path = '../data/BraTS2020/CLIP_label/train'
    valid_save_path = "../data/BraTS2020/CLIP_label/valid"

    train_filenames = train_csv['image_pth']
    valid_filenames = valid_csv['image_pth']
    

    get_clip_label(train_filenames, model, preprocess, train_save_path)
    get_real_label(valid_filenames, valid_save_path)





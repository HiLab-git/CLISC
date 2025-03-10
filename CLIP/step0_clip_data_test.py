# 保存为了resnet50，预处理后的图

import json
import os
import torch
import clip
from tqdm import tqdm
import glob
from PIL import Image
import SimpleITK as sitk
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
    
device = "cuda:3" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)


root_dir = '/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/volume_pre/image/valid' # 用验证集val调整训练参数，用含噪声的train来训练后续的resnet
root_dir_label = '/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/volume_pre/label/valid'
filenames = glob.glob(root_dir + "/*.nii.gz") 
print(len(filenames))



def classify_thresh(probs, threshold = 0.6):
    max_prob = np.max(probs)
    pred_cls = np.argmax(probs)

    if max_prob >= threshold:
        return pred_cls, True
    else:
        return -1, False

# text = clip.tokenize(["an image of brain tissue with uniform signal intensity and no abnormal hyperintense areas",
#                      "an image of brain tissue with tumor, depicted by a hyperintense abnormal signal distribution"]).to(device)

# text = clip.tokenize(["an image of brain tissue with uniform signal intensity and no abnormal hyperintense areas",
#                       "an image of brain tissue with Glioma tumor, depicted by a hyperintense abnormal signal distribution"]).to(device)

# text = clip.tokenize(["an image of brain tissue showing uniform signal intensity, excluding common normal hyperintensities such as cerebrospinal fluid in ventricles and perivascular spaces",
#                       "an image of brain tissue with Glioma tumor, depicted by a hyperintense abnormal signal distribution"]).to(device)

# text = clip.tokenize(["an image of brain tissue showing typical signal intensity without any regions of abnormal intensity or suspicious mass",
#                       "an image of brain tissue showing a tumor with uneven hyperintensity and irregular borders distinct from surroundings"]).to(device)


# best
json_path = "/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/CLIP/prompt.json"
json_file = open(json_path, encoding='utf-8')
prompts = json.load(json_file)


prompt_normal = "This MRI scan shows a normal occipital lobe without any enhancing lesions or abnormalities."
prompt_abnormal = "This MRI scan highlights a contrast-enhancing lesion in the occipital lobe, typical of a malignant tumor."

label_pred, label_gt = [], []


text = clip.tokenize([prompt_normal, prompt_abnormal]).to(device)


for idx in range(len(filenames)): 
    filename = filenames[idx]
    # [扫描层数, h, w]，每张大小是[h, w]
    image_3d = sitk.GetArrayFromImage(sitk.ReadImage(filename))
    name = os.path.basename(filename)
    label = sitk.GetArrayFromImage(sitk.ReadImage(root_dir_label + '/' + name))

    slice_list = []
    for i in range(image_3d.shape[0]):  
        image_2d = image_3d[i, :, :]  # 提取二维图像
        if label[i,:,:].sum():
            label_cls = 1
        else:
            label_cls = 0

        image_pil = Image.fromarray(image_2d)  
        image_preprocessed = preprocess(image_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_preprocessed)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image_preprocessed, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            pred, is_certain = classify_thresh(probs, threshold = 0.6)  ### 根据置信度选择

        if is_certain: 
            slice_list.append([i, pred, label_cls, image_2d])

    
    for i, (idx, pred, label_cls, image_2d) in enumerate(slice_list):
        # 使用周围的标签来定义此处的标签
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


    


cm = confusion_matrix(label_gt, label_pred)
print('normal_prompt: ', prompt_normal)
print('abnormal_prompt: ', prompt_abnormal)
pr_result(cm, label_gt, label_pred)





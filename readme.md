
# CLISC: Bridging CLIP and SAM by Enhanced CAM for Unsupervised Brain Tumor Segmentation


## 1. Environment
```bash
# install pytorch_grad_cam
python /CLIP/pytorch-grad-cam-master/setup.py install
pip install -r requirements.txt
```

## 2. Structure

![struct](structure/structure.png)
In this paper, we propose a novel unsupervised brain tumor segmentation method by adapting foundation models (CLIP and SAM) rather than directly using them for inference. The contribution is three-fold. First, we propose a framework named **CLISC** that bridges CLIP and SAM by enhanced Class Activation Maps (CAM) for **unsupervised brain tumor segmentation**, where image-level labels obtained by CLIP is used to train a classification model that obtains CAM, and the CAM  is used to generate prompts for SAM to obtain pseudo segmentation labels.  Second, to obtain high-quality prompts, we propose an **Adaptive Masking-based Data Augmentation (AMDA) strategy** for improved CAM quality. Thirdly, to reject low-quality segmentation pseudo-labels, we propose a **SAM-Seg Similarity-based Filtering (S3F) strategy** in a self-learning method for training a segmentation model. Evaluation on the BraTS2020 dataset shows that our method outperforms five state-of-the-art unsupervised segmentation methods by more than 10 percentage points with an average DSC of 85.60\%. Besides, our approach outperforms zero-shot SAM inference, achieving performance on par with fully supervised learning.


## 3. How to use
### 3.1 Dataset Download
Click [here](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) to download BraTS2020 and place it as follows:
```bash
data_BraTS/BraTS2020_TrainingData
```
### 3.1 Data preprocessing
We use 361 flair images from the BraTS2020_TrainingData for unsupervised segmentation.
The BraTS2020 is split into train:valid:test=7:1:2. The preprocessing methods include: cropping the area without brain in the MRI slice and homogenization.
```bash
python CLIP/Preprocessing.py
# generate processed data to data_BraTS/volume_pre
```

## 3.2 CLISC for Unsupervised Segmentation
### (a) Enhanced CAM from CLIP-derived image labels
```bash
python CLIP/step0_get_clip_label.py # get clip label
python CLIP/step0_get_real_label.py # get real label to calculate accuracy
python CLIP/step1_bare_res50.py # train res50 with clip label
python CLIP/step2-1_camv5.py --stage raw # get raw cam 
python CLIP/step2-2_binary_process.py --stage raw # get binarized cam
python CLIP/step2-3_post_process.py --stage raw # stage the largest component
python CLIP/step2-4_get_bounding_area # obtain bounding box of raw cam with postprocessing
python CLIP/step2-5_random_hidden.py # Adaptive Masking-based Data Augmentation (AMDA)
python CLIP/step1_aug_res50.py # train res50 with clip label and augmented images
python CLIP/step2-1_camv5.py --stage aug # get enhanced cam 
python CLIP/step2-2_binary_process.py --stage aug # get binarized cam
python CLIP/step2-3_post_process.py --stage aug # stage the largest component
```
### (b) Pseudo-Label for Segmentation based on SAM
```bash
python CLIP/step3_bbox_fore.py # inference by SAM with Enhanced CAM-derived bounding box and points
```

### (c) Self_Training with low-quality label filtering
```bash
python Code_seg_BraTS/train_fully_supervised_3D.py # train a 3D UNet with SAM-derived pseudo label
python Code_seg_BraTS/step4-1_U-net_reget_pseg.py # get pesudo label for traning set by UNet
python Code_seg_BraTS/step4-2_SAM_Pseg.py # SAM-Seg Similarity-based Filtering (S3F)
python Code_seg_BraTS/train_fully_supervised_3D.py # Self-Traning with low-quality label filtering
```

### (d) Test
```bash
# Note that you need to set the model path in the code
python Code_seg_BraTS/test_3D.py --valid # inference on validation set.
python Code_seg_BraTS/test_3D.py --test # inference on test set.
```


## 4. Citation 
```bash
@article{ma2025clisc,
  title={CLISC: Bridging clip and sam by enhanced cam for unsupervised brain tumor segmentation},
  author={Ma, Xiaochuan and Fu, Jia and Liao, Wenjun and Zhang, Shichuan and Wang, Guotai},
  journal={arXiv preprint arXiv:2501.16246},
  year={2025}
}
```

# Weakly supervised brain tumor segmentation with deep learning


## 目录
- [环境]()
- [整体流程]()
- [步骤0：data pre-processing]()
- [步骤1：CLIP prompts & pseudo classification]()
- [步骤2：Resnet50 training, CAM & aggregated mask]()
- [步骤3：SAM prompts & pseudo segmentation]()
- [步骤4：UNet 3D training]()

## 环境
```bash
# 安装pytorch_gram_cam库
python /CLIP/pytorch-grad-cam-master/setup.py install
# 更改环境
conda activate FYP
# 项目目录
cd /media/ubuntu/data3/xuzhen/CLIP
```

## 整体流程
如图所示，项目的整体流程。论文：temp/2613991_Xu_UESTC4006P_Report_23_24.pdf

![struct](temp/struct.png)

该项目描述了一个零次分割的流程，共分为四个步骤。在此之前首先介绍数据预处理和存放。

## 步骤0：数据预处理
使用BraTS2020数据集，train:valid:test=7:1:2。预处理方法包括：裁减MRI slice中没有大脑的区域，均一化。预处理代码：
```bash
python CLIP/preprocessing.py
```
处理前后的数据集目录：
```bash
# 处理前
data_BraTS/volume
# 处理后
data_BraTS/volume_pre
```
对比处理前后的图像：

![preprocess](temp/preprocess.jpg)

## 步骤1：伪分类生成
在第一步中，在二分类标签（CLS）的监督下，设计文本提示以指导CLIP进行零次分类，从而生成伪类标签（P_cls）。具体步骤如下：
1. 使用CLIP提示工程生成CLIP模型所需的文本提示。
2. 使用CLIP模型对MRI轴向切片进行分类，并通过置信度选择过程筛选出高置信度的结果。
```bash
# 先用valid调整prompts，再在train上运行_train获得P_cls
python CLIP/step0_clip_res50_train.py
# resnet50的分类准确率需要用ground truth计算，运行以下得到gt
python CLIP/step0_res50_val.py
```
```bash
# input
data_BraTS/volume_pre
# output
data_BraTS/CLIP_label_resnet50
```
置信度值：0.6。使用的prompt：

p0: an image of brain tissue showing typical signal intensity without any regions of abnormal intensity or suspicious mass

p1: an image of brain tissue showing a tumor with uneven hyperintensity and irregular borders distinct from surroundings

## 步骤2-1：粗定位，训练resnet50
在第二步中，使用获得的P_cls训练一个分类网络ResNet50，具体步骤如下：

```bash
# 训练resnet
python CLIP/step1_res50_method1.py
```
```bash
# input
data_BraTS/volume_pre
# output
data_BraTS/CLIP_label_resnet50
```
其中，resnet50的分类头改为2分类。/CLIP 文件夹中还有另外几种resnet训练方法，但计算到CAM时候发现提升分类准确率不会对CAM dice有很高的提升。所以只需要用method1就可以了。

## 步骤2-2：粗定位，生成aggregated mask
1. 从训练好的ResNet50中提取CAM（Class Activation Map）图像。
2. 使用传统的阈值分割方法生成初步的分割掩码。
3. 聚合这些初步的掩码以形成一个聚合的掩码。

![agg](temp/agg.png)

```bash
# 生成CAM
python CLIP/step2-1_camv5.py
# 生成传统阈值mask
python CLIP/step2-2_adaptive_thresh.py
# 生成aggregated mask
ptrhon CLIP/step2-3_append_cams.py
python CLIP/step2-4_post_process.py

# 如果只用CAM的话，寻找CAM的最佳二值化阈值
python CLIP/step3_find_best_threshold.py --method "layercam_l3" --region 3
```
在valid上调整参数。传统阈值方法的最佳二值化阈值为0.65，valid的保存路径：
```bash
# input
data_BraTS/volume_pre/valid
# output 1: 传统阈值方法生成的mask；after step 2-2
data_BraTS/adap_thresh
# output 2: 直接将传统mask与cam相乘；after step2-3
data_BraTS/processed_cam/binary_th0
# output 3: 后处理去噪；after step2-4
data_BraTS/processed_cam/binary_th0/cam_mor #经过开运算
data_BraTS/processed_cam/binary_th0/cam_lcnt #两种连通域：2d或者3d
data_BraTS/processed_cam/binary_th0/cam_lcnt_2d
```
test保存路径：
```bash
# input
data_BraTS/volume_pre/test
# output 
data_BraTS/test/test_processed/cam_lcnt_2d
```


## 步骤3：伪分割生成
在第三步中，从CAM图像和传统阈值分割方法生成的掩码中提取适合的提示，生成用于SAM的提示。具体步骤如下：
1. 从aggregated mask提取SAM模型所需的提示。
2. 使用SAM模型对聚合的掩码进行分割，生成伪分割标签（P_seg）。
```bash
# SAM prompts & P_seg generation
python CLIP/step4_bbox_fore.py
```
valid上调整策略：5个前景点(强度前10%，20%，30%...的点)+2d连通域下的bounding box(不扩展,d=0)。可视化结果中，蓝色表示p_seg，星星表示前景点，绿色框表示bounding box。
```bash
f_path = '/media/ubuntu/data3/xuzhen/data_BraTS'
# input
cam_folder = f_path + '/processed_cam/binary_th0/cam_lcnt_3d'
bbx_cam_folder = cam_folder
# output (testing)
output_path = f_path + '/sam/testing/bb/' # 二值化mask
out_point_fig_dir = f_path + '/sam/testing/box/' # 可视化结果
# output (best)
output_path = f_path + '/sam/best/bb/'
out_point_fig_dir = f_path + '/sam/best/box/'
```
train & test save dir:
```bash
# train output
output_path = f_path + '/train/train_sam/bb/'
out_point_fig_dir = f_path + '/train/train_sam/box/'
# test output
output_path = f_path + '/test/test_sam/bb/'
out_point_fig_dir = f_path + '/test/test_sam/box/'
```
以往实验结果：temp/sam.xlsx

## 步骤4：细化伪分割
在第四步中，利用P_seg训练一个分割网络U-Net 3D，以获得最终的细化分割标签。具体步骤如下：
1. 将P_seg作为输入，训练U-Net 3D模型。
2. 使用训练好的U-Net 3D模型生成细化的分割掩码。
```bash
# 选择监督
parser.add_argument('--sup', type=str, default='pseudo', help='model_name') # or 'label'，label即全监督
# 训练unet 3d.
python Code_seg_BraTS/train_fully_supervised_3D.py
# 测试unet 3d.
python Code_seg_BraTS/test_3D.py
```

这四个步骤协同工作，最终实现zero-shot segmetation。
import SimpleITK as sitk
import numpy as np
from skimage import exposure

def brain_bbox(data, gt):
    mask = (data != 0)
    brain_voxels = np.where(mask != 0)
    mask = (data != 0).astype(np.uint8)
    brain_voxels_mask = np.where(mask != 0)
    new_mask = np.zeros_like(data, dtype=np.uint8)
    new_mask[brain_voxels_mask] = 1

    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))
    data_bboxed = data[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    gt_bboxed = gt[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    mask_bboxed = new_mask[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]

    return data_bboxed, gt_bboxed, mask_bboxed


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())


def calculate_percent(img, img2):
    """
    根据原始图像 img 和处理后的图像 img2 计算 percent 参数值。
    """
    # 计算原始图像的累积分布函数（CDF）
    cdf = exposure.cumulative_distribution(img)
    
    # 找到 img2 的最大像素值
    max_value = img2.max()
    
    # 在 CDF 中找到对应的 percent 值
    percent = cdf[0][cdf[1] >= max_value][0]
    return percent


# 示例
p = "/data/mxc/CLISC/data/BraTS2020/raw_data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii"
data = sitk.GetArrayFromImage(sitk.ReadImage(p))
lab = sitk.GetArrayFromImage(sitk.ReadImage(p.replace("flair", "seg")))
img, lab, mask = brain_bbox(data, lab)
img2 = sitk.GetArrayFromImage(sitk.ReadImage("/data/mxc/CLISC/data/BraTS2020/image/valid/BraTS20_Training_001.nii.gz"))

# 计算 percent
percent = calculate_percent(img, img2)
print(f"Calculated percent: {percent}")
percent = 0.57
img = (img - img.mean()) / img.std()
img = MedicalImageDeal(img, percent=0.999).valid_img


print(img.mean(), img.std(), img.min(), img.max())
print(img2.mean(), img2.std(), img2.min(), img2.max())

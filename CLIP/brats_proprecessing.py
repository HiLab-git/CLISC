import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
import nibabel as nib
import SimpleITK as sitk
import glob

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

    # return data_bboxed, gt_bboxed
    return data_bboxed, gt_bboxed, mask_bboxed


def volume_bounding_box(data, gt, expend=0, status="train"):
    data, gt = brain_bbox(data, gt)
    print(data.shape)
    mask = (gt != 0)
    brain_voxels = np.where(mask != 0)
    z, x, y = data.shape
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))

    minZidx_jitterd = max(minZidx - expend, 0)
    maxZidx_jitterd = min(maxZidx + expend, z)
    minXidx_jitterd = max(minXidx - expend, 0)
    maxXidx_jitterd = min(maxXidx + expend, x)
    minYidx_jitterd = max(minYidx - expend, 0)
    maxYidx_jitterd = min(maxYidx + expend, y)

    data_bboxed = data[minZidx_jitterd:maxZidx_jitterd,
                       minXidx_jitterd:maxXidx_jitterd, minYidx_jitterd:maxYidx_jitterd]
    print([minZidx, maxZidx, minXidx, maxXidx, minYidx, maxYidx])
    print([minZidx_jitterd, maxZidx_jitterd,
           minXidx_jitterd, maxXidx_jitterd, minYidx_jitterd, maxYidx_jitterd])

    if status == "train":
        gt_bboxed = np.zeros_like(data_bboxed, dtype=np.uint8)
        gt_bboxed[expend:maxZidx_jitterd-expend, expend:maxXidx_jitterd -
                  expend, expend:maxYidx_jitterd - expend] = 1
        return data_bboxed, gt_bboxed

    if status == "test":
        gt_bboxed = gt[minZidx_jitterd:maxZidx_jitterd,
                       minXidx_jitterd:maxXidx_jitterd, minYidx_jitterd:maxYidx_jitterd]
        return data_bboxed, gt_bboxed


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size=volume.shape)
#     out[volume == 0] = out_random[volume == 0]
    out = out.astype(np.float32)
    return out


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())


# 处理flair数据
all_flair = glob.glob("/media/ubuntu/data3/maxiaochuan/data/BraTS2020_TrainingData/*/*_flair.nii")
# all_flair = glob.glob("flair/*_flair.nii.gz")

# 在处理flair数据的循环中增加保存mask的代码
for p in all_flair:
    data = sitk.GetArrayFromImage(sitk.ReadImage(p))
    lab = sitk.GetArrayFromImage(sitk.ReadImage(p.replace("flair", "seg")))
    img, lab, mask = brain_bbox(data, lab)  # 获取mask
    img = MedicalImageDeal(img, percent=0.999).valid_img
    # img = itensity_normalize_one_volume(img)
    lab[lab > 0] = 1
    uid = p.split("/")[-1]
    
    # 保存图像和标签
    # sitk.WriteImage(sitk.GetImageFromArray(img), f"/media/ubuntu/data3/maxiaochuan/data/BraTS2020_preprocess/flair_image/{uid}")
    # sitk.WriteImage(sitk.GetImageFromArray(lab), f"/media/ubuntu/data3/maxiaochuan/data/BraTS2020_preprocess/label/{uid}")
    
    # 保存mask
    sitk.WriteImage(sitk.GetImageFromArray(mask), f"/media/ubuntu/data3/maxiaochuan/data_BraTS/BraTS2020_preprocess/mask/{uid}")
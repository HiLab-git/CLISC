from skimage import measure
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from medpy import metric
import torch
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def largestConnectComponent(binaryimg):
    label_image, num = measure.label(binaryimg, background=0, return_num=True)
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    if num > 1:
        for region in measure.regionprops(label_image):
            if (region.area < areas[-1]):
                # print(region.area)
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1], coordinates[2]] = 0
    label_image = label_image.astype(np.int8)
    label_image[np.where(label_image > 0)] = 1

    return label_image

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0
    
    
def test_single_volume(image, label, net, classes, patch_size=[256, 256, 256], save_name=None, postprocessing=False):
    image, label = image.squeeze().cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        z, x, y = image.shape
        image = zoom(image, (patch_size[0] / z, patch_size[1] / x, patch_size[2] / y), order=0)
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction = zoom(out, (z / patch_size[0], x / patch_size[1], y / patch_size[2]), order=0)
            if postprocessing:
                prediction = largestConnectComponent(prediction)
    if save_name is not None:
        sitk.WriteImage(sitk.GetImageFromArray(prediction), save_name)
    
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list
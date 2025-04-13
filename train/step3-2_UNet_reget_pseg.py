import numpy as np
import pandas as pd
from networks.unet_3D import unet_3D, BiNet, TriNet
import torch
import os
import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import zoom
from skimage import measure

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

if __name__ == "__main__": 
    U_net_model_path = './UNet3D_model/SAM_Sup/unet_3D/unet_3D_best_model.pth'
    net = unet_3D(n_classes=2, in_channels=1).cuda()
    net.load_state_dict(torch.load(U_net_model_path))
    output_path = "../data/BraTS2020/UNet_label"
    os.makedirs(output_path, exist_ok=True)

    files = pd.read_csv("../data/BraTS2020/splits/train.csv")["image_pth"]

    net.eval()
    # [1, 1, 128, 128, 128]

    all_dices = []
    for i, file_path in enumerate(files):
        label_path = file_path.replace('image', 'label')
        image = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        z, x, y = image.shape
        image = zoom(image, (128 / z, 128 / x, 128 / y), order=0)
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(image), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction = zoom(out, (z / 128, x / 128, y / 128), order=0)
            prediction = largestConnectComponent(prediction)
        
        
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        if not prediction.sum():
            print(f'{i} / {len(files)}, erase one sample without any anomaly detected')
            continue

        dice_score = metric.binary.dc(prediction, label)
        print(f"{i} / {len(files)}, {dice_score}")
        sitk.WriteImage(sitk.GetImageFromArray(prediction), os.path.join(output_path, os.path.basename(file_path)))
        all_dices.append(dice_score)

    all_dices = np.array(all_dices)
    print(all_dices)
    print(all_dices.mean(), all_dices.std(), all_dices.max(), all_dices.min())
            
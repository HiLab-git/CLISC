import numpy as np
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

U_net_model_path = '/media/ubuntu/maxiaochuan/CLISC/Code_seg_BraTS/model/BraTs2020_Un_Supervised_gce/unet_3D/iter_2000_dice_0.8333.pth'

device = "cuda:3" if torch.cuda.is_available() else "cpu"

net = unet_3D(n_classes=2, in_channels=1).to(device)

net.load_state_dict(torch.load(U_net_model_path))

output_path = "/media/ubuntu/maxiaochuan/CLISC/Code_seg_BraTS/Unet_Pseg"

os.makedirs(output_path, exist_ok=True)

image_path = '/media/ubuntu/maxiaochuan/CLISC/data_BraTS/volume_pre/image/train'


all_dices = []
files = os.listdir(image_path)
net.eval()
# [1, 1, 128, 128, 128]

for i, file in enumerate(files):
    file_path = os.path.join(image_path, file)
    label_path = file_path.replace('image', 'label')
    image = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
    z, x, y = image.shape
    image = zoom(image, (224 / z, 224 / x, 224 / y), order=0)
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(image), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        prediction = zoom(out, (z / 224, x / 224, y / 224), order=0)
        prediction = largestConnectComponent(prediction)
    
    
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
    if not prediction.sum():
        print(f'{i} / {len(files)}, erase one sample without any anomaly detected')
        continue

    dice_score = metric.binary.dc(prediction, label)
    print(f"{i} / {len(files)}, {dice_score}")
    sitk.WriteImage(sitk.GetImageFromArray(prediction), os.path.join(output_path, file))
    all_dices.append(dice_score)

all_dices = np.array(all_dices)
print(all_dices)
print(all_dices.mean(), all_dices.std(), all_dices.max(), all_dices.min())
        
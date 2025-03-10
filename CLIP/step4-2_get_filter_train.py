import os
import shutil

input_path = "/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/volume_pre/pseudo/train"
train_image_path = "/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/volume_pre/image/train"
output_image_path = "/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/volume_pre/image/filter_train"
pesudo_labels = os.listdir(input_path)
train_images = os.listdir(train_image_path)

os.makedirs(output_image_path, exist_ok=True)

for file in pesudo_labels:
    for image in train_images:
        if file == image:
            shutil.copy(f"{train_image_path}/{file}", f"{output_image_path}/{file}")
            break
    
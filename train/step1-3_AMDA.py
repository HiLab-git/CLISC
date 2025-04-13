import os
import cv2
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import argparse

def find_bounding_box_2d(mask):
    """
    Find the bounding box of the region where mask == 1 in a 2D array.
    """
    assert mask.ndim == 2, "The mask must be a 2D numpy array."
    ones_indices = np.where(mask == 1)
    if not ones_indices[0].size:
        return None, None, None, None
    min_y = np.min(ones_indices[0])
    max_y = np.max(ones_indices[0])
    min_x = np.min(ones_indices[1])
    max_x = np.max(ones_indices[1])
    return min_y, max_y, min_x, max_x

def process_cam_and_generate_bounding_boxes(cam_path, bounding_box_path):
    """
    Process CAM files to generate bounding boxes and save them as text files.
    """
    os.makedirs(bounding_box_path, exist_ok=True)
    for file in tqdm(os.listdir(cam_path)):
        cam_file = os.path.join(cam_path, file)

        if not os.path.exists(cam_file):
            continue
        cam = sitk.GetArrayFromImage(sitk.ReadImage(cam_file))
        for slice_idx in range(cam.shape[0]):
            cam_slice = cam[slice_idx, :, :]
            min_y, max_y, min_x, max_x = find_bounding_box_2d(cam_slice)
            if min_y is None:
                continue
            txt_path = os.path.join(bounding_box_path, file.replace('.nii.gz', f'_slice_{slice_idx}.txt'))
            with open(txt_path, 'w') as f:
                f.write(f"{min_y} {max_y} {min_x} {max_x}")

def apply_cam_based_hiding(input_path, bounding_box_path, cam_path, output_path, dilation=10):
    """
    Apply CAM-based hiding to images based on bounding boxes and CAM intensity.
    """
    os.makedirs(output_path, exist_ok=True)
    for phe in ['positive', 'negative']:
        os.makedirs(os.path.join(output_path, phe), exist_ok=True)
        for file in tqdm(os.listdir(os.path.join(input_path, phe))):
            img = cv2.imread(os.path.join(input_path, phe, file))
            if phe == "positive":
                bounding_box_file = os.path.join(bounding_box_path, file.replace('_label_1.png', '.txt'))
            else:
                bounding_box_file = os.path.join(bounding_box_path, file.replace('_label_0.png', '.txt'))
            cam_file = os.path.join(cam_path, phe, file.replace('.png', '.nii.gz'))
            if not os.path.exists(bounding_box_file) or not os.path.exists(cam_file):
                img[img > ((img.min() + (img.max() - img.min()) * 0.8))] = 0
                cv2.imwrite(os.path.join(output_path, phe, file), img)
                continue

            with open(bounding_box_file, 'r') as f:
                min_y, max_y, min_x, max_x = map(int, f.readline().split())
            cam = sitk.GetArrayFromImage(sitk.ReadImage(cam_file))

            # Resize bounding box and CAM to match image dimensions
            x_ratio = 224 / cam.shape[1]
            y_ratio = 224 / cam.shape[2]
            min_x = int(min_x * x_ratio)
            max_x = int(max_x * x_ratio)
            min_y = int(min_y * y_ratio)
            max_y = int(max_y * y_ratio)
            cam = cv2.resize(cam, (224, 224))

            # Compute threshold for hiding
            min_one = cam[min_x:max_x + 1, min_y:max_y + 1].min()
            max_one = cam[min_x:max_x + 1, min_y:max_y + 1].max()
            d = max_one - min_one
            k = 8
            threshold = min_one + k / 10 * d

            # Apply hiding based on CAM intensity
            for i in range(min_x, max_x + 1):
                for j in range(min_y, max_y + 1):
                    if cam[i, j] > threshold:
                        img[i, j] = 0

            # Save the modified image
            cv2.imwrite(os.path.join(output_path, phe, file), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    base_path = "../data/BraTS2020"
    input_path = os.path.join(base_path, "CLIP_label", "train")
    cam_path = os.path.join(base_path, "LayerCAM", "raw", "train", "cam_post")
    bounding_box_path = os.path.join(base_path, "CLIP_label", "bounding_box")
    output_path = os.path.join(base_path, "CLIP_label", "aug_train")

    # Step 1: Generate bounding boxes from CAMs
    print("Generating bounding boxes...")
    process_cam_and_generate_bounding_boxes(cam_path, bounding_box_path)

    # Step 2: Apply CAM-based hiding
    print("Applying CAM-based hiding...")
    apply_cam_based_hiding(input_path, bounding_box_path, cam_path, output_path)
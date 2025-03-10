import argparse
import os
import shutil
from glob import glob
import pandas as pd
import torch

from networks.unet_3D import unet_3D, BiNet, TriNet
# from test_3D_util import test_all_case
import numpy as np
from torch.utils.data import DataLoader
from dataloaders.brats2019 import BraTS2020
from scipy.ndimage import zoom
from util import *



def Inference(FLAGS, num_classes = 2):
    device = FLAGS.device
    snapshot_path = "model/{}/{}".format(FLAGS.exp, FLAGS.save_model_name)
    test_save_path = "model/{}/{}/Prediction/{}".format(FLAGS.exp, FLAGS.save_model_name, FLAGS.stage)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    
    # save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    save_mode_path = "/media/ubuntu/maxiaochuan/CLISC/Code_seg_BraTS/model/BraTs2020_Un_Supervised_ce/unet_3D/iter_1800_dice_0.8173.pth"
    # save_mode_path = "/media/ubuntu/maxiaochuan/CLISC/Code_seg_BraTS/model/BraTs2020_Un_Supervised_ce/unet_3D/Stage2_0.8387_0.856_highother.pth"
    if FLAGS.model == 'unet_3D':
        net = unet_3D(n_classes=num_classes, in_channels=1).to(device)
    elif FLAGS.model == 'BiNet':
        net = BiNet(n_classes=num_classes, in_channels=1).to(device)
    elif FLAGS.model == 'trinet':
        net = TriNet(n_classes=num_classes, in_channels=1).to(device)
    else:
        print('wrong model name')
    
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    db_test = BraTS2020(base_dir=FLAGS.root_path, split=FLAGS.stage)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    all_metric = []
    metric_list_dice, metric_list_hd95 = [], []
    metric_csv = []
    for i_batch, sampled_batch in enumerate(testloader):
        save_name = '{}/{}'.format(test_save_path, sampled_batch['filename'][0])
        metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], net, classes=num_classes, patch_size=FLAGS.patch_size, save_name=save_name, device=device)
        metric_i = np.array(metric_i)[0]
        print(sampled_batch['filename'][0], metric_i[0], metric_i[1])
        
        metric_list_dice.append(metric_i[0])
        metric_list_hd95.append(metric_i[1])
        metric_csv.append({"filename": save_name, "dice": metric_i[0], "hd95": metric_i[1]})

        all_metric.append([sampled_batch["label"].sum(), metric_i[0], metric_i[1]])
    metric_list_dice = np.array(metric_list_dice)
    metric_list_hd95 = np.array(metric_list_hd95)
    
    metric_csv.append({"filename": 'mean', "dice": metric_list_dice.mean(), "hd95": metric_list_hd95.mean()})
    metric_csv.append({"filename": 'std', "dice": metric_list_dice.std(), "hd95": metric_list_hd95.std()})

    dataframe = pd.DataFrame(metric_csv, columns=["filename", "dice", "hd95"])
    dataframe.to_csv('{}/results_{}.csv'.format(os.path.dirname(test_save_path), FLAGS.stage), index=False)
    
    dice_mean = metric_list_dice.mean()
    dice_std = metric_list_dice.std()
    hd95_mean = metric_list_hd95.mean()
    hd95_std = metric_list_hd95.std()
    
    all_metric.sort()
    for a, b, c in all_metric:
        print(a, b, c)
    
    return dice_mean, dice_std, hd95_mean, hd95_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/media/ubuntu/maxiaochuan/CLISC/data_BraTS/volume_pre/image', help='Name of Experiment')
    parser.add_argument('--exp', type=str, default='BraTs2020_Un_Supervised_ce', help='experiment_name')
    parser.add_argument('--stage', type=str, default='test')
    parser.add_argument('--patch_size', type=list,  default=[128, 128, 128], help='patch size of network input')
    parser.add_argument('--model', type=str, default='unet_3D', help='model_name')
    parser.add_argument('--save_model_name', type=str, default='unet_3D', help='model_name')
    parser.add_argument("--postprocessing", type=str2bool, default=False)
    parser.add_argument("--device", type=int, default=0, help="cuda number")

    FLAGS = parser.parse_args()
    dice_mean, dice_std, hd95_mean, hd95_std = Inference(FLAGS)    

    print(FLAGS.save_model_name, FLAGS.stage, round(dice_mean * 100.0, 2), round(dice_std * 100.0, 2), round(hd95_mean, 2), round(hd95_std, 2))

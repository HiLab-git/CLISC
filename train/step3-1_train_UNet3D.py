import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.brats2020 import (BraTS2020, RandomCrop, RandomRotFlip, ToTensor)
from networks.net_factory_3d import net_factory_3d
from utils import losses
from utils.utils import *
import torch.nn.functional as F

# gce loss

def config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--training_csv', type=str, default='../data/BraTS2020/splits/train.csv', help='path to the training csv file')
    parser.add_argument('--exp', type=str, default='SAM_Sup', choices=["SAM_Sup", "Label_Sup", "UNet_Sup"], help='experiment_name')
    parser.add_argument('--model', type=str, default='unet_3D', help='model_name')
    parser.add_argument('--max_epoch', type=int, default=400, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[128, 128, 128], help='patch size of network input')
    parser.add_argument('--seed', type=int,  default=2023, help='random seed')
    
    return parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    training_csv = args.training_csv
    batch_size = args.batch_size
    num_classes = 2
    model = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
    db_train = BraTS2020(
        csv_path=training_csv,
        exp = args.exp,
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(args.patch_size),
            ToTensor(),
        ]),
    )
    db_val = BraTS2020(csv_path="../data/BraTS2020/splits/valid.csv", exp="Label_Sup", transform=transforms.Compose([ToTensor()]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    
    
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)
     

    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_iterations = args.max_epoch * len(trainloader)
    best_performance = 0.0
    iterator = tqdm(range(args.max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):


            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)
            loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_ 

            iter_num = iter_num + 1

            # logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num > 0 and iter_num % 200 == 0: ## 200
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)


                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]

                if performance > best_performance:
                    best_performance = performance
                    save_model_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_model_path)
                    torch.save(model.state_dict(), save_best)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()


            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    return "Training Finished!"


if __name__ == "__main__":
    args = config()
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = f"./UNet3D_model/{args.exp}/{args.model}"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
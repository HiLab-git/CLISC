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
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.brats2019 import (BraTS2020, RandomCrop, RandomRotFlip, ToTensor)
from networks.net_factory_3d import net_factory_3d
from utils import losses
from util import *
import torch.nn.functional as F

# gce loss
class GeneralizedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, q=0.7):
        """
        q: The hyperparameter for controlling the degree to which the loss focuses on hard examples. 
           Typically q < 1. When q=1, it becomes equivalent to the standard cross-entropy loss.
        """
        super(GeneralizedCrossEntropyLoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        """
        logits: The logits predictions from the model (before softmax).
                Shape [batch_size, num_classes].
        targets: The ground truth labels.
                 Shape [batch_size].
        """
        ## avoid nan
        epsilon = 1e-8

        probabilities = F.softmax(logits, dim=1)
        true_probabilities = probabilities.gather(dim=1, index=targets.unsqueeze(1)).squeeze()
        true_probabilities = torch.clamp(true_probabilities, min=epsilon)
        loss = (1 - true_probabilities ** self.q) / self.q
        return loss.mean()


def config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root_path', type=str, default='/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/data_BraTS/volume_pre/image', help='Name of Experiment')
    parser.add_argument('--exp', type=str, default='BraTs2020_Un_Supervised_ce', help='experiment_name')
    parser.add_argument('--model', type=str, default='unet_3D', help='model_name')
    parser.add_argument('--save_model_name', type=str, default='unet_3D')
    parser.add_argument('--sup', type=str, default='pseudo', help='model_name') # or 'label'
    parser.add_argument('--max_epoch', type=int, default=400, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[128, 128, 128], help='patch size of network input')
    parser.add_argument('--seed', type=int,  default=2023, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='cuda number')
    parser.add_argument('--train_set', type=str, default='bare', help='unet_3D_train_set')
    
    return parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    device = args.device
    num_classes = 2
    model = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes, device=device)
    db_train = BraTS2020(base_dir=train_data_path,
                         split='train',
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]),
                         sup=args.sup,
                         train_set=args.train_set)
    db_val = BraTS2020(base_dir=args.root_path, split="valid", transform=transforms.Compose([ToTensor()]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    
    
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
     
    # ce_loss = GeneralizedCrossEntropyLoss(q=0.85) #0.7, gce loss
    # dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_iterations = args.max_epoch * len(trainloader)
    best_performance = 0.0
    iterator = tqdm(range(args.max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):


            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            print(volume_batch.shape)
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch.long())
            # loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            # loss = 0.5 * (loss_dice + loss_ce)
            loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_ 

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            # logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num > 0 and iter_num % 200 == 0: ## 200
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, patch_size=args.patch_size, device=device)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            # if iter_num % 3000 == 0:
            #     save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
            #     torch.save(model.state_dict(), save_mode_path)
            #     logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
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

    snapshot_path = "/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/Code_seg_BraTS/model/{}/{}".format(args.exp, args.save_model_name)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
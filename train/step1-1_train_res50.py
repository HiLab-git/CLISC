import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch.nn.functional as F
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


    

def train_model(model, criterion, optimizer, scheduler, num_epochs, stage):
    best_acc = 0.0
    for i, epoch in enumerate(range(num_epochs)):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
            
            print(f'{phase}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')

            if phase == 'valid' and (i + 1) % 1 == 0:
                torch.save(model.state_dict(), f'./resnet50_model/{stage}/final_model.pth')
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), f'./resnet50_model/{stage}/iter{i + 1}_{epoch_acc:.4f}.pth')
                    torch.save(model.state_dict(), f'./resnet50_model/{stage}/best_model.pth')
            
            
                
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--stage', type=str, choices=["raw", "aug"], default='raw', help='training stage')
    args = parser.parse_args()
   
    stage = args.stage
    setup_seed(args.seed)

    dataset_path = '../data/BraTS2020/CLIP_label'

    # 创建数据集的transforms



    train_set = "train" if stage == "raw" else "aug_train"
    data_transforms = {
        train_set: transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ]),
        'valid': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x), data_transforms[x]) for x in [train_set, 'valid']}
    train_dataloader = DataLoader(image_datasets[train_set], batch_size=8, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(image_datasets['valid'], batch_size=8, shuffle=False, num_workers=4)
    dataloaders = {'train': train_dataloader, 'valid': valid_dataloader}
    dataset_sizes = {"train": len(image_datasets[train_set]), "valid": len(image_datasets['valid'])}
    class_names = image_datasets[train_set].classes

    model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Flatten(), 
        nn.Linear(num_ftrs, 256), 
        nn.ReLU(), 
        nn.Dropout(0.5),  
        nn.Linear(256, 2) 
    )


    model = model.cuda()
    criterion = nn.CrossEntropyLoss()


    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    os.makedirs(f"./resnet50_model/{stage}", exist_ok=True)

    train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=400, stage=stage)

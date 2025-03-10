import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import copy
import torch.nn.functional as F
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




dataset_path = '/media/ubuntu/maxiaochuan/CLISC/data_BraTS/CLIP_label'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 创建数据集的transforms
data_transforms = {
    'aug_train': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ]),
    'gt': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x), data_transforms[x])
                  for x in ['aug_train', 'gt']}
aug_train_dataloader = DataLoader(image_datasets['aug_train'], batch_size=8, shuffle=True, num_workers=4)
gt_dataloader = DataLoader(image_datasets['gt'], batch_size=8, shuffle=False, num_workers=4)
dataloaders = {'aug_train': aug_train_dataloader, 'gt': gt_dataloader}

dataset_sizes = {x: len(image_datasets[x]) for x in ['aug_train', 'gt']}
class_names = image_datasets['aug_train'].classes

model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')

# 更改分类头
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Flatten(), 
    nn.Linear(num_ftrs, 256), 
    nn.ReLU(), 
    nn.Dropout(0.5),  
    nn.Linear(256, 2) 
)


model = model.to(device)
criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters(), lr=0.0001)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
threshold = 0.7 

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for i, epoch in enumerate(range(num_epochs)):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['aug_train', 'gt']:
            if phase == 'aug_train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'aug_train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'aug_train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            if phase == 'aug_train':
                scheduler.step()

            epoch_acc = running_corrects.double() / dataset_sizes[phase]


            if phase == 'gt' and (i + 1) % 10 == 0:
                torch.save(model.state_dict(), f'/media/ubuntu/maxiaochuan/CLISC/CLIP/resnet_model/aug_res50_iter{i + 1}_{epoch_acc:.4f}.pth')
                torch.save(model.state_dict(), f'/media/ubuntu/maxiaochuan/CLISC/CLIP/resnet_model/aug_res50_best_model.pth')
    return 

train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=400)
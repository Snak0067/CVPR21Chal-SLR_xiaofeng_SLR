#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 14:11
# @Author  : Xiaofeng
# @File    : C3D_test.py
import os
import sys
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from Conv3D.models.Conv3D import r2plus1d_18
from Conv3D.dataset_sign_clip import Sign_Isolated
from Conv3D.train import train_epoch
from Conv3D.validation_clip import val_epoch
from collections import OrderedDict

from model_comparision.models.CNN3D import CNN3D


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


# Path setting
exp_name = 'rgb_final'
data_path = "../data-prepare/data/frame/train_frame_data"
data_path2 = "../data-prepare/data/frame/test_frame_data"
label_train_path = "../data-prepare/data/small_sample_data/label/train_labels.csv"
label_val_path = "../data-prepare/data/small_sample_data/label/test_labels.csv"
model_path = "checkpoint/{}".format(exp_name)
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(os.path.join('results', exp_name)):
    os.mkdir(os.path.join('results', exp_name))
log_path = "log/myModel_sign_CNN3D_{}_{:%Y-%m-%d_%H-%M-%S}.log".format(exp_name, datetime.now())
sum_path = "runs/myModel_sign_CNN3D_{}_{:%Y-%m-%d_%H-%M-%S}".format(exp_name, datetime.now())
# phase = 'Test'
phase = 'Train'
# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')
writer = SummaryWriter(sum_path)

# Use specific gpus
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_classes = 7  # 226
epochs = 10
# batch_size = 16
batch_size = 8
learning_rate = 1e-3  # 1e-4 Train 1e-5 Finetune
log_interval = 80
sample_size = 128
sample_duration = 32
attention = False
drop_p = 0.0
hidden1, hidden2 = 512, 256
num_workers = 0


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# Train with 3DCNN
if __name__ == '__main__':
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = Sign_Isolated(data_path=data_path, label_path=label_train_path, frames=sample_duration,
                              num_classes=num_classes, train=True, transform=transform)
    val_set = Sign_Isolated(data_path=data_path2, label_path=label_val_path, frames=sample_duration,
                            num_classes=num_classes, train=False, transform=transform)
    logger.info("Dataset samples: {}".format(len(train_set) + len(val_set)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # Create model
    model = CNN3D(sample_size=sample_size, sample_duration=sample_duration, drop_p=drop_p,
                  hidden1=hidden1, hidden2=hidden2, num_classes=num_classes)
    # model = resnet18(pretrained=True, progress=True, sample_size=sample_size, sample_duration=sample_duration,
    #                 attention=attention, num_classes=num_classes).to(device)
    # model = resnet50(pretrained=True, progress=True, sample_size=sample_size, sample_duration=sample_duration,
    #                 attention=attention, num_classes=num_classes).to(device)
    # print(model)

    model = model.to(device)
    model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                           threshold=0.0001)

    # Start training
    if phase == 'Train':
        logger.info("Training Started".center(60, '#'))
        for epoch in range(epochs):
            print('lr: ', get_lr(optimizer))
            # Train the model
            train_epoch(model, criterion, optimizer, train_loader, device, epoch, logger, log_interval, writer)

            # Validate the model
            val_loss = val_epoch(model, criterion, val_loader, device, epoch, logger, writer)
            scheduler.step(val_loss)

            # Save model
            torch.save(model.state_dict(),
                       os.path.join(model_path, "sign_CNN3D_epoch{:03d}.pth".format(epoch + 1)))
            logger.info("Epoch {} Model Saved".format(epoch + 1).center(60, '#'))
    elif phase == 'Test':
        logger.info("Testing Started".center(60, '#'))
        val_loss = val_epoch(model, criterion, val_loader, device, 0, logger, writer, phase=phase, exp_name=exp_name)

    logger.info("Finished".center(60, '#'))

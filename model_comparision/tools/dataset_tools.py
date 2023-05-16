#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/16 9:48
# @Author  : Xiaofeng
# @File    : dataset_tools.py
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Conv3D.dataset_sign_clip import Sign_Isolated, logger

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
# OPTIMIZER
ADAM_EPS = 1e-3
ADAM_WEIGHT_DECAY = 1e-8
# data_path
data_path = "../data-prepare/data/frame/train_frame_data"
data_path2 = "../data-prepare/data/frame/test_frame_data"
label_train_path = "../data-prepare/data/small_sample_data/label/train_labels.csv"
label_val_path = "../data-prepare/data/small_sample_data/label/test_labels.csv"


def get_dataset():
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
    return train_loader, val_loader

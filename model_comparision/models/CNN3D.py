#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 14:53
# @Author  : Xiaofeng
# @File    : CNN3D_Train.py
import torch
import torch.nn as nn


class CNN3D(nn.Module):
    def __init__(self, sample_size, sample_duration, drop_p, hidden1, hidden2, num_classes):
        super(CNN3D, self).__init__()
        self.sample_size = sample_size
        self.sample_duration = sample_duration

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc1 = nn.Linear(256 * (sample_duration // 8) * (sample_size // 32) * sample_size, hidden1)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=drop_p)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=drop_p)

        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout1(self.relu4(self.fc1(x)))
        x = self.dropout2(self.relu5(self.fc2(x)))

        x = self.fc3(x)

        return x


# 创建CNN3D模型
sample_size = 64
sample_duration = 16
drop_p = 0.5
hidden1 = 256
hidden2 = 128
num_classes = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN3D(sample_size, sample_duration, drop_p, hidden1, hidden2, num_classes).to(device)

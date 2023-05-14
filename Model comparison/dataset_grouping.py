# -*- coding:utf-8 -*-
# @FileName :dataset_grouping.py
# @Time :2023/5/14 10:00
# @Author :Xiaofeng
import os
import random

import pandas as pd

train_label_path = "E:/dataset/AUSTL/raw_data/train_data/train_labels.csv"
val_label_path = "E:/dataset/AUSTL/raw_data/validation_data/ground_truth.csv"
test_label_path = "E:/dataset/AUSTL/raw_data/test_data/ground_truth.csv"


def select_rows(label_path, random_numbers):
    selected_list = []
    label_file = open(label_path, 'r', encoding='utf-8')
    for line in label_file.readlines():
        lines = line.strip()
        lines_label = lines.split(',')[1]
        if int(lines_label) in random_numbers:
            selected_list.append(line)

    # 创建文件夹
    folder_path = 'data_label/'
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, os.path.basename(label_path))

    # 打开文件，以写入模式写入数据
    with open(save_path, 'w') as f:
        # 将每个元素写入文件，并在元素之间加上换行符
        for item in selected_list:
            f.write(item)
    return selected_list


if __name__ == '__main__':
    random.seed(123)
    random_numbers = [random.randint(0, 225) for i in range(5)]
    # print(select_rows(train_label_path, random_numbers))
    print(select_rows(val_label_path, random_numbers))
    # print(select_rows(test_label_path, random_numbers))

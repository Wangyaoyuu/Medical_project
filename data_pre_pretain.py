# %% Fundus

import time
import cv2
import csv
import numpy as np
import os

csv_path = '/home/wangyaoyu/data/Kaggle/train.csv'  # add csv path here
base_path = '/home/wangyaoyu/data/Kaggle/train_images/'  # add data path here

save_path = '/home/wangyaoyu/data/Kaggle/'  # add path to save the data
if not os.path.exists(save_path):
    os.makedirs(save_path)

x = []
y = []
count = 0
with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    next(csv_reader)
    for row in csv_reader:
        print(count)
        image_path = row[0] + '.png'
        file_path = base_path + image_path
        print(file_path)
        img = cv2.imread(file_path)
        img = np.asarray(cv2.resize(img, (512, 512)), dtype=np.uint8)
        class_name = int(row[1])
        if count == 0:
            x = np.expand_dims(img, axis=0)
            y.append(class_name)
        else:
            x = np.concatenate((x, np.expand_dims(img, axis=0)), axis=0)
            y.append(class_name)
        count = count + 1

np.savez(save_path + '/train_512.npz', x, y)

csv_path =  '/home/wangyaoyu/data/Kaggle/test.csv'   # add csv path here
base_path = '/home/wangyaoyu/data/Kaggle/test_images/'  # add data path here

save_path = '/home/wangyaoyu/data/Kaggle/'  # add path to save the data
if not os.path.exists(save_path):
    os.makedirs(save_path)

x = []
y = []
count = 0
with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    next(csv_reader)
    for row in csv_reader:
        print(count)
        image_path = row[0] + '.png'
        file_path = base_path + image_path
        print(file_path)
        img = cv2.imread(file_path)
        img = np.asarray(cv2.resize(img, (512, 512)), dtype=np.uint8)
        class_name = int(row[1])

        if count == 0:
            x = np.expand_dims(img, axis=0)
            y.append(class_name)
        else:
            x = np.concatenate((x, np.expand_dims(img, axis=0)), axis=0)
            y.append(class_name)
        count = count + 1

np.savez(save_path + '/test_512.npz', x, y)
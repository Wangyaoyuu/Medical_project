# %% Fundus

import time
import cv2
import csv
import numpy as np
import os

csv_path = '/home/wangyaoyu/data/fundus_data/regular-fundus-training/regular-fundus-training.csv'  # add csv path here
base_path = '/home/wangyaoyu/data/fundus_data/regular-fundus-training'  # add data path here

save_path = '/home/wangyaoyu/data/fundus_data/'  # add path to save the data
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
        image_path = row[2][24::].replace('\\','/')
        file_path = base_path + image_path
        print(file_path)
        img = cv2.imread(file_path)
        img = np.asarray(cv2.resize(img, (224, 224)), dtype=np.uint8)
        if row[1][-2] == 'l':
            class_name = int(row[4])
        else:
            class_name = int(row[5])

        if count == 0:
            x = np.expand_dims(img, axis=0)
            y.append(class_name)
        else:
            x = np.concatenate((x, np.expand_dims(img, axis=0)), axis=0)
            y.append(class_name)
        count = count + 1

np.savez(save_path + '/train_224.npz', x, y)

csv_path = '/home/wangyaoyu/data/fundus_data/regular-fundus-validation/regular-fundus-validation.csv'  # add csv path here
base_path = '/home/wangyaoyu/data/fundus_data/regular-fundus-validation/'  # add data path here

save_path = '/home/wangyaoyu/data/fundus_data/'  # add path to save the data
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
        image_path = row[2][26::].replace('\\', '/')
        file_path = base_path + image_path
        img = cv2.imread(file_path)
        img = np.asarray(cv2.resize(img, (224, 224)), dtype=np.uint8)
        if row[1][-2] == 'l':
            class_name = int(row[4])
        else:
            class_name = int(row[5])

        if count == 0:
            x = np.expand_dims(img, axis=0)
            y.append(class_name)
        else:
            x = np.concatenate((x, np.expand_dims(img, axis=0)), axis=0)
            y.append(class_name)
        count = count + 1

np.savez(save_path + '/test_224.npz', x, y)
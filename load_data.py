import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler, DataLoader
import random

def load_data(image_size):

    if image_size == 256:
        path = '/home/wangyaoyu/data/fundus_data/'
        temp = np.load(path + '/train.npz')
        trainS = np.asarray(temp['arr_0'] / 255, dtype=np.float32)
        # label = np.asarray(temp['arr_1'])
        labelTr = temp['arr_1']
        del temp

        temp = np.load(path + '/test.npz')
        testS = np.asarray(temp['arr_0'] / 255, dtype=np.float32)
        # label = np.asarray(temp['arr_1'])
        labelTs = temp['arr_1']
        del temp

        trainS = trainS.swapaxes(1,3)
        testS = testS.swapaxes(1,3)

    else:
        path = '/home/wangyaoyu/data/fundus_data/'
        temp = np.load(path + f'/train_{image_size}.npz')
        trainS = np.asarray(temp['arr_0'] / 255, dtype=np.float32)
        # label = np.asarray(temp['arr_1'])
        labelTr = temp['arr_1']
        del temp

        temp = np.load(path + f'/test_{image_size}.npz')
        testS = np.asarray(temp['arr_0'] / 255, dtype=np.float32)
        # label = np.asarray(temp['arr_1'])
        labelTs = temp['arr_1']
        del temp

        trainS = trainS.swapaxes(1,3)
        testS = testS.swapaxes(1,3)

    return trainS, labelTr, testS, labelTs

def load_data_pretrain(image_size):

    path = '/home/wangyaoyu/data/Kaggle'
    temp = np.load(path + f'/train_{image_size}.npz')
    trainS = np.asarray(temp['arr_0'] / 255, dtype=np.float32)
    # label = np.asarray(temp['arr_1'])
    labelTr = temp['arr_1']
    del temp

    temp = np.load(path + f'/test_{image_size}.npz')
    testS = np.asarray(temp['arr_0'] / 255, dtype=np.float32)
    # label = np.asarray(temp['arr_1'])
    labelTs = temp['arr_1']
    del temp

    trainS = trainS.swapaxes(1,3)
    testS = testS.swapaxes(1,3)

    return trainS, labelTr, testS, labelTs

class trainSet(Dataset):
    def __init__(self, data, label, augment=False):
        self.data, self.label, self.augment = data, label, augment
        self.size = len(label)

    def __getitem__(self, idx):
        idx = idx
        aug = random.randrange(4)

        data = self.data[idx].astype(np.float32)
        label = self.label[idx].astype(np.int64)

        if not self.augment or aug == 0: pass
        elif aug == 1:
            data = np.rot90(data, k=1, axes=(1, 2))
        elif aug == 2:
            data = np.rot90(data, k=2, axes=(1, 2))
        elif aug == 3:
            data = np.rot90(data, k=3, axes=(1, 2))

        return {'data':data.copy(), 'label':label}

    def __len__(self):
        return self.size


class testSet(Dataset):
    def __init__(self, data, label):
        self.data, self.label = data, label
        self.size = len(label)

    def __getitem__(self, idx):
        idx = idx
        data = self.data[idx].astype(np.float32)
        label = self.label[idx].astype(np.int64)

        return {'data':data.copy(), 'label':label}

    def __len__(self):
        return self.size
from kappa import *
import numpy as np


# actuals = np.array([0,1,2,3,4])
# preds = np.array([0,1,2,3,4])
######### Load Pacakges #######
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import numpy as np
import random
import os

from load_data import *
from load_model import *

import torch as pt

# %% set paths and parameters
dataset = 'Fundus'
model_name =  'Efficient'
image_size = 512
# ['ViT','ResNet50', 'ResNet50_sl1', 'Efficient', 'Efficient-onehot', 'Efficient-fine', 'Efficient-not']

seed = 0
model = load_model(model_name)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(model))

batchsize = 32
base_output_path = './output/'
weight_path = base_output_path + dataset + '/' + model_name
fileEnd = '.h5'

if not os.path.exists(weight_path):
    os.makedirs(weight_path)

srcfn = weight_path + '/model'
try:
    state = pt.load(srcfn)
    model.load_state_dict(state['model'])
    print('#model load:', srcfn)
except Exception as e:
    print('#model transfer:', srcfn, e)


trainS, labelTr, testS, labelTs = load_data(image_size)
no_class = len(np.unique(labelTr))
print('#[Test]shape: ', testS.shape, labelTs.shape)

testset = testSet(testS, labelTs)
testloader = DataLoader(testset, batch_size=batchsize,
                         num_workers=2, prefetch_factor=batchsize)

print('#testing model ...')
summary_true = []
summary_predict = []

for batchvalid in testloader:
    model.eval()
    with pt.no_grad():
        x = batchvalid['data'].clone().detach().cuda()
        yy = model(x)
        yy = yy.cpu()
        y_true = batchvalid['label']
        # y_predict = pt.argmax(yy, dim=-1)
        y_predict = yy

        summary_true.extend(y_true)
        summary_predict.extend(y_predict)


summary_predict = np.array(summary_predict)
summary_predict = np.round(summary_predict)

summary_true = np.array(summary_true)
print(summary_predict.shape)
print(summary_true.shape)
# print(len(summary_true))
odd_index = np.arange(0, 400, 2)
even_index = np.arange(1, 400, 2)

left_true = summary_true[odd_index]
left_predict = summary_predict[odd_index]

right_true = summary_true[even_index]
right_predict = summary_predict[even_index]

compare_predict1 = left_predict > right_predict
compare_true1 = left_true > right_true

compare_predict2 = left_predict <= right_predict
compare_true2 = left_true <= right_true


final_predict = left_predict * compare_predict1 + right_predict * compare_predict2
final_true = left_true * compare_true1 + right_true * compare_true2

# q = quadratic_weighted_kappa(summary_true, summary_predict)
q = quadratic_weighted_kappa(final_true, final_predict)
print(q)



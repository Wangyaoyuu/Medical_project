######### Load Pacakges #######
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import numpy as np
import random
import os
import time
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from load_data import *
from load_model import *
from collections import Counter

import torch as pt
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F, SmoothL1Loss
from kappa import *




# %% define functions

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    pt.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# %% load data
seed = 0
reset_random_seeds(seed)
#240 0.69
image_size = 512
trainS, labelTr, testS, labelTs = load_data(image_size)
no_class = len(np.unique(labelTr))
# labelsCat = to_categorical(labelTr)

print('#[Train]shape: ', trainS.shape, labelTr.shape)
print('#[Test]shape: ', testS.shape, labelTs.shape)

# %% set paths and parameters
dataset = 'Fundus'
model_name = 'Efficient-not'  # ['ViT','ResNet50', 'ResNet50_sl1', 'Efficient', 'Efficient-onehot', 'Efficient-pretrain', 'Efficient-not']


model = load_model(model_name)

batchsize = 16
base_output_path = './output/'
weight_path = base_output_path + dataset + '/' + model_name
fileEnd = '.h5'

if not os.path.exists(weight_path):
    os.makedirs(weight_path)

srcfn = weight_path + '/model'

labelsize = 5


trainset = trainSet(trainS, labelTr, augment=True)
trainloader = DataLoader(trainset, batch_size=batchsize,
                         num_workers=2, prefetch_factor=batchsize, shuffle=True)
testset = trainSet(testS, labelTs, augment=False)
testloader = DataLoader(testset, batch_size=batchsize,
                         num_workers=2, prefetch_factor=batchsize, shuffle=False)
epochsize = (len(trainset) + batchsize - 1) // batchsize

lr_init, lr_exp = 5e-4, 1e-5  # note
sched_chk, sched_cycle = 4, 32
epochlast = sched_cycle * 32
print('#scheduler:', sched_cycle, epochlast)
optimizer, sched_lr = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9), lr_exp * 2
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, 2)
best = 0
try:
    state = pt.load(srcfn)
    model.load_state_dict(state['model'])
    batchidx = state['epoch'] * epochsize
    best = state['best']
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    print('#model load:', srcfn)
    print('#best:', best)
except Exception as e:
    print('#model transfer:', srcfn, e)
best = 0
print(best)
SL1 = SmoothL1Loss()

print('#training model ...')
summary = []
tepoch = tcheck = time.perf_counter()
batchidx = 0
epoches = 100
for i in range(epoches):
    summary_true = []
    summary_predict = []
    for batchtrain, batchvalid in zip(trainloader, testloader):
        # schedule
        with pt.no_grad():
            if batchidx % epochsize == 0 and batchidx >= epochsize * 2:
                epoch = batchidx // epochsize
                if epoch % sched_chk == 0:
                    scheduler.base_lrs = [max(lr / 2, sched_lr) for lr in scheduler.base_lrs]
                    if epoch >= sched_cycle:
                        scheduler.step(epoch % sched_cycle + sched_cycle - 1)
                    else:
                        scheduler.step(epoch - 1)
                    sched_chk = min(sched_chk * 2, sched_cycle)
                else:
                    if epoch >= sched_cycle:
                        scheduler.step(epoch % sched_cycle + sched_cycle - 1)
                    else:
                        scheduler.step(epoch - 1)

        # train
        model.train()
        optimizer.zero_grad()
        x = batchtrain['data'].clone().detach().cuda()
        y = batchtrain['label'].clone().detach().cuda()
        yy = model(x)
        yy = pt.squeeze(yy, -1)
        # lossy = lossFocal(yy, y, 0, labelsize)

        lossy = SL1(y, yy)
        loss = lossy
        loss.backward()
        # nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()
        model.eval()

        with pt.no_grad():
            batchidx += 1
            if batchidx > epochsize * epochlast: break
            # valid
            # if batchidx % 8 == 0:
            x = batchvalid['data'].clone().detach().cuda()
            yy = model(x)
            yy = pt.squeeze(yy, -1)
            yy = yy.cpu()
            loss = SL1(yy, batchvalid['label'])
            # accy = pt.mean((pt.argmax(yy, dim=-1) == batchvalid['label']).type_as(x)).item() * 100
            # t = pt.tensor([lossy.item(), accy])
            # accy = pt.mean((pt.argmax(yy, dim=-1) == batchvalid['label']).type_as(x)).item() * 100
            t = pt.tensor([loss.item(), lossy.item()])
            summary.append(t.numpy())

            y_true = batchvalid['label']
            # y_predict = pt.argmax(yy, dim=-1)
            y_predict = yy

            summary_true.extend(y_true)
            summary_predict.extend(y_predict)

            epoch = batchidx / epochsize
            # msg = np.mean(np.array(summary), axis=0)
            # print('#check[%.3f]: %.3f %.1f%%' % (epoch, *msg))
            msg = np.mean(np.array(summary), axis=0)
            print('#check[%.3f]: %.3f %.3f' % (epoch, *msg))

            tcurr, lr = time.perf_counter(), optimizer.param_groups[0]['lr']
            if batchidx % epochsize == 0:

                epoch = batchidx // epochsize
                tdiff = (tcurr - tepoch) / 60
                msg, summary = np.mean(np.array(summary), axis=0), []
                print('#epoch[%.3f]: %.3f %.3f %.1e %.1fm' % (epoch, *msg, lr, tdiff))

    q = quadratic_weighted_kappa(summary_true, summary_predict)
    if q > best:
        best = q
        pt.save({'epoch': i, 'best': best,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict()}, weight_path + '/model')
        # print('#epoch[%.3f]: %.3f %.1f%% %.1e %.1fm *' % (epoch, *msg, lr, tdiff))
        print('#epoch[%.3f]: %.3f *' % (epoch, q))
    else:
        # print('#epoch[%.3f]: %.3f %.1f%% %.1e %.1fm' % (epoch, *msg, lr, tdiff))
        print('#epoch[%.3f]: %.3f' % (epoch, q))
    print()
    assert (not np.isnan(msg[0]))  # debug
    tepoch = tcheck = tcurr
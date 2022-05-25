# Medicial_project
(代码写的有点丑，仅作抛砖引玉之用，会出现bug和很多错误) 

## Model
load_model.py

['ViT','ResNet50', 'ResNet50_sl1', 'Efficient', 'Efficient-onehot', 'Efficient-fine', 'Efficient-not']

ResNet50 和 Efficient-onehot使用focal loss训练，其余都是使用 SmoothL1 loss。所以这两个模型需要用train_test_onehot.py来单独训练

其余网络使用train_test_torch.py来训练

Efficient-not没有使用预训练的参数，除了Efficient-not以外的网络默认使用在ImageNet上预训练的权重参数。

### image_size
'ViT','ResNet50', 'ResNet50_sl1'使用256 $\times$ 256

'Efficient', 'Efficient-onehot', 'Efficient-fine', 'Efficient-not'使用512 $\times$ 512

## Data
load_data.py包括函数

load_data(image_size) 读取基本的数据,image_size确认读取image_size的大小。（不同image_size的数据应该提前处理好）

load_data_pretrain(image_size) 读取额外的数据，预训练使用 需要额外下载10个G的数据，ps:额外的预训练没什么作用。
APTOS. (2019) Available: https://www.kaggle.com/c/aptos2019-blindnessdetection

trainSet和testSet作为训练过程中数据的组合输入

### ps:数据有问题，使用data_pre.py读入的时候要手动修改一些数据的问题。

## Pretrain
pretrain.py

fine_tune.py

先运行pretrain.py得到pretrain模型，使用额外数据训练，使用全部基本数据测试。再运行fine_tune.py微调。

## Train
基本上就是dataload之后直接训练，每个epoch会测试一下结果。学习率可以再仔细调整。

## Kappa
test_kappa.py可根据预训练模型和数据得到对应的kappa分数，选取两眼中更大的标签作为预测和结果。
但是这一步应该是***写错了***，因为在数据处理的时候左右眼好像没有根据奇数和偶数放在一起，仅作为参考。




from ResNet50 import *
from VIT import ViT
from efficientnet_pytorch import EfficientNet

def load_model(model_name):
    if model_name == 'ViT':
        model = ViT().cuda()

    elif model_name == 'ResNet50':
        model = ResNet50(num_classes=5).cuda()

    elif model_name == 'ResNet50_sl1':
        model = ResNet50_sl1().cuda()

    elif model_name == 'Efficient':
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=1).cuda()

    elif model_name == 'Efficient-onehot':
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=5).cuda()

    elif model_name == 'Efficient-pretrain':
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=1).cuda()

    elif model_name == 'Efficient-fine':
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=1).cuda()

    elif model_name == 'Efficient-not':
        model = EfficientNet.from_name('efficientnet-b1', num_classes=1).cuda()
    else:
        model = EfficientNet.from_name('efficientnet-b1', num_classes=1).cuda()

    return model


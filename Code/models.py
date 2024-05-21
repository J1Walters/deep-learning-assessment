import torch.nn as nn
from torchvision.models import efficientnet_b0, mnasnet1_0, mobilenet_v3_large, resnext101_64x4d

def EfficientNet():
    model = efficientnet_b0(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=7, bias=True)
    return model

def MNASNet():
    model = mnasnet1_0(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=7, bias=True)
    return model

def MobileNetV3():
    model = mobilenet_v3_large(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[3] = nn.Linear(in_features=model.classifier[3].in_features, out_features=7, bias=True)
    return model

def ResNeXt():
    model = resnext101_64x4d(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=7, bias=True)
    return model
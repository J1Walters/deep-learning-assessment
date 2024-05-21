import torch
from torch import nn

def WeightedCELoss():
    if torch.cuda.is_available:
        loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1/1113, 1/6705, 1/514, 1/327, 1/1099, 1/115, 1/142]).to('cuda'))
    else:
        loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1/1113, 1/6705, 1/514, 1/327, 1/1099, 1/115, 1/142]))
    return loss_func
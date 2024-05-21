from torch.optim.lr_scheduler import CosineAnnealingLR

def CosineScheduler(optimiser, T_max=10, eta_min=0.0001):
    return CosineAnnealingLR(optimiser, T_max=T_max, eta_min=eta_min)
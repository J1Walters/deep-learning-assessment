import torch
import copy
from models import EfficientNet, MNASNet, MobileNetV3, ResNeXt

# Set path
model_path = '/home/jake/University/6G7V0024_Deep_Learning/Project/Data/models/'

# Load models to test
# EfficientNet
def eff_trained():
    eff_untrans = copy.deepcopy(EfficientNet())
    eff_aug = copy.deepcopy(EfficientNet())
    if torch.cuda.is_available():
        eff_untrans.load_state_dict(torch.load(model_path + 'weightloss_eff_freeze_default_adam.pth'))
        eff_aug.load_state_dict(torch.load(model_path + 'weightloss_eff_freeze_augmenttest_adam.pth'))
    else:
        eff_untrans.load_state_dict(torch.load(model_path + 'weightloss_eff_freeze_default_adam.pth', map_location='cpu'))
        eff_aug.load_state_dict(torch.load(model_path + 'weightloss_eff_freeze_augmenttest_adam.pth', map_location='cpu'))       
    return eff_untrans, eff_aug

# MNASNet
def mnas_trained():
    mnas_untrans = copy.deepcopy(MNASNet())
    mnas_aug = copy.deepcopy(MNASNet())
    if torch.cuda.is_available():
        mnas_untrans.load_state_dict(torch.load(model_path + 'weightloss_mnas_freeze_default_adam.pth'))
        mnas_aug.load_state_dict(torch.load(model_path + 'weightloss_mnas_freeze_augmenttest_adam.pth'))
    else:
        mnas_untrans.load_state_dict(torch.load(model_path + 'weightloss_mnas_freeze_default_adam.pth', map_location='cpu'))
        mnas_aug.load_state_dict(torch.load(model_path + 'weightloss_mnas_freeze_augmenttest_adam.pth', map_location='cpu'))        
    return mnas_untrans, mnas_aug

# MobileNet
def mobile_trained():
    mobile_untrans = copy.deepcopy(MobileNetV3())
    mobile_aug = copy.deepcopy(MobileNetV3())
    if torch.cuda.is_available():
        mobile_untrans.load_state_dict(torch.load(model_path + 'weightloss_mobile_freeze_default_adam.pth'))
        mobile_aug.load_state_dict(torch.load(model_path + 'weightloss_mobile_freeze_augmenttest_adam.pth'))
    else:
        mobile_untrans.load_state_dict(torch.load(model_path + 'weightloss_mobile_freeze_default_adam.pth', map_location='cpu'))
        mobile_aug.load_state_dict(torch.load(model_path + 'weightloss_mobile_freeze_augmenttest_adam.pth', map_location='cpu'))        
    return mobile_untrans, mobile_aug

# ResNeXt
def resnext_trained():
    resnext_untrans = copy.deepcopy(ResNeXt())
    resnext_aug = copy.deepcopy(ResNeXt())
    if torch.cuda.is_available():
        resnext_untrans.load_state_dict(torch.load(model_path + 'weightloss_resnext_freeze_default_adam.pth'))
        resnext_aug.load_state_dict(torch.load(model_path + 'weightloss_resnext_freeze_augmenttest_adam.pth'))
    else:
        resnext_untrans.load_state_dict(torch.load(model_path + 'weightloss_resnext_freeze_default_adam.pth', map_location='cpu'))
        resnext_aug.load_state_dict(torch.load(model_path + 'weightloss_resnext_freeze_augmenttest_adam.pth', map_location='cpu'))
    return resnext_untrans, resnext_aug
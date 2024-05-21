import torch
from torchvision.transforms import v2

# Mean and std for normalisation
means = [194.7717 , 137.25015, 143.20988]
stds = [22.315918, 30.71689 , 34.64285]

# Crop and resize transform
def CropResizeNormal():
    crop_resize_normal = v2.Compose([
    v2.CenterCrop(size=450),
    v2.Resize((224,224)),
    v2.ToDtype(torch.float32, scale=False),
    v2.Normalize(mean=means, std=stds)
    ])
    return crop_resize_normal

# Most common transforms
def CommonTransforms():
    common_transforms = v2.Compose([
        v2.CenterCrop(size=450),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ColorJitter(brightness=0.25, contrast=0.25),
        v2.Resize((224,224)),
        v2.ToDtype(torch.float32, scale=False),
        v2.Normalize(mean=means, std=stds)
    ])
    return common_transforms
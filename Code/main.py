from torch.utils.data import DataLoader
from dataset import ISIC2018Task3Dataset
from transforms import CropResizeNormal, CommonTransforms
from models import EfficientNet, MNASNet, MobileNetV3, ResNeXt
from optimisers import MakeAdam
from scheduler import CosineScheduler
from loss import WeightedCELoss
from train import Train

def main():
    # Get data
    train_data = ISIC2018Task3Dataset(image_dir='/content/ISIC2018Task3/ISIC2018Task3/train', csv_path='/content/ISIC2018Task3/ISIC2018Task3/labels/train.csv', transform = CropResizeNormal())
    augment_train_data = ISIC2018Task3Dataset(image_dir='/content/ISIC2018Task3/ISIC2018Task3/train', csv_path='/content/ISIC2018Task3/ISIC2018Task3/labels/train.csv', transform = CommonTransforms())
    val_data = ISIC2018Task3Dataset(image_dir='/content/ISIC2018Task3/ISIC2018Task3/val', csv_path='/content/ISIC2018Task3/ISIC2018Task3/labels/val.csv', transform=CropResizeNormal())
    test_data = ISIC2018Task3Dataset(image_dir='/content/ISIC2018Task3/ISIC2018Task3/test', csv_path='/content/ISIC2018Task3/ISIC2018Task3/labels/test.csv', transform=CropResizeNormal())

    # Dataloaders
    # Parameters
    batch_size = 32

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    augment_train_loader = DataLoader(augment_train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Get models
    efficientnet_untrans = EfficientNet()
    efficientnet_aug = EfficientNet()
    mnasnet_untrans = MNASNet()
    mnasnet_aug = MNASNet()
    mobilenet_untrans = MobileNetV3()
    mobilenet_aug = MobileNetV3()
    resnext_untrans = ResNeXt()
    resnext_aug = ResNeXt()

    # Make optimisers
    eff_adam = MakeAdam(EfficientNet().parameters())
    mnas_adam = MakeAdam(MNASNet().parameters())
    mobile_adam = MakeAdam(MobileNetV3().parameters())
    resnext_adam = MakeAdam(ResNeXt().parameters())

    # Make schedulers
    eff_adam_scheduler = CosineScheduler(eff_adam)
    mnas_adam_scheduler = CosineScheduler(mnas_adam)
    mobile_adam_scheduler = CosineScheduler(mobile_adam)
    resnext_adam_scheduler = CosineScheduler(resnext_adam)

    # Define loss function
    loss_func = WeightedCELoss()

    # Train models
    Train(efficientnet_untrans, 400, train_loader, val_loader, loss_func, eff_adam, scheduler=eff_adam_scheduler, early_stopping=20, metrics_path='../Data/metrics/weightloss_eff_freeze_default_adam_metrics.json', save_path='../Data/models/weightloss_eff_freeze_default_adam.pth')
    Train(efficientnet_aug, 400, augment_train_loader, val_loader, loss_func, eff_adam, scheduler=eff_adam_scheduler, early_stopping=20, metrics_path='../Data/metrics/weightloss_eff_freeze_augment_adam_metrics.json', save_path='../Data/models/weightloss_eff_freeze_augment_adam.pth')

if __name__ == '__main__':
    main()
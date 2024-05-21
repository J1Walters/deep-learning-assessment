import torch
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm
from dataset import ISIC2018Task3Dataset
from transforms import CropResizeNormal
from trained_models import eff_trained, mnas_trained, mobile_trained, resnext_trained

def get_inference_time(model, test_data):
    # Check if can run on cuda
    if torch.cuda.is_available():
        device = 'cuda'
        model.cuda()
    else:
        device = 'cpu'
    # Tell user what device running on
    print(f'Running on {device}')
    # Get start time
    start_time = time()
    # Put model into eval mode
    model.eval()
    for image, label in tqdm(test_data):
        # Move to device
        image, label = image.to(device), label.to(device)
        # Don't calculate gradient
        with torch.no_grad():
            # Make predictions
            model(image)
    # Get end time
    end_time = time()
    # Calculate time taken
    time_taken = end_time - start_time
    print(f'Time Taken: {time_taken}')

def main():
    # Get test data
    test = ISIC2018Task3Dataset(image_dir='/home/jake/University/6G7V0024_Deep_Learning/Project/Data/ISIC2018Task3/test/', csv_path='/home/jake/University/6G7V0024_Deep_Learning/Project/Data/ISIC2018Task3/labels/test.csv', transform=CropResizeNormal())
    test_loader = DataLoader(test, batch_size=32, shuffle=False, pin_memory=True)

    # Get models
    eff_untrans, eff_aug = eff_trained()
    mnas_untrans, mnas_aug = mnas_trained()
    mobile_untrans, mobile_aug = mobile_trained()
    resnext_untrans, resnext_aug = resnext_trained()

    # Run inference for each model
    print('=== EfficientNet ===')
    get_inference_time(eff_untrans, test_loader)
    get_inference_time(eff_aug, test_loader)
    print('=== MNASNet ===')
    get_inference_time(mnas_untrans, test_loader)
    get_inference_time(mnas_aug, test_loader)
    print('=== MobileNetV3 ===')
    get_inference_time(mobile_untrans, test_loader)
    get_inference_time(mobile_aug, test_loader)
    print('=== ResNeXt ===')
    get_inference_time(resnext_untrans, test_loader)
    get_inference_time(resnext_aug, test_loader)

if __name__ == '__main__':
    main()
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class ISIC2018Task3Dataset(Dataset):
  """ Custom class for ISIC 2018 Task 3 dataset. """

  def __init__(self, image_dir, csv_path, transform=None):
    """ Initialise dataset class.

    Args:
      image_dir (str): Path to image directory.
      csv_path (str): Path to csv file containing image name with class labels.
      transform (callable, optional): Optional transformations to apply on samples.
    """
    self.image_dir = image_dir
    self.csv_path = csv_path
    self.transform = transform
    df = pd.read_csv(csv_path, header=0)
    # Get class labels as label encoded not one-hot encoded
    df['class'] = df[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values.argmax(1)
    self.image_names = df['image'].values
    self.image_labels = df['class'].values

  def __len__(self):
    """ Returns number of samples in dataset. """
    return len(self.image_names)

  def __getitem__(self, idx):
    """ Returns dictionary containing the image as key and class label as value. """
    image_path = os.path.join(self.image_dir, self.image_names[idx] + '.jpg')
    image = read_image(image_path)
    label = self.image_labels[idx]

    if self.transform is not None:
      image = self.transform(image)

    return image, label
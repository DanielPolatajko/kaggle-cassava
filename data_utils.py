# Utility class to provide the Kaggle cassava dataset for Pytorch

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import numpy as np
import torch.nn as nn

def construct_datasets(data_path, batch_size, train_size=0.7, transform=None, shuffle=True):
    """
    Utility function to create a dataloader for Pytorch with the images, split into a training and validation set
    :param data_path: Path to the dataset
    :param train_size: The size of the split for training and validation sets
    :param batch_size: Batch size for model training
    :param transform: Any transforms or augmentations to apply to the images in dataset
    :param shuffle: Whether to shuffle the dataset at each epoch
    :return:
    """

    # apply transforms if necessary
    if transforms is not None:
        image_data = datasets.ImageFolder(data_path, transform=transform)
    else:
        image_data = datasets.ImageFolder(data_path)

    # split into train and validation
    mask = np.random.choice(len(image_data), int(train_size*len(image_data)), replace=False)
    train_data = Subset(image_data, mask)
    val_data = Subset(image_data, [x for x in range(len(image_data)) if x not in mask])

    return DataLoader(train_data, batch_size, shuffle), DataLoader(val_data, batch_size, shuffle)

def reshape_model(model, num_classes):
    """
    Function to reshape final layer of model
    """

    # features passed to final layer
    features = model.fc.in_features

    # reshape final layer
    model.fc = nn.Linear(features, num_classes)

    return model

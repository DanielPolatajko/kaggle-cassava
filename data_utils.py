# Utility class to provide the Kaggle cassava dataset for Pytorch

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import numpy as np
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from typing import Any, Tuple


class AlbumentationsImageFolder(datasets.ImageFolder):
    """
    Abstracts ImageFolder class to be compatible with albumentations transforms, basically same as DatasetFolder code
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(image=np.array(sample))['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

class DatasetConstructor:
    """
    Class to construct datasets for use with cross-validation
    """

    def __init__(self, data_path, batch_size, k=5, transform=None, shuffle=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform

        if transform is not None:
            self.image_data = AlbumentationsImageFolder(data_path)
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((384, 384)),
                 transforms.Normalize((0.4342, 0.4967, 0.3154),(0.2300, 0.2319, 0.2186))
                 ]
            )

            self.image_data = datasets.ImageFolder(data_path, transform=transform)

        self.folds = StratifiedKFold(n_splits=k)

        self.labels = np.array(self.image_data.targets)

        self.k = k

        self.splits = []

        for train, val in self.folds.split(np.zeros(len(self.labels)), self.labels):
            self.splits.append([train,val])

    def get_split(self, iteration):
        train_idx, val_idx = self.splits[iteration]

        # split into train and validation
        train_data = Subset(self.image_data, train_idx)
        val_data = Subset(self.image_data, val_idx)

        if self.transform is not None:
            train_data.dataset.transform = self.transform['train_transform']
            val_data.dataset.transform = self.transform['eval_transform']

        return DataLoader(train_data, self.batch_size, self.shuffle), DataLoader(val_data, self.batch_size, self.shuffle)




def reshape_model(model, num_classes):
    """
    Function to reshape final layer of model
    """

    # features passed to final layer
    features = model.fc.in_features

    # reshape final layer
    model.fc = nn.Linear(features, num_classes)

    return model

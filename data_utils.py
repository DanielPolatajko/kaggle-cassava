# Utility classes to provide the Kaggle cassava dataset for Pytorch

from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, models, transforms
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from typing import Any, Tuple

import numpy as np
from PIL import Image

import torch

from collections import OrderedDict, Counter

DISEASE_NAMES = {'0': 'Cassava Bacterial Blight (CBB)',
 '1': 'Cassava Brown Streak Disease (CBSD)',
 '2': 'Cassava Green Mottle (CGM)',
 '3': 'Cassava Mosaic Disease (CMD)',
 '4': 'Healthy'}


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

        # set some useful variables
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.k = k

        # handle data augmentations
        if transform is not None:
            # if transforms are given, apply them
            self.image_data = AlbumentationsImageFolder(data_path)
        else:
            # if not, add some standard ones
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((384, 384)),
                 transforms.Normalize((0.4342, 0.4967, 0.3154),(0.2300, 0.2319, 0.2186))
                 ]
            )

            self.image_data = datasets.ImageFolder(data_path, transform=transform)

        # create masks for cross validation folds
        self.folds = StratifiedKFold(n_splits=k)

        # save the labels for ease
        self.labels = np.array(self.image_data.targets)


        # get the splits out of the generator and store them, in case we want to save them
        self.splits = []
        for train, val in self.folds.split(np.zeros(len(self.labels)), self.labels):
            self.splits.append([train,val])

    def get_split(self, iteration):
        """
        Method to split the Pytorch dataset based on a given cross validation split
        and add the datasets to a data loader for training
        :param iteration: which split to use (handy for GPU's with limited runtime)
        :return:
        """

        # get the masks for training validation split
        train_idx, val_idx = self.splits[iteration]

        # split into train and validation
        train_data = Subset(self.image_data, train_idx)
        val_data = Subset(self.image_data, val_idx)

        # a = dict(Counter(self.labels[train_idx]))
        #
        # class_sample_count = []
        #
        # for i in range(5):
        #     class_sample_count.append(a[i])
        #
        # print(class_sample_count)
        #
        # weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
        # samples_weights = weights[self.labels[train_idx]]
        #
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(
        #     weights=samples_weights,
        #     num_samples=len(samples_weights),
        #     replacement=True)

        # apply different transforms sets to the split dataset
        if self.transform is not None:
            train_data.dataset.transform = self.transform['train_transform']
            val_data.dataset.transform = self.transform['eval_transform']

        # return the data loader objects
        return DataLoader(train_data, batch_size=self.batch_size, shuffle=True), DataLoader(
            val_data, self.batch_size)




def reshape_model(model, num_classes, dropout=True):
    """
    Function to reshape final layer of model
    """

    # features passed to final layer
    features = model.fc.in_features

    # reshape final layer, adding dropout if we want
    if dropout:
        model.fc = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(0.5)),
            ('linear', nn.Linear(in_features=features, out_features=num_classes, bias=True))]
        ))

    else:
        model.fc = nn.Linear(features, num_classes)

    return model


class KaggleTrainDataset(Dataset):
    def __init__(self, df, data_path, transform=None):
        self.df = df
        self.data_path = data_path
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{self.data_path}/{file_name}'
        image = Image.open(file_path)
        if self.transform:
            augmented = self.transform(image=np.array(image))
            image = augmented['image']

        label = self.labels[idx]
        return image, label


class KaggleDatasetConstructor:
    """
    Class to construct datasets for use with cross-validation
    """

    def __init__(self, data_path, df, batch_size, transform, k=5, shuffle=True, splits=None):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.df = df

        self.image_data = KaggleTrainDataset(df, data_path, transform)

        self.folds = StratifiedKFold(n_splits=k)

        self.labels = np.array(self.image_data.labels)

        self.k = k

        if splits is None:
            self.splits = []

            for train, val in self.folds.split(np.zeros(len(self.labels)), self.labels):
                self.splits.append([train, val])

        else:
            self.splits = splits

    def get_split(self, iteration):
        train_idx, val_idx = self.splits[iteration]

        # split into train and validation
        train_data = Subset(self.image_data, train_idx)
        val_data = Subset(self.image_data, val_idx)

        sliced_df = self.df.ix[train_idx]

        a = sliced_df['disease'].value_counts()

        class_sample_count = []

        for i in range(5):
            class_sample_count.append(sliced_df['disease'].value_counts()[DISEASE_NAMES[str(i)]])


        weights = 1 / torch.Tensor(class_sample_count)
        weights = weights.double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, self.batch_size)

        if self.transform is not None:
            train_data.dataset.transform = self.transform['train_transform']
            val_data.dataset.transform = self.transform['eval_transform']

        return DataLoader(train_data, batch_size=self.batch_size, sampler=sampler, shuffle=self.shuffle), DataLoader(val_data, self.batch_size,
                                                                                 self.shuffle)

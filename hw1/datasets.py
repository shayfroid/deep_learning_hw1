import numpy as np

import torch
from torch.utils.data import Dataset


class RandomImageDataset(Dataset):
    """
    A dataset returning random noise images of specified dimensions
    """

    def __init__(self, num_samples, num_classes, C, W, H):
        """
        :param num_samples: Number of samples (labeled images in the dataset)
        :param num_classes: Number of classes (labels)
        :param C: Number of channels per image
        :param W: Image width
        :param H: Image height
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.image_dim = (C, W, H)

    def __getitem__(self, index):

        if index < 0 or index >= self.num_samples:
            raise IndexError(f'Out of bound index: {index}')

        old_seed = torch.initial_seed()
        torch.manual_seed(index)

        random_image = torch.randint(0, 256, self.image_dim, dtype=torch.uint8)
        random_label = torch.randint(0, self.num_classes, (1,), dtype=torch.long).item()

        torch.manual_seed(old_seed)

        return random_image, random_label

    def __len__(self):
        return self.num_samples


class SubsetDataset(Dataset):
    """
    A dataset that wraps another dataset, returning a subset from it.
    """
    def __init__(self, source_dataset: Dataset, subset_len, offset=0):
        """
        Create a SubsetDataset from another dataset.
        :param source_dataset: The dataset to take samples from.
        :param subset_len: The total number of sample in the subset.
        :param offset: The offset index to start taking samples from.
        """
        if offset + subset_len > len(source_dataset):
            raise ValueError("Not enough samples in source dataset")

        self.source_dataset = source_dataset
        self.subset_len = subset_len
        self.offset = offset

    def __getitem__(self, index):
        # ====== YOUR CODE: ======
        if index < 0 or index >= self.subset_len:
            raise IndexError(f'Invalid index: {index}')

        return self.source_dataset[self.offset+index]
        # ========================

    def __len__(self):
        # ====== YOUR CODE: ======
        return self.subset_len
        # ========================


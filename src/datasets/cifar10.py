from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting

import torch
import torchvision.transforms as transforms
import random
import numpy as np
from sklearn.model_selection import train_test_split

class CIFAR10_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 5, known_outlier_class: int = 3, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        # CIFAR-10 preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyCIFAR10(root=self.root, train=True, transform=transform, target_transform=target_transform,
                              download=True)
        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(np.array(train_set.targets), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = MyCIFAR10(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                  download=True)

class AutoAttck_CIFAR10_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 5, known_outlier_class: int = 3, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, 
                 target_class = 0, advserial_data_path = None):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        # CIFAR-10 preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyCIFAR10(root=self.root, train=True, transform=transform, target_transform=target_transform,
                                  download=True)
        # First prepare original data and label
        original_dataset = train_set.data.copy()
        original_labels = np.array(train_set.targets, copy=True)

        # Then load the adversarial data
        test_loader = torch.load(advserial_data_path)['adv_complete']
        tensors = test_loader[0].permute(0, 2, 3, 1)
        tensors = (tensors * 255).numpy().astype(np.uint8)
        targets = np.array(test_loader[1])
        # get the tensors with target class ONLY
        tensors_with_target_class = tensors[np.where(targets == target_class)]
        num_of_adversarial = min(100, len(tensors_with_target_class))
       
        # Mix the original data and the adversarial data. there should be num_of_adversarial of adversarial data
        mixed_data = tensors_with_target_class[:num_of_adversarial]
        original_target_data = original_dataset[np.where(original_labels == target_class)][:num_of_adversarial]
        mixed_data = np.concatenate((mixed_data, original_target_data))
        # so labels we have are num_of_adversarial * 2 (adverserial and original). labeled as 1 and 0 respectively
        mixed_labels = [1 for i in range(num_of_adversarial)]
        mixed_labels.extend([0 for i in range(num_of_adversarial)])
        X_train, X_test, y_train, y_test = train_test_split(mixed_data, mixed_labels, test_size=0.2, random_state=123)
        # replace train set data with target class
        # training and testing split are 0.8 : 0.2
        train_set.data = X_train
        train_set.targets = y_train

        train_set.semi_targets = torch.zeros(len(train_set.targets), dtype=torch.int64)
        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(np.array(train_set.targets), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        semi_targets = semi_targets[:len(train_set.semi_targets)]
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = MyCIFAR10(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                  download=True)
        self.test_set.data = X_test
        self.test_set.targets = y_test
        self.test_set.semi_targets = torch.zeros(len(self.test_set.targets), dtype=torch.int64)

class AutoAttck_CIFAR100_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 5, known_outlier_class: int = 3, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, 
                 target_class = 0, advserial_data_path = None):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        # CIFAR-10 preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyCIFAR100(root=self.root, train=True, transform=transform, target_transform=target_transform,
                                  download=True)
        # First prepare original data and label
        original_dataset = train_set.data.copy()
        original_labels = np.array(train_set.targets, copy=True)

        # Then load the adversarial data
        test_loader = torch.load(advserial_data_path)['adv_complete']
        tensors = test_loader[0].permute(0, 2, 3, 1)
        tensors = (tensors * 255).numpy().astype(np.uint8)
        targets = np.array(test_loader[1])
        # get the tensors with target class ONLY
        tensors_with_target_class = tensors[np.where(targets >= target_class) and np.where(targets <= target_class + 10)]
        num_of_adversarial = min(100, len(tensors_with_target_class))
       
        # Mix the original data and the adversarial data. there should be num_of_adversarial of adversarial data
        mixed_data = tensors_with_target_class[:num_of_adversarial]
        original_target_data = original_dataset[np.where(targets >= target_class) and np.where(targets <= target_class + 10)][:num_of_adversarial]
        mixed_data = np.concatenate((mixed_data, original_target_data))
        # so labels we have are num_of_adversarial * 2 (adverserial and original). labeled as 1 and 0 respectively
        mixed_labels = [1 for i in range(num_of_adversarial)]
        mixed_labels.extend([0 for i in range(num_of_adversarial)])
        X_train, X_test, y_train, y_test = train_test_split(mixed_data, mixed_labels, test_size=0.2, random_state=123)
        # replace train set data with target class
        # training and testing split are 0.8 : 0.2
        train_set.data = X_train
        train_set.targets = y_train

        train_set.semi_targets = torch.zeros(len(train_set.targets), dtype=torch.int64)
        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(np.array(train_set.targets), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        semi_targets = semi_targets[:len(train_set.semi_targets)]
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = MyCIFAR100(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                  download=True)
        self.test_set.data = X_test
        self.test_set.targets = y_test
        self.test_set.semi_targets = torch.zeros(len(self.test_set.targets), dtype=torch.int64)

class MyCIFAR10(CIFAR10):
    """
    Torchvision CIFAR10 class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyCIFAR10, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros(len(self.targets), dtype=torch.int64)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], self.targets[index], int(self.semi_targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target, index
    
class MyCIFAR100(CIFAR100):
    """
    Torchvision CIFAR10 class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyCIFAR100, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros(len(self.targets), dtype=torch.int64)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], self.targets[index], int(self.semi_targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        # try:
        #     img = Image.fromarray(img)
        # except Exception:
        #     pass
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target, index

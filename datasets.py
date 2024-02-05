from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset


# list of all datasets
DATASETS = ["imagenet", "cifar10", "cifar100", "cifar100-new-split", "mnist", "fashion_mnist", "tiny_imagenet"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "cifar100":
        return _cifar100(split)
    elif dataset == "cifar100-new-split":
    	return _cifar100_new_split(split)
    elif dataset == "mnist":
        return _mnist10(split)
    elif dataset == "tiny_imagenet":
        return _tinyimagenet(split)
    elif dataset == "fashion_mnist":
        return _fashion_mnist10(split)
    

def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "cifar100":
        return 100
    elif dataset == "cifar100-new-split":
    	return 100
    elif dataset == "mnist":
        return 10
    elif dataset == "fashion_mnist":
        return 10

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STDDEV = (0.2023, 0.1994, 0.2010)

_MNIST_MEAN = (0.1307,)
_MNIST_STDDEV = (0.3081,)

_FASHION_MNIST_MEAN = (0.28604,)
_FASHION_MNIST_STDDEV = (0.35302,)

_TINY_MEAN = [0.480, 0.448, 0.398]
_TINY_STD = [0.277, 0.269, 0.282]

def _mnist10(split: str) -> Dataset:
    if split == "train":
        return datasets.MNIST("./dataset_cache", train=True, download=True, transform=transforms.Compose([
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)
        ]))
    
    elif split == "test":
        return datasets.MNIST("./dataset_cache", train=False, download=True, transform=transforms.Compose([
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)
        ]))
    

def _fashion_mnist10(split: str) -> Dataset:
    if split == "train":
        return datasets.FashionMNIST("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_FASHION_MNIST_MEAN, _FASHION_MNIST_STDDEV)
        ]))
    elif split == "test":
        return datasets.FashionMNIST("./dataset_cache", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_FASHION_MNIST_MEAN, _FASHION_MNIST_STDDEV)
        ]))

def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STDDEV)
            
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STDDEV)
        ]))
    
def _cifar100(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR100("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            
        ]))
    elif split == "test":
        return datasets.CIFAR100("./dataset_cache", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]))
class CIFAR100WithValSplit(datasets.CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = 'cifar-100-python'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    filename = 'cifar-100-python.tar.gz'
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '00b4df24681b7c4d4bc8f0f49f25beda'],
    ]
    val_list = [
        ['val', 'e9d24a4dd8ade6737b0abffab85f94eb']
    ]

    test_list = [
        ['test', '588fa8d951f0154158c16e9e46129698'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '1580defa312ed3539344f148645d707a',
    }


def _cifar100_new_split(split: str) -> Dataset:
    if split == "train":
        return CIFAR100WithValSplit("./cifar100-new-split", train=True, download=False, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            
        ]))
    elif split == "val":
        """
        This is certainly not a clean way to do this, but I wanted to stay consistent with how the datasets are created here.
        It's either something like this or change torchvision. I prefer the former.
        """
        return CIFAR100WithValSplit("./cifar100-new-split-val", train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ]))

    elif split == "test":
        return CIFAR100WithValSplit("./cifar100-new-split", train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]))


def _imagenet(split: str) -> Dataset:
    if split == "train":
        subdir = os.path.join("path/to/imagenet", "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STDDEV)
        ])
    elif split == "test":
        subdir = os.path.join("path/to/imagenet", "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STDDEV)
        ])
    return datasets.ImageFolder(subdir, transform)

def _tinyimagenet(split: str) -> Dataset:
    if split == "train":
        subdir = os.path.join("./tiny-imagenet-200", "train")
        transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_TINY_MEAN, _TINY_STD)
        ])
    elif split == "test":
        subdir = os.path.join("./tiny-imagenet-200", "val")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_TINY_MEAN, _TINY_STD)
        ])
    return datasets.ImageFolder(subdir, transform)

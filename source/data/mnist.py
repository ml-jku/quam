import os
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset
from ..constants import MNIST_PATH

transform = transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def get_mnist(seed=42):

    full_train = datasets.MNIST(MNIST_PATH, train=True, download=True, transform=transform)
    test = datasets.MNIST(MNIST_PATH, train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed=seed)
    val_inds = rng.choice(np.arange(len(full_train)), size=len(full_train) // 6, replace=False)
    train_inds = np.delete(np.arange(len(full_train)), val_inds)

    train = Subset(full_train, indices=train_inds)
    val = Subset(full_train, indices=val_inds)

    return train, test, val


def get_fmnist():
    return \
        datasets.FashionMNIST(MNIST_PATH, train=True, download=True, transform=transform), \
            datasets.FashionMNIST(MNIST_PATH, train=False, download=True, transform=transform), None


def get_emnist():
    # emnist is transposed compared to mnist
    emnist_transform = transforms.Compose([
        transform,
        transforms.Lambda(lambda x: torch.transpose(x, -2, -1))
    ])

    return \
        datasets.EMNIST(MNIST_PATH, split="letters", train=True, download=True, transform=emnist_transform), \
            datasets.EMNIST(MNIST_PATH, split="letters", train=False, download=True, transform=emnist_transform), None


def get_kmnist():
    return datasets.KMNIST(MNIST_PATH, train=True, download=True, transform=transform),\
        datasets.KMNIST(MNIST_PATH, train=False, download=True, transform=transform), None


def get_omni():
    omni_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomInvert(1),
        transform
    ])

    return None, datasets.Omniglot(MNIST_PATH, download=True, background=False, transform=omni_transform), None

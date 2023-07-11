import os
import shutil
import wget
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import TensorDataset, Subset
from ..constants import CIFAR_PATH

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                             std=[0.2023, 0.1994, 0.2010],)
    ])

train_transform = transforms.Compose([
    transforms.RandomCrop((32, 32), padding=4),
    transforms.RandomHorizontalFlip(),
    transform
])


def get_cifar10(seed:int=42):

    full_train = datasets.CIFAR10(CIFAR_PATH, train=True, download=True, transform=train_transform)
    test = datasets.CIFAR10(CIFAR_PATH, train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed=seed)
    val_inds = rng.choice(np.arange(len(full_train)), size=len(full_train) // 6, replace=False)
    train_inds = np.delete(np.arange(len(full_train)), val_inds)

    train = Subset(full_train, indices=train_inds)
    val = Subset(full_train, indices=val_inds)

    return train, test, val


def get_cifar100():
    return None, datasets.CIFAR100(CIFAR_PATH, train=False, download=True, transform=transform), None


def get_svhn():
    return datasets.SVHN(CIFAR_PATH, split="train", download=True, transform=transform), datasets.SVHN(CIFAR_PATH, split="test", download=True, transform=transform), None


def get_lsun():
    _download_lsun(CIFAR_PATH)
    return None, datasets.ImageFolder(root=os.path.join(CIFAR_PATH, "LSUN_resize"), transform=transform), None


def get_tim():
    _download_tiny_imagenet(CIFAR_PATH)

    # resize to be compatible with cifar10
    tim_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transform
    ])
    
    return None, datasets.ImageFolder(root=os.path.join(CIFAR_PATH, "tiny-imagenet-200", "test"), transform=tim_transform), None


corruptions = [
    # hyperparameter selection validation corruptions (https://arxiv.org/abs/1903.12261)
    "speckle_noise",
    "glass_blur",
    "spatter",
    "saturate",
    # test corruptions
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur"
]


def get_cifar10_c(corruption:int=0, severity:int=1):
    assert 0 <= severity <= 5, f"'severity' must be in [0, 5], but was {severity}."
    assert 0 <= corruption <= len(corruptions), f"'corruption' must be in [0, {len(corruptions)}], was {corruption}"
    
    if severity == 0:
        # severity 0 is original testset
        return datasets.CIFAR10(CIFAR_PATH, train=False, download=True, transform=transform)
    
    _download_cifar10_c(CIFAR_PATH)

    test_x = np.load(os.path.join(CIFAR_PATH, "CIFAR-10-C", "brightness.npy"))
    test_y = np.load(os.path.join(CIFAR_PATH, "CIFAR-10-C", "labels.npy"))

    test_x = test_x[10_000 * (severity - 1) : 10_000 * severity]

    transformed_test_x = list()
    for i in range(len(test_x)):
        transformed_test_x.append(transform(test_x[i]))
        if transform(test_x[i]).shape != (3, 32, 32):
            print(transform(test_x[i]).shape)

    test_y = test_y[10_000 * (severity - 1) : 10_000 * severity]

    return None, TensorDataset(torch.stack(transformed_test_x, dim=0), torch.Tensor(test_y)), None


def _download_cifar10_c(path:str):
    if not (os.path.exists(os.path.join(path, "CIFAR-10-C.tar")) or os.path.exists(os.path.join(path, "CIFAR-10-C"))):
        wget.download("https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1", os.path.join(path, "CIFAR-10-C.tar"))
    if not os.path.exists(os.path.join(path, "CIFAR-10-C")):
        print("unpacking...")
        shutil.unpack_archive(os.path.join(path, "CIFAR-10-C.tar"), os.path.join(path))
        print("unpacked")


def _download_lsun(path:str):
    if not (os.path.exists(os.path.join(path, "LSUN_resize.tar.gz")) or os.path.exists(os.path.join(path, "LSUN_resize"))):
        wget.download("https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz?dl=1", os.path.join(path, "LSUN_resize.tar.gz"))
    if not os.path.exists(os.path.join(path, "LSUN_resize")):
        print("unpacking...")
        shutil.unpack_archive(os.path.join(path, "LSUN_resize.tar.gz"), os.path.join(path))
        print("unpacked")


def _download_tiny_imagenet(path:str):
    if not (os.path.exists(os.path.join(path, "tiny-imagenet-200.zip")) or os.path.exists(os.path.join(path, "tiny-imagenet-200"))):
        wget.download("http://cs231n.stanford.edu/tiny-imagenet-200.zip", os.path.join(path, "tiny-imagenet-200.zip"))
    if not os.path.exists(os.path.join(path, "tiny-imagenet-200")):
        print("unpacking...")
        shutil.unpack_archive(os.path.join(path, "tiny-imagenet-200.zip"), os.path.join(path))
        print("unpacked")

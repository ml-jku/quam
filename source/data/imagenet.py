import os
import shutil
import wget
from tqdm import tqdm
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset, Dataset, DataLoader
from ..constants import IMAGENET_PATH, IMAGENET_AO_PATH, PROJECTED_IMAGENET_PATH
from ..networks.efficientnet import get_efficientnet

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])


class NpzDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        self.projections = data['projections']
        self.targets = data['targets']
    
    def __len__(self):
        return len(self.projections)

    def __getitem__(self, idx):
        return self.projections[idx], self.targets[idx]


def get_imagenet_projected(short=False, prefix='s_'):
    # set short to true to load the smaller one (saves memory)
    if short:
        return NpzDataset(os.path.join(os.path.abspath(PROJECTED_IMAGENET_PATH),prefix+"imagenet_train_subset100000.npz")), \
            NpzDataset(os.path.join(os.path.abspath(PROJECTED_IMAGENET_PATH), prefix+"imagenet_test.npz")), None
    else:
        return NpzDataset(os.path.join(os.path.abspath(PROJECTED_IMAGENET_PATH),prefix+"imagenet_train.npz")), \
            NpzDataset(os.path.join(os.path.abspath(PROJECTED_IMAGENET_PATH),prefix+"imagenet_test.npz")), None


def get_imagenet_o_projected(prefix='s_'):
    return None, NpzDataset(os.path.join(os.path.abspath(PROJECTED_IMAGENET_PATH),prefix+"imagenet-o_test.npz")), None


def get_imagenet_a_projected(prefix='s_'):
    return None, NpzDataset(os.path.join(os.path.abspath(PROJECTED_IMAGENET_PATH), prefix+"imagenet-a_test.npz")), None


def project_data(efficientnet_version: str, dataset_id: str, subset: str, use_ds_subset: float = None, seed=42):

    dataset_getters = {
        'imagenet': get_imagenet,
        'imagenet-o': get_imagenet_o,
        'imagenet-a': get_imagenet_a,
    }
    subset_index = {
    'train': 0,
    'test': 1,
    'val': 2,
    }

    ds_getter = dataset_getters[dataset_id]
    dss = ds_getter(use_projections=False, efficientnet_version=efficientnet_version)
    selected_ds = dss[subset_index[subset]]
    
    if use_ds_subset:
        rng = np.random.default_rng(seed=seed)
        subset_inds = rng.choice(np.arange(len(selected_ds)), size=use_ds_subset, replace=False)
        selected_ds = Subset(selected_ds, indices=subset_inds)

    dataloader = DataLoader(dataset=selected_ds,
                    batch_size=512,
                    num_workers=len(selected_ds) if len(selected_ds) < 32 else 32)

    device = f"cuda" if torch.cuda.is_available() else "cpu"
    network = get_efficientnet(version=efficientnet_version, use_projections=False, use_bfloat16=False)
    network.to(device)
    network.eval()

    projections, targets = list(), list()
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            out = network.project(x.to(device))
            projections.append(out.cpu().numpy().squeeze())
            targets.append(y.numpy())
        
        projections = np.concatenate(projections, axis=0)
        targets = np.concatenate(targets, axis=0)

        np.savez(os.path.join(PROJECTED_IMAGENET_PATH, f"{efficientnet_version}_{dataset_id}_{subset}{'_subset' + str(use_ds_subset) if use_ds_subset else ''}.npz"), projections=projections, targets=targets)


def get_imagenet(seed=42, use_projections=False, efficientnet_version="s", use_train_ds_subset=None):

    if use_projections:
        train_path = os.path.join(PROJECTED_IMAGENET_PATH, f"{efficientnet_version}_imagenet_train{'_subset' + str(use_train_ds_subset) if use_train_ds_subset else ''}.npz")
        if not os.path.exists(train_path):
            print(f"no projected ImageNet train {'subset'+ str(use_train_ds_subset) if use_train_ds_subset else 'dataset'} for EfficientNet '{efficientnet_version}' found on disk - preparing data ...")
            project_data(efficientnet_version=efficientnet_version, dataset_id="imagenet", subset="train", use_ds_subset=use_train_ds_subset, seed=seed)
        full_train = NpzDataset(train_path)

        test_path = os.path.join(PROJECTED_IMAGENET_PATH, f"{efficientnet_version}_imagenet_test.npz")
        if not os.path.exists(test_path):
            print(f"no projected ImageNet test dataset for EfficientNet '{efficientnet_version}' found on disk - preparing data ...")
            project_data(efficientnet_version=efficientnet_version, dataset_id="imagenet", subset="test", seed=seed)
        test = NpzDataset(test_path)

    else:
        full_train = datasets.ImageFolder(root=os.path.join(IMAGENET_PATH, "ImageNet1K", "train"), transform=train_transform)
        test = datasets.ImageFolder(root=os.path.join(IMAGENET_PATH, "ImageNet1K", "val"), transform=train_transform)

    rng = np.random.default_rng(seed=seed)
    # 50_000 is the size of the extra val set, we use as test set.
    val_inds = rng.choice(np.arange(len(full_train)), size=50_000, replace=False)
    train_inds = np.delete(np.arange(len(full_train)), val_inds)
    
    train = Subset(full_train, indices=train_inds)
    val = Subset(full_train, indices=val_inds)

    return train, val, test


def get_imagenet_a(use_projections=False, efficientnet_version="s"):
    if use_projections:
        test_path = os.path.join(PROJECTED_IMAGENET_PATH, f"{efficientnet_version}_imagenet-a_test.npz")
        if not os.path.exists(test_path):
            print(f"no projected ImageNet-A dataset for EfficientNet '{efficientnet_version}' found on disk - preparing data ...")
            project_data(efficientnet_version=efficientnet_version, dataset_id="imagenet-a", subset="test")
        dataset = NpzDataset(test_path)
    else:
        _download_imagenet_a(IMAGENET_AO_PATH)
        dataset = datasets.ImageFolder(root=os.path.join(IMAGENET_AO_PATH, "imagenet-a"), transform=transform)
    
    return None, dataset, None


def get_imagenet_o(use_projections=False, efficientnet_version="s"):
    if use_projections:
        test_path = os.path.join(PROJECTED_IMAGENET_PATH, f"{efficientnet_version}_imagenet-o_test.npz")
        if not os.path.exists(test_path):
            print(f"no projected ImageNet-O dataset for EfficientNet '{efficientnet_version}' found on disk - preparing data ...")
            project_data(efficientnet_version=efficientnet_version, dataset_id="imagenet-o", subset="test")
        dataset = NpzDataset(test_path)
    else:
        _download_imagenet_o(IMAGENET_AO_PATH)
        dataset = datasets.ImageFolder(root=os.path.join(IMAGENET_AO_PATH, "imagenet-o"), transform=transform)

    return None, dataset, None


def _download_imagenet_a(path:str):
    if not (os.path.exists(os.path.join(path, "imagenet-a.tar")) or os.path.exists(os.path.join(path, "imagenet-a"))):
        wget.download("https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar", os.path.join(path, "imagenet-a.tar"))
    if not os.path.exists(os.path.join(path, "imagenet-a")):
        print("unpacking...")
        shutil.unpack_archive(os.path.join(path, "imagenet-a.tar"), os.path.join(path))
        print("unpacked")


def _download_imagenet_o(path:str):
    if not (os.path.exists(os.path.join(path, "imagenet-o.tar")) or os.path.exists(os.path.join(path, "imagenet-o"))):
        wget.download("https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar", os.path.join(path, "imagenet-o.tar"))
    if not os.path.exists(os.path.join(path, "imagenet-o")):
        print("unpacking...")
        shutil.unpack_archive(os.path.join(path, "imagenet-o.tar"), os.path.join(path))
        print("unpacked")

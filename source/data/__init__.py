from .mnist import *
from .cifar import *
from .imagenet import *

from .wrappers import wrap_dataset_or_loader
from torch.utils.data import DataLoader as tDataLoader


# provide a reference for dataset getters
dataset_getters = {
    'mnist': get_mnist,
    'fmnist': get_fmnist,
    'emnist': get_emnist,
    'kmnist': get_kmnist,
    'omni': get_omni,
    'cifar-10': get_cifar10,
    'cifar-100': get_cifar100,
    'imagenet': get_imagenet,
    'imagenet-o': get_imagenet_o,
    'imagenet-a': get_imagenet_a,
    'svhn': get_svhn,
    'lsun': get_lsun,
    'tiny-imagenet': get_tim,
    'imagenet-projected': get_imagenet_projected,
    'imagenet-o-projected': get_imagenet_o_projected,
    'imagenet-a-projected': get_imagenet_a_projected,
}

subset_index = {
    'train': 0,
    'test': 1,
    'val': 2,
}


def get_dataset(
    id: str,
    subset: str,
    properties: dict = {},
):
    ds_getter = dataset_getters[id]
    dss = ds_getter(**properties)
    selected_ds = dss[subset_index[subset]]
    return selected_ds


def get_dataset_as_configured(
    id: str,
    subset: str,
    properties: dict = {},
    ds_wrappers: dict = [],
    dl_wrappers: dict = [],
    dl_properties: dict = {},
):
    ds = get_dataset(id, subset, properties)

    # apply all the wrappers from the registry
    ds = wrap_dataset_or_loader(ds, ds_wrappers)

    # create the dataloader
    dl = tDataLoader(ds, **dl_properties)
    dl = wrap_dataset_or_loader(dl, dl_wrappers)

    return ds, dl

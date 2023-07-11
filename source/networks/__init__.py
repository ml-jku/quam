from .lenet import get_lenet
from .resnets import get_cifar10_resnet20
from .efficientnet import get_efficientnet, get_efficientnet_ll
from typing import List

from torch.nn import ModuleList


def get_ensemble(configs: List, take_n: 1):
    # each item of the list contains a description of a network,
    ensemble = ModuleList([get_network_as_configured(**configs[i]) for i in range(take_n)])
    return ensemble


network_name_getter_index = {
    'ensemble': get_ensemble,
    'mnist_lenet': get_lenet,
    'cifar_10_resnet_20': get_cifar10_resnet20,
    'cifar_10_vgg': None,
    'efficient_net_ll': get_efficientnet_ll,
    'efficient_net': get_efficientnet,
    'text_bert': None,
}


def get_network_as_configured(
    net_id: str,
    net_properties: dict,
    device: str = "cuda:0",
):
    net_getter = network_name_getter_index[net_id]
    # net getter function may even train a network if needed (e.g. for a specific seed)
    return net_getter(**net_properties).to(device)

import torch


def get_cifar10_resnet20(checkpoint: str = None):
    if checkpoint is None:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    else:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False)
        model.load_state_dict(torch.load(checkpoint)['model'])
    return model

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader as tDataLoader

from typing import Callable, List, Union, Tuple


def xonly(batch):
    '''
    Extracts the input tensor from an arbitrary dataloader
    :param batch:
    :return:
    '''
    return batch if isinstance(batch, torch.Tensor) else batch['x'] if isinstance(batch, dict) else batch[0]


def yonly(batch):
    if isinstance(batch, torch.Tensor):
        return torch.Tensor([-1])
    elif isinstance(batch, dict):
        return get_whichever(batch, ['target', 'y', 'label'])
    else:
        return ensure_tensor(batch[-1], dtype=torch.long)


def get_whichever(src: dict, options: list):
    for o in options:
        if o in src:
            return src[o]
    return -1


def ensure_tensor(rand, dtype=torch.float):
    if isinstance(rand, torch.Tensor):
        return rand
    else:
        return torch.Tensor(rand, dtype=dtype)


def dict_to_cpu(d: dict):
    return {k: v.cpu() if isinstance(v, torch.Tensor) or isinstance(v, nn.Module) else v for k, v in d.items()}


def dict_detach(d: dict):
    return {k: v.detach() if isinstance(v, torch.Tensor) or isinstance(v, nn.Module) else v for k, v in d.items()}


def _forward_result(
    x,
    model: nn.Module,
    device_movements_async = False,
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).data.dtype
    x = x.to(device=device, dtype=dtype, non_blocking=device_movements_async)

    y_hat = model.forward(x)

    return y_hat


def _forward(
    x, y,
    model: nn.Module,
    loss: nn.Module,
    device_movements_async = False,
):
    y_hat = _forward_result(
        x,
        model,
        device_movements_async=device_movements_async
    )
    y = y.to(next(model.parameters()).device, non_blocking=device_movements_async)
    l = loss(y_hat, y)

    return l, y_hat


@torch.enable_grad()
def train_episode(
    model: nn.Module,
    opt: optim.Optimizer,
    loss: nn.Module,
    dl: tDataLoader,
    logger_callback: Callable = None
) -> List[torch.Tensor]:
    
    model.train()
    losses = []

    for x, y in dl:
        opt.zero_grad()

        l, _ = _forward(x, y, model, loss)

        l.backward()
        opt.step()

        l = l.detach().cpu()
        losses.append(l)

        if logger_callback:
            logger_callback(l)

    return losses


@torch.no_grad()
def validate(
    model: nn.Module,
    loss: nn.Module,
    dl: tDataLoader,
    logger_callback: Callable = None
) -> List[torch.Tensor]:
    
    model.eval()
    losses = []

    for x, y in dl:
        l, _ = _forward(x, y, model, loss)

        l = l.detach().cpu()
        losses.append(l.detach().cpu())

        if logger_callback:
            logger_callback(l)

    return losses

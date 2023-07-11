import numpy as np
import torch
from torch import nn
from torch import optim

from torch.optim import Optimizer as tOptimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
from torch.nn.functional import one_hot

from source.utils.training import _forward, xonly, dict_to_cpu, dict_detach, yonly
from ..networks.batched_modules import feed_batch_model_image_multiple, batch_that_net, ModelBatchedCELoss, mirror_member_net, feed_batch_model_vector_data

from torch.utils.data import DataLoader as tDataLoader

from typing import Callable, List, Dict, Union

from tqdm import tqdm

from ..constants import VERBOSITY_SETTING_SCORES, VERBOSITY_SETTING_SILENT, VERBOSITY_SETTING_PROGRESS, VERBOSITY_SETTING_INTERMEDIATES

import wandb


def configure_penopt_schedule(
    type: str,
    c0: float,
    eta: float,
    update_c_every: int,
):
    # eta means different things depending on the mode
    if type == 'exp':
        return lambda step: c0*eta**(step//update_c_every)
    elif type == 'lin':
        return lambda step: c0+eta*(step//update_c_every)
    else:
        print(type)
        raise NotImplementedError


def configure_opt_lr_scheduler(
    type: str,
    value: float,
    update_every: int
):
    if type == 'const':
        return lambda step: value
    elif type == 'lin':
        return lambda step: 1/(1+value*(step//update_every))
    elif type == 'exp':
        return lambda step: value**(-(step//update_every))
    else:
        print(type)
        raise NotImplementedError


def configure_attack_optimizer(type: str):
    if type=="Adam":
        return optim.Adam
    elif type=="SGD":
        return optim.SGD
    else:
        print(type)
        raise NotImplemented


def substitute_quam_config(config: dict):
    if 'c_scheduling' in config.keys():
        config['c_scheduling'] = configure_penopt_schedule(**config['c_scheduling'])
    if 'opt_lr_schedule' in config.keys():
        config['opt_lr_schedule'] = configure_opt_lr_scheduler(**config['opt_lr_schedule'])
    if 'loss_fn' in config.keys():
        config['loss_fn'] = nn.CrossEntropyLoss()
    if 'attack_optimizer' in config.keys():
        config['attack_optimizer'] = configure_attack_optimizer(**config['attack_optimizer'])
    return config


@torch.enable_grad()
def predict_quam(
    model: nn.Module,
    train_dl: tDataLoader,
    loss_fn: nn.Module,  # ModelBatchedCELoss()
    point: torch.Tensor,
    n_classes: int,
    n_epochs: int,
    gamma: float,
    c_scheduling: Callable[[int], float] = lambda step: 1 * 2 ** step,
    opt_lr_schedule: Callable[[int], float] = lambda step: 1.,
    log_callback: Callable = None, # consider removing or actually implementing
    attack_optimizer=optim.Adam,
    optimizer_kwargs: Dict = {},
    penalties_per_optimisation_step: int = -1,
    verbosity: int = VERBOSITY_SETTING_PROGRESS,
    toggle_train_mode_for_optimisation: bool = True, # avoid dropout during optimization, use True for batchnorm
    gradient_clipping_value: float = -1.,
    gradient_clipping_norm: float = -1.,
    parameter_subset_to_optimise: List[str] = None,
    move_cuda_async=False,
    use_main_memory_for_results=False,
    precomputed_train_loss = None,
    n_classes_total: int = -1,
):
    device = next(model.parameters()).data.device
    store_device = 'cpu' if use_main_memory_for_results else device
    dtype = next(model.parameters()).data.dtype
    point = point.to(device).to(dtype)

    model.eval()  # important because of dropout

    # handle the corner case of one point
    if point.dim() < 2:
        point = torch.unsqueeze(point, 0)

    n_points = point.shape[0]

    if penalties_per_optimisation_step < 0:
        penalties_per_optimisation_step = len(train_dl)
    n_penalties = len(train_dl) // penalties_per_optimisation_step

    if n_classes_total < n_classes:
        n_classes_total = n_classes

    assert len(train_dl) % n_penalties == 0, \
        "The number of batches per epoch must be divisible by number of penalties."

    # compute the loss on the training set for the initial model
    # DO NOT PROVIDE SHUFFLED DATALOADERS!
    if precomputed_train_loss is None:
        model.eval()
        # compute the loss on the given train set
        with torch.no_grad():
            model_train_loss = torch.zeros((len(train_dl), 1), dtype=dtype,
                                           device=device)  # this is stored on the device, since we need quick access

            if verbosity > VERBOSITY_SETTING_SCORES:
                print('Evaluating the model on the training data...')

            for i, (x, y) in enumerate(train_dl) if verbosity < VERBOSITY_SETTING_PROGRESS else tqdm(enumerate(train_dl)):
                model_train_loss[i, :], _ = _forward(x, y, model, loss_fn, device_movements_async=move_cuda_async)
                model_train_loss[i, :] = model_train_loss[i, :] # / len(train_dl)
    else:
        # use precomputed train loss here
        model_train_loss = precomputed_train_loss.to(device)

    with torch.no_grad():
        model.eval()
        average_net_pred = torch.softmax(model(point), dim=-1).detach()

    test_pt_preds = torch.zeros((n_points, n_epochs, n_penalties, n_classes, n_classes_total), dtype=dtype, device=store_device)

    # optimisation losses - losses on the test point
    opt_losses = torch.zeros((n_points, n_epochs, n_penalties, n_classes), dtype=dtype, device=store_device)
    # penalty loss dimensions unrelated to n_points
    pen_losses = torch.zeros((n_epochs, n_penalties, n_classes), dtype=dtype, device=store_device)

    # losses as collected by the validate (so per batch losses)
    train_losses = torch.zeros((n_epochs, len(train_dl), n_classes), dtype=dtype, device=store_device)

    # upscale the model now in batch dimension and add noise
    # ( only apply noise to parameters that, are optimized, otherwise its a bit stupid)
    adversarial = batch_that_net(model, n_classes)  # batch that net already performs deep copy

    batch_loss_fn = ModelBatchedCELoss(n_models=n_classes, reduction='none')
    # this reduces the batch dimension
    loss_red = lambda l: l.mean(dim=0) if loss_fn.reduction == 'mean' else l.sum(dim=0)

    if parameter_subset_to_optimise:
        opt = attack_optimizer([adversarial.get_parameter(n) for n in parameter_subset_to_optimise], **optimizer_kwargs)
        # disable gradient computation for all other parameters
        for n, p in adversarial.named_parameters():
            if not n in parameter_subset_to_optimise:
                p.requires_grad = False
    else:
        opt = attack_optimizer(adversarial.parameters(), **optimizer_kwargs)
    # search
    if toggle_train_mode_for_optimisation:
        adversarial.train()
    else:
        adversarial.eval()
    opt_lr_scheduler = LambdaLR(opt, lr_lambda=[opt_lr_schedule])

    for epoch in range(n_epochs) if verbosity < VERBOSITY_SETTING_PROGRESS else tqdm(range(n_epochs)):
        losses = torch.zeros(n_classes, dtype=dtype, device=device)
        for i, (x, y) in enumerate(train_dl):
            l, _ = _forward(feed_batch_model_image_multiple(x, n_classes) if x.ndim > 2 else feed_batch_model_vector_data(x, n_classes),
                            y, adversarial, batch_loss_fn, device_movements_async=move_cuda_async)
            # this makes the penalty term apply only if the loss on the train set rises during this optimisation
            # above a certain threshold (gamma)
            # assuming the dataloader does not shuffle
            # l at this point is [B, M], model_train_loss is [1] -> reduce the batched loss, upscale the
            # print(l.shape)

            # this is the consistent way to do it - I reduce it the same way as I do when loading the nets.
            # When this ls are summed, the dataset loss is obtained
            l = loss_red(l)
            c = c_scheduling(epoch*len(train_dl)+i)
            pen = (c * (l - model_train_loss[i].detach().expand(n_classes)))
            pen.sum(dim=0).backward()
            with torch.no_grad():
                losses += pen.detach()
                train_losses[epoch, i, :] = l.detach()

            if (i + 1) % penalties_per_optimisation_step == 0:
                cur_penalty = ((i + 1) // penalties_per_optimisation_step) - 1
                loss_adv, pt_pred = _forward(feed_batch_model_image_multiple(point, n_classes) if x.ndim > 2 else feed_batch_model_vector_data(point, n_classes),
                                    torch.arange(n_classes).unsqueeze(0).repeat(n_points, 1),
                                    adversarial,
                                    batch_loss_fn,
                                    )
                loss_adv = loss_red(loss_adv)
                loss_adv.sum(dim=0).backward()

                # gradient clipping section (disabled by default)
                if gradient_clipping_value > 0.:
                    for p in opt.param_groups:
                        clip_grad_value_(p['params'], gradient_clipping_value)
                if gradient_clipping_norm > 0.:
                    for p in opt.param_groups:
                        clip_grad_norm_(p['params'], gradient_clipping_norm)

                # apply the gradient to the parameters
                opt.step()
                opt.zero_grad()
                opt_lr_scheduler.step()

                with torch.no_grad():
                    test_pt_preds[:, epoch, cur_penalty, :, :] = torch.softmax(pt_pred.detach(), dim=-1).\
                        transpose(0, 1).to(store_device, non_blocking=move_cuda_async)

                    # record the losses
                    pen_losses[epoch, cur_penalty, :] = losses.to(store_device, non_blocking=move_cuda_async)
                    opt_losses[:, epoch, cur_penalty, :] = loss_adv.detach().to(store_device,
                                                                                non_blocking=move_cuda_async)

                    losses = torch.zeros_like(losses)

    pen_losses = pen_losses.flatten(0, 1).unsqueeze(0).expand(n_points, -1, -1)
    opt_losses = opt_losses.flatten(1, 2)
    train_losses = train_losses.flatten(0, 1).unsqueeze(0).expand(n_points, -1, -1)

    test_pt_preds = test_pt_preds.flatten(1, 2)

    return {
        'average_net_pred': average_net_pred,  # the initial network is regarded as the average network
        'sample_preds': test_pt_preds,
        'model_train_loss': model_train_loss,
        'train_loss': train_losses,
        'pen_loss': pen_losses,
        'obj_loss': opt_losses,
        'ensemble': adversarial if False else None,
        # 'best_opt_model_losses': best_opt_losses if sample_selection == "best" else None,
        # 'best_opt_model_update': best_opt_update if sample_selection == "best" else None,
        # 'best_opt_model_ensemble': adv_bestiary if sample_selection == "best" and return_models else None,
    }


def predict_quam_multiple(
    test_dl: tDataLoader,
    verbosity: int = VERBOSITY_SETTING_PROGRESS,
    record_true_label: bool = True,
    **quam_args,
):
    retlist = []
    model_train_loss = None
    for i, pt in (enumerate(test_dl) if verbosity < VERBOSITY_SETTING_PROGRESS else tqdm(enumerate(test_dl))):
        sample = predict_quam(point=xonly(pt), **quam_args, verbosity=verbosity-1, precomputed_train_loss=model_train_loss)
        model_train_loss = sample['model_train_loss'] # optimisation to not redo this every time
        if record_true_label:
            sample['target'] = yonly(pt)
        sample = dict_to_cpu(dict_detach(sample))
        retlist.append(sample)

    return retlist


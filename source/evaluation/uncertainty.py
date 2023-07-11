import torch
import numpy as np
from typing import Tuple
from torch.nn.functional import one_hot


@torch.no_grad()
def combine_sample_best(
    average_net_pred: torch.Tensor,  # [n_points, n_output_features]
    sample_preds: torch.Tensor,  # [n_points, n_samples, n_models, n_output_features]
    train_loss,  # [n_points, n_samples, n_models]
    model_train_loss, # [n_samples//n]
    obj_loss, # [n_points, n_samples, n_models]
    window_size: int = 1,
    gamma_slack: float = 1e-2,
    **kwargs
):
    n_points = sample_preds.shape[0]
    n_samples = sample_preds.shape[1]
    n_models = sample_preds.shape[2]
    n_classes = sample_preds.shape[3]

    # n_epochs = train_loss.shape[1] // model_train_loss.shape[0]
    # match the sizes up, should be off by just a constant FIXME: might break with penalty n > 1
    # model_train_loss = model_train_loss.unsqueeze(0).repeat(train_loss.shape[1])
    # model_train_loss_checkem = model_train_loss.unsqueeze(1).repeat(1, n_samples//model_train_loss.shape[0]).flatten(0, 1).unsqueeze(1).unsqueeze(0) # [1, n_samples, 1]
    model_train_loss_checkem = model_train_loss.repeat(n_samples // model_train_loss.shape[0]).unsqueeze(1).unsqueeze(0)  # [1, n_samples, 1]
    model_train_loss_checkem = model_train_loss_checkem.unfold(1, window_size, 1) # [1, n_windows, 1, windows_size]
    # this results in [n_points, n_windows, n_models, window_size]
    train_loss_checkem = train_loss.unfold(1, window_size, 1) # dimension, window_size, stride
    boundary_passed = torch.all(model_train_loss_checkem > (train_loss_checkem-gamma_slack), dim=-1)
    boundary_passed[:, 0, :] = True # set the original model to be within the boundary regardless of the window
    # [n_points, n_windows, n_models]
    # half_window = window_size//2
    # window_parity = window_size - half_window*2

    # get the lowest indices
    # opt_checkem = obj_loss[:, half_window-1*(1-window_parity):(-half_window), :] if window_size > 1 else obj_loss
    opt_checkem = obj_loss[:, :-window_size+1, :] if window_size > 1 else obj_loss
    opt_checkem[~boundary_passed] = torch.inf # [n_points, n_windows, n_models]
    pick_samples = torch.min(opt_checkem, dim=1)[1] # [n_points, n_models]

    # create a mask
    # pick_samples_bool = one_hot(pick_samples, sample_preds.shape[0]).to(dtype=torch.bool)
    # sample_preds_ = sample_preds
    # sample_preds_[pick_samples] = 0
    # now remove all those that are below

    # my vectorization doesnt seem to work today :(
    buffer = torch.zeros_like(sample_preds)[:, 1, :, :].unsqueeze(1)
    for i in range(n_points):
        for j in range(n_models):
            buffer[i, 0, j, :] = sample_preds[i, pick_samples[i, j], j, :]

    return {
        'average_net_pred': average_net_pred,
        'sample_preds': buffer,
    }


@torch.no_grad()
def combine_sample_last(
    average_net_pred: torch.Tensor, # [n_points, n_output_features]
    sample_preds: torch.Tensor, # [n_points, n_samples, n_models, n_output_features]
    train_loss, # [n_points, n_samples, n_models]
    **kwargs
):
    return {
        'average_net_pred': average_net_pred,
        'sample_preds': sample_preds[:, -1, :, :].unsqueeze(1),
    }


@torch.no_grad()
def combine_sample_softmax(
    average_net_pred: torch.Tensor,  # [n_points, n_output_features]
    sample_preds: torch.Tensor,  # [n_points, n_samples, n_models, n_output_features]
    train_loss: torch.Tensor,  # [n_points, n_samples, n_models]
    temperature: float = 1.,
    num_samples: int = None,
    **kwargs
):
    if num_samples is None:
        num_samples = train_loss.shape[2]

    sample_preds = sample_preds[:, :num_samples, :, :]
    train_loss = train_loss[:, :num_samples, :]

    train_loss = train_loss.contiguous()
    train_loss_shape = train_loss.shape
    # softmax of train set losses over the whole 'trajectory', so for all classes

    train_loss_sm = torch.softmax(-train_loss.view(train_loss_shape[0], -1)/temperature, dim=-1).view(*train_loss_shape)
    return {
        'average_net_pred': average_net_pred,
        'sample_preds': sample_preds,
        'sample_weights': train_loss_sm
    }


@torch.no_grad()
def combine_sample_all(
    average_net_pred: torch.Tensor, # [n_points, n_output_features]
    sample_preds: torch.Tensor, # [n_points, n_samples, n_models, n_output_features]
    **kwargs
):
    return {
        'average_net_pred': average_net_pred,
        'sample_preds': sample_preds[:, -1, :, :].unsqueeze(1),
    }


@torch.no_grad()
def calculate_uncertainty_setting_b(
    average_net_pred: torch.Tensor,
    sample_preds: torch.Tensor,
    sample_weights: torch.Tensor = None,
    gamma=1e-10,
    **kwargs
):
    '''

    :param average_net_pred:
    [n_points, n_categories]
    :param sample_preds:
    [n_points, n_posterior_samples, n_models, n_classes]
    :param sample_weights:
    [n_points, n_posterior_samples, n_models]
    :param kwargs:
    :return:
    '''

    if sample_weights is None:
        total = - torch.mean(
            torch.sum(
                (average_net_pred.unsqueeze(1).unsqueeze(2) * torch.log(sample_preds + gamma)), dim=-1), # .unsqueeze(2)
            dim=(1, 2))
        aleatoric = - torch.sum( (average_net_pred+gamma) * torch.log(average_net_pred+gamma), dim=-1)
        # print(total, aleatoric)
        epistemic = total - aleatoric
    else:
        # sample weight implied to be normalized
        total = - torch.sum(
            torch.sum(
                (average_net_pred.unsqueeze(1).unsqueeze(2) * torch.log(sample_preds + gamma)), dim=-1)*sample_weights, # .unsqueeze(2)
            dim=(1, 2))
        aleatoric = - torch.sum( (average_net_pred+gamma) * torch.log(average_net_pred+gamma), dim=-1)
        # print(total, aleatoric)
        epistemic = total - aleatoric

    return {
        'total': total,
        'aleatoric': aleatoric,
        'epistemic': epistemic,
    }


def calculate_uncertainty_setting_a(
    average_net_pred: torch.Tensor,
    sample_preds: torch.Tensor,
    sample_weights: torch.Tensor = None,
    gamma=1e-10,
    **kwargs
):
    '''
    if provided, uses the optimization steps
    :param kwargs:
    :return:
    '''
    if sample_weights is None:
        total = - torch.sum(torch.mean(sample_preds, dim=(0, 1)) * torch.log(torch.mean(sample_preds, dim=(0, 1))), dim=0)
        aleatoric = - torch.mean(torch.sum(sample_preds * torch.log(sample_preds), dim=-1), dim=(0, 1))
        epistemic = total - aleatoric
    else:
        total = - torch.sum(torch.sum(sample_preds*sample_weights, dim=(0, 1)) * torch.log(torch.sum(sample_preds*sample_weights, dim=(0, 1))), dim=0)
        aleatoric = - torch.mean(torch.sum(sample_preds * torch.log(sample_preds), dim=-1), dim=(0, 1))
        epistemic = total - aleatoric

    return {
        'total': total,
        'aleatoric': aleatoric,
        'epistemic': epistemic,
    }

import torch
from torch import nn
from typing import List, Dict, Union

from torch.utils.data import DataLoader

from tqdm import tqdm
from source.constants import VERBOSITY_SETTING_PROGRESS
from source.utils.training import xonly, yonly, dict_to_cpu, dict_detach


@torch.no_grad()
def predict_ensemble(
    model: nn.ModuleList,
    points: torch.Tensor,
    verbosity: int = 0,
    **kwargs,  # actually unnecessary as long as the dataloaders are specified properly in the config
):
    points = points.to(next(model[0].parameters()).device).to(next(model[0].parameters()).dtype)

    [n.eval() for n in model]
    preds = torch.softmax(torch.stack([n(points) for n in model], dim=1), dim=-1)
    
    mean_preds = preds.mean(1)
    # mean_preds = preds[:, 0, :] # treat the first model as the determinisitc model

    preds = preds.unsqueeze(1)

    return {
        'sample_preds': preds,
        'average_net_pred': mean_preds,
    }


def predict_ensemble_multiple(
    test_dl: DataLoader,
    model: nn.ModuleList,
    verbosity: int = VERBOSITY_SETTING_PROGRESS,
    record_true_label: bool = True,
    **kwargs
):
    retlist = []
    for i, batch in enumerate(test_dl) if verbosity < VERBOSITY_SETTING_PROGRESS else tqdm(enumerate(test_dl)):
        sample = predict_ensemble(
            points = xonly(batch),
            model=model,
            verbosity=verbosity-1,
            **kwargs,
        )
        if record_true_label:
            sample['target'] = yonly(batch)
        retlist.append(dict_to_cpu(dict_detach(sample)))
    return retlist


def predict_ensemble_parallel(
    test_dl: DataLoader,
    verbosity: int = 0,
    **kwargs
):
    raise NotImplementedError

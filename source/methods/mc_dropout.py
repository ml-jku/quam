import torch
from torch import nn

from tqdm import tqdm
from source.constants import VERBOSITY_SETTING_PROGRESS
from source.utils.training import xonly, yonly, dict_to_cpu, dict_detach

from torch.utils.data import DataLoader


@torch.no_grad()
def predict_mc_dropout(
    model: nn.Module,
    points: torch.Tensor, # [n_points, n_features]
    n_samples: int,
    verbosity: int = 0,
    **kwargs,
):
    points = points.to(next(model.parameters()).device).to(next(model.parameters()).data.dtype)

    model.eval()
    avg_net_pred = torch.softmax(model(points), dim=-1).detach()

    # consider here, that the 'persistent dropout' layers as I have them are not friends with batching
    model.train()
    sample_list = []
    for i in range(n_samples) if verbosity < VERBOSITY_SETTING_PROGRESS else tqdm(range(n_samples)):
        sample_pred = torch.softmax(model(points), dim=-1).detach()
        sample_list.append(sample_pred)

    sample = torch.stack(sample_list, dim=0)
    sample = torch.reshape(sample, (n_samples, points.shape[0], 1, *sample.shape[2:])).transpose(0, 1)

    return {
        'average_net_pred': avg_net_pred,
        'sample_preds': sample,
    }


def predict_mc_dropout_multiple(
    test_dl: DataLoader,
    verbosity: int = VERBOSITY_SETTING_PROGRESS,
    record_true_label: bool = True,
    **kwargs
):
    retlist = []
    for i, batch in enumerate(test_dl) if verbosity < VERBOSITY_SETTING_PROGRESS else tqdm(enumerate(test_dl)):
        sample = predict_mc_dropout(
            points = xonly(batch),
            **kwargs,
            verbosity=verbosity-1,
        )
        if record_true_label:
            sample['target'] = yonly(batch)
        retlist.append(dict_to_cpu(dict_detach(sample)))
    return retlist


def predict_mc_dropout_parallel(
    test_dl: DataLoader,
    verbosity: int = 0,
    **kwargs
):
    raise NotImplementedError

import torch
from typing import Dict, List

from copy import deepcopy

from .mc_dropout import predict_mc_dropout_multiple
from .ensembles import predict_ensemble_multiple
from .quam import predict_quam_multiple, substitute_quam_config
from .sgmcmc import predict_sgmcmc_dl
from .laplace import predict_laplace_dl


def get_method_as_configured(m_config: dict):
    for k, v in m_config.items():
        if k == 'mc_dropout':
            return predict_mc_dropout_multiple, v
        elif k == 'ensemble':
            return predict_ensemble_multiple, v
        elif k == 'quam':
            # copy the dictionary in order to avoid unreadable config dumps
            return predict_quam_multiple, substitute_quam_config(deepcopy(v))
        elif k == 'sgmcmc':
            return predict_sgmcmc_dl, v
        elif k == 'laplace':
            return predict_laplace_dl, v
        else:
            raise NotImplementedError


def concat_method_outputs(outputs: List[Dict], label_subset: List = None):
    assert len(outputs) != 0, "Empty outputs provided"

    if label_subset is None:
        label_subset = list(outputs[0].keys())

    nudic = {}

    for l in label_subset:
        # if isinstance(outputs[0][l], torch.Tensor):
        #     # concatenate
        # print(l, outputs[0][l].shape if isinstance(outputs[0][l], torch.Tensor) else 'nontensor')
        if l == "average_net_pred":
            nudic[l] = torch.cat([o[l] if o[l].ndim == 2 else o[l].unsqueeze(0) for o in outputs], dim=0)
        elif l == "sample_preds":
            nudic[l] = torch.cat([o[l] if o[l].ndim == 4 else o[l].unsqueeze(0) for o in outputs], dim=0)
        elif l == "train_loss" or l == "pen_loss" or l == "obj_loss":
            nudic[l] = torch.cat([o[l] for o in outputs], dim=0)
        elif l == "model_train_loss":
            # the loss of the base model on the training set, should be the same for all methods for all points
            # nudic[l] = torch.cat([o[l].squeeze().unsqueeze(0) for o in outputs], dim=0)
            nudic[l] = outputs[0][l].squeeze()
        elif l == "target":
            # the label
            nudic[l] = torch.cat([o[l] for o in outputs], dim=0)
        else:
            nudic[l] = [o[l] for o in outputs]
   
    return nudic

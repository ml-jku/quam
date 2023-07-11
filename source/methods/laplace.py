import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from laplace import Laplace

from source.constants import VERBOSITY_SETTING_PROGRESS, VERBOSITY_SETTING_SILENT
from source.utils.training import xonly, dict_to_cpu, dict_detach, yonly


def predict_laplace(
    model: nn.Module,
    train_dl: DataLoader,
    points: torch.Tensor, # [n_points, n_features]
    lr: float,
    h_epochs: int,
    n_samples: int,
    likelihood: str = 'classification',  # 'classification' or 'regression'
    subset_of_weights: str = 'all',  # 'last_layer' or 'all' ('subnetwork' also an option, but more complicated)
    hessian_structure: str = 'diag',  # 'diag', 'kron', 'full', 'lowrank'
    pred_type: str = 'glm', # 'glm', 'nn'
    verbosity: int = 0,
):  
    points = points.to(next(model.parameters()).device).to(next(model.parameters()).data.dtype)

    # get original network prediction
    model.eval()
    avg_net_pred = torch.softmax(model(points), dim=-1).detach()

    # laplace approximation
    model.train()
    la = Laplace(model, likelihood, subset_of_weights, hessian_structure)

    la.fit(train_dl)
    
    # Optimize prior precision
    if likelihood == "regression":
        log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        hyper_optimizer = optim.Adam([log_prior, log_sigma], lr=lr)
    else:
        log_prior = torch.ones(1, requires_grad=True)
        hyper_optimizer = optim.Adam([log_prior], lr=lr)

    for i in tqdm(range(h_epochs)):
        hyper_optimizer.zero_grad()
        if likelihood == "regression":
            neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        else:
            neg_marglik = - la.log_marginal_likelihood(log_prior.exp())
        neg_marglik.backward()
        hyper_optimizer.step()

    model.eval()
    pred_list = la.predictive_samples(points, pred_type=pred_type, n_samples=n_samples).permute(1,0,2).unsqueeze(2)
    print('pred list shape', pred_list.shape)
    # n_samples is first dimension!

    return {
        'average_net_pred': avg_net_pred,
        'sample_preds': pred_list,
    }


def predict_laplace_dl(
    test_dl: DataLoader,
    verbosity: int = VERBOSITY_SETTING_PROGRESS,
    record_true_label: bool = True,
    **laplace_args
):
    retlist = []
    # ideally, this would contain only one batch
    for i, pt in tqdm(enumerate(test_dl)):
        sample = predict_laplace(points=xonly(pt), **laplace_args, verbosity=verbosity)
        if record_true_label:
            sample['target'] = yonly(pt)
        sample = dict_to_cpu(dict_detach(sample))
        retlist.append(sample)

    return retlist

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_sgld import SGLD, CosineLR

from tqdm import tqdm

from source.constants import VERBOSITY_SETTING_PROGRESS
from source.data.wrappers import InfiniteDataLoaderWrapper
from source.utils.training import xonly, dict_to_cpu, dict_detach, yonly

"""
How do algorithms belong to each other? (All are Stochastic Gradient MCMC (SG-MCMC) algorithms):
if n_cycles > 1:
    if momentum > 0:
        cSG-HMC
    elif momentum == 0:
        cSGLD
elif n_cycles == 1:
    if momentum > 0:
        SG-HMC
    elif momentum == 0:
        SGLD
"""


def predict_sgmcmc(
    model: nn.Module,
    train_dl: DataLoader,
    points: torch.Tensor, # [n_points, n_features]
    n_cycles: int,
    n_iterations: int,
    n_samples: int,
    momentum: float,
    lr: float,
    sched_beta: float,
    temperature: float,
    loss_fn: nn.Module = nn.CrossEntropyLoss(),
    verbosity: int = 0,
    **kwargs,
):  
    device = next(model.parameters()).device
    dtype = next(model.parameters()).data.dtype
    points = points.to(device=device, dtype=dtype)

    # Get original prediction
    model.eval()
    avg_net_pred = torch.softmax(model(points), dim=-1).detach()

    # Initialize MCMC samples and scheduler
    sgmcmc = SGLD(model.parameters(), lr=lr, momentum=momentum, temperature=temperature)
    sgmcmc_scheduler = CosineLR(sgmcmc, n_cycles=n_cycles, n_samples=n_samples, T_max=n_iterations, beta=sched_beta)

    # Setup for sampling
    model.train()
    train_dl = InfiniteDataLoaderWrapper(train_dl, max_n=n_iterations)
    sample_preds = list()

    # Sampling
    for i, (x, y) in enumerate(train_dl) if verbosity < VERBOSITY_SETTING_PROGRESS else tqdm(enumerate(train_dl)):
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device)

        sgmcmc.zero_grad()

        loss = loss_fn(model.forward(x), y)

        loss.backward()
        if sgmcmc_scheduler.get_last_beta() <= sgmcmc_scheduler.beta:
            sgmcmc.step(noise=False)
        else:
            sgmcmc.step()

            if sgmcmc_scheduler.should_sample():
                with torch.no_grad():
                    sample_preds.append(torch.softmax(model(points), dim=-1).detach())

        sgmcmc_scheduler.step()

    return {
        'average_net_pred': avg_net_pred,
        'sample_preds': torch.stack(sample_preds, dim=1).unsqueeze(2),
    }


def predict_sgmcmc_dl(
    test_dl: DataLoader,
    verbosity: int = VERBOSITY_SETTING_PROGRESS,
    record_true_label: bool = True,
    **sgmcmc_kwargs
):
    retlist = []
    # ideally, this would contain only one batch
    for i, pt in tqdm(enumerate(test_dl)):
        sample = predict_sgmcmc(points=xonly(pt), **sgmcmc_kwargs, verbosity=verbosity)
        if record_true_label:
            sample['target'] = yonly(pt)
        sample = dict_to_cpu(dict_detach(sample))
        retlist.append(sample)

    return retlist

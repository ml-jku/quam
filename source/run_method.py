import torch
from torch import nn

import argparse
from .config import load_master_config, save_results

from typing import Dict, Callable, List

# uncertainty quantification on a given model for given dataset using given method
from torch.utils.data import DataLoader

from .data import get_dataset_as_configured
from .networks import get_network_as_configured
from .methods import get_method_as_configured, concat_method_outputs

from .utils.seeding import fix_seeds

import os
import yaml


def run_method(
    dataloaders: dict,
    network: nn.Module,
    method: Callable,
    method_args: dict,
    config,
):
    if 'RNG_SEED' in os.environ:
        rng_seed = os.environ['RNG_SEED']
        config['RNG_SEED'] = int(rng_seed)
        fix_seeds(int(rng_seed))
    else:
        fix_seeds()
        config['RNG_SEED'] = 42
    # dump the predictions to disk
    savedir_path = os.path.abspath(config['save_to']['path'])
    # the prefix allows different subsets to write to the same directory without overwriting
    run_prefix = str(config['save_to']['prefix'])+'_' if 'prefix' in config['save_to'] and config['save_to']['prefix'] is not None else ''
    os.makedirs(savedir_path, exist_ok=True)

    with open(os.path.join(savedir_path, run_prefix+'effective_config.yaml'), 'wt') as f:
        yaml.dump(config, f)

    preds = method(model=network, **dataloaders, **method_args)
    preds = concat_method_outputs(preds)

    if 'items' in config['save_to']:
        for fname, outs in config['save_to']['items'].items():
            # aggregate the needed outputs
            dumpable = {k: v for k, v in preds.items() if k in outs}
            save_results(os.path.join(savedir_path, run_prefix + fname), dumpable)

    return preds


def run_method_configured(config_name: str):

    master_config = load_master_config(config_name)

    ds_config = master_config['dataset']
    net_config = master_config['network']
    method_config = master_config['method']

    dsdl = {k: get_dataset_as_configured(**v) for k, v in ds_config.items()}
    dataloaders = {k: v[1] for k, v in dsdl.items()}
    # load the network the same way
    network = get_network_as_configured(**net_config)

    # initialize method and the parameters
    method, method_args = get_method_as_configured(method_config)

    return run_method(
        dataloaders,
        network,
        method,
        method_args,
        master_config,
    )


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='parse key pairs into a dictionary')
    parser.add_argument("--config", dest="config", action='store',
                        metavar="Name of the configuration file in the config folder.", default='default')
    parser.add_argument("--para", dest="para", action='store',
                        metavar="Name of the configuration file in the config folder.", default='0/0')
    args = parser.parse_args()
    config_name = args.config
    pointwise_split = args.para

    if pointwise_split!='0/0':
        if 'RUN_ON_SUBSAMPLE' not in os.environ:
            raise Exception("Attempting to split the test sets without DS_SUBSET variable set")
        curr, total = [int(a) for a in pointwise_split.split('/')]
        dss_low, dss_high = [int(a) for a in os.environ['RUN_ON_SUBSAMPLE'].split(':')]

        assert curr < total
        assert dss_low < dss_high

        true_offset_low = dss_low + ((dss_high-dss_low) // total) * curr
        true_offset_high = dss_high - ((dss_high-dss_low) // total) * (total - curr - 1)

        os.environ['RUN_ON_SUBSAMPLE'] = f"{true_offset_low}:{true_offset_high}"
        os.environ['RUN_PREFIX'] = f"{curr}_"

        print(f"Note: para has been specified, current run will evaluate on range {true_offset_low}:{true_offset_high} with prefix {curr}, make sure that all the configuration is set appropriately.")

    run_method_configured(config_name)

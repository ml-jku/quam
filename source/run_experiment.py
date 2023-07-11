import argparse
import os

import torch
import pickle

from .config import load_master_config, dict_to_env, dict_to_env_unset, flatten_dicts, timestamp_str, load_results
from .run_method import run_method_configured
from .evaluation import uncertainty_fns_map, evaluation_fns_map, filter_sample_fns_map
from .evaluation.ood import evaluate_ood
from .evaluation.uncertainty import combine_sample_softmax, calculate_uncertainty_setting_b

import wandb

from functools import partial, reduce

import traceback

import yaml
from copy import deepcopy

from typing import Tuple, List


def obtain_resource(path: Tuple[str, List] = None,
                    env: dict = {},
                    config: str = None):
    if path is not None:
        if isinstance(path, str):
            return load_results(path)
        elif isinstance(path, List):
            return reduce(lambda a, b: {**a, **b}, [load_results(p) for p in path], {})
        else:
            assert False, f"Confusing datatype of path {type(path)}"
    elif config is not None:
        # preconfigure the method
        dict_to_env(env)
        # load the configuration file
        values = run_method_configured(config)
        # reset all the env variables for that run of the method
        dict_to_env_unset(env)
        return values
    else:
        raise NotImplementedError("")


def config_and_launch_experiment(config_name: str):
    # now we need to load all samples mentioned in the configuration
    master_config = load_master_config(config_name)

    predictions = {}
    for sname, sample_cfg in master_config['load_samples'].items():
        predictions[sname] = obtain_resource(**sample_cfg)

    # compute the uncertainties based on them as per configuration
    uncerts_config = master_config['samples2uncerts']
    computed_uncertainties = {}
    for uncert, u_config in uncerts_config.items():
        unc_fn = uncertainty_fns_map[uncert]
        if 'filter' in u_config:
            filter_params = u_config['filter']
            filter_fn = filter_sample_fns_map[u_config['filter']['name']]
            filter_params = deepcopy(filter_params)
            del(filter_params['name'])
            params = u_config['unc_params']
        else:
            filter_fn = None
            params = u_config['unc_params']
        computed_uncertainties[uncert] = {}
        for unc_type in ['aleatoric', 'epistemic', 'total']:
            computed_uncertainties[uncert][unc_type] = {}
        for pred_name, pred_values in predictions.items():
            if filter_fn is not None:
                pred_values = filter_fn(**pred_values, **filter_params)
            unc_dict = unc_fn(**pred_values, **params)
            for unc_type, unc_vals in unc_dict.items():
                # this order, because we want to unpack the pred_name dimension to the eval_fn later on
                computed_uncertainties[uncert][unc_type][pred_name] = unc_vals

    # perform the evaluation as per configuration
    resdic = {}
    for experiment_name, experiment_params in master_config['experiment_evaluation'].items():
        eval_fn = evaluation_fns_map[experiment_name]
        extra_body = {}
        for uncert, unc_vals in computed_uncertainties.items():
            for unc_type, unc_package in unc_vals.items():
                # print(experiment_name, uncert, unc_type)
                if 'provide_extras' in experiment_params:
                    extras = experiment_params['provide_extras']
                    for e in extras:
                        extra_body[list(unc_package.keys())[0]+'_'+e] = predictions[list(unc_package.keys())[0]][e]
                    del(experiment_params['provide_extras'])
                resdic[experiment_name+'_'+uncert+'_'+unc_type] = eval_fn(**unc_package, **experiment_params, **extra_body)

    if 'save_to' in master_config:
        dump_path = master_config['save_to']['path']
        os.makedirs(dump_path, exist_ok=True)
        with open(os.path.join(dump_path, "result.pkl"), "wb") as f:
            pickle.dump(resdic, f)
        with open(os.path.join(dump_path, "uncerts_computed.pkl"), "wb") as f:
            pickle.dump(computed_uncertainties, f)
        with open(os.path.join(dump_path, "experiment_config.yaml"), "wt") as f:
            yaml.dump(master_config, f)
        # dump the result twice to make it easier to track things down
        with open(os.path.join(dump_path, "result.yaml"), "wt") as f:
            yaml.dump(resdic, f)

    return resdic, predictions


def wandb_run_experiment(experiment_config: str):
    try:
        current_run = wandb.init()

        hparams_config = current_run.config
        hparams_config['WANDB_RNAME'] = current_run.name    # add the wandb run name to the env variables

        print(experiment_config)
        dict_to_env(hparams_config)

        # run the experiment
        retvals, predictions = config_and_launch_experiment(experiment_config)

        # log the result into wandb
        current_run.log(flatten_dicts(retvals))

        best_setting = analyze_nsteps_temp(predictions)

    except Exception as e:
        print(str(e))
        print(traceback.format_exc())


def analyze_nsteps_temp(predictions, num_epochs=5):
        # plot auroc w.r.t temperature and number of update steps
        num_samples = list(range(20, 60, 10)) + list(range(64, num_epochs * 64 + 1, 64))

        t_values = [round(x * 1e-4, 4) for x in
                    list(range(1, 10, 1)) +
                    list(range(10, 100, 10))] + [1] 
        
        best_setting = {"best_auroc": 0}
        results_list = list()
        for s in num_samples:
            epi_auroc_list = list()
            for t in t_values:
                epi_auroc = evaluate_ood(**{k: calculate_uncertainty_setting_b(**combine_sample_softmax(temperature=t, num_samples=s, **v))['epistemic'] for k, v in predictions.items()})['AUROC']
                if epi_auroc > best_setting["best_auroc"]:
                    best_setting["best_auroc"] = epi_auroc
                    best_setting["best_nsamp"] = s
                    best_setting["best_temp"] = t
                epi_auroc_list.append(epi_auroc)

            results_list.append(epi_auroc_list)

        wandb.log({"auroc_epistemic_uncertainty" : wandb.plot.line_series(xs=t_values, ys=results_list, keys=num_samples, title="Two Random Metrics", xname="Temperature")})
        wandb.log(best_setting)

        return best_setting


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='parse key pairs into a dictionary')
    parser.add_argument("--config", dest="config", action='store',
                        metavar="Name of the configuration file in the config folder.", default='default')
    parser.add_argument("--use_wandb_sweep", dest="wandb_sweep_id", action='store',
                        metavar="Name of the wandb sweep to use for fetching hyperparameters.", default=None)
    parser.add_argument("--n_tries", dest="n_tries", action='store', type=int,
                        metavar="Number of wandb hparams to fetch and try.", default=1)
    args = parser.parse_args()
    config_name = args.config
    wandb_sweep_id = args.wandb_sweep_id
    wandb_ntries = args.n_tries

    if wandb_sweep_id is not None:
        sweep_id_parsed = wandb_sweep_id.split('/')
        wb_entity, wb_project, wb_sweep = [None]*(3-len(sweep_id_parsed)) + sweep_id_parsed
        # set the sweep name to env
        os.environ['WANDB_SWEEPNAME'] = wb_sweep
        # start and agent
        wandb.agent(entity=wb_entity, project=wb_project, sweep_id=wb_sweep,
                    function=partial(wandb_run_experiment, experiment_config=config_name), count=wandb_ntries)

    else:
        if not 'WANDB_SWEEPNAME' in os.environ:
            os.environ['WANDB_SWEEPNAME'] = "standalone"
        if not 'WANDB_RNAME' in os.environ:
            os.environ['WANDB_RNAME'] = timestamp_str()

        # current_run = wandb.init(project="quam_imagenet", name=f"hp1")
        resdic, predictions = config_and_launch_experiment(config_name)

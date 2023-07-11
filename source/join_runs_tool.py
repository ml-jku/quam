import argparse
import pickle
import os

from .config import load_master_config
from .methods import concat_method_outputs


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='parse key pairs into a dictionary')
    parser.add_argument("--config", dest="config", action='store',
                        metavar="Name of the configuration file in the config folder.", default='default')
    parser.add_argument("--n_instances", dest="insta", action='store', type=int,
                        metavar="Name of the configuration file in the config folder.", default='1')
    args = parser.parse_args()
    config_name = args.config
    n_instances = args.insta


    config = load_master_config(config_name)
    files_to_combine = list(config['save_to']['items'].keys())
    accumulator = {fn: [] for fn in files_to_combine}
    # load all dump files from the save_to directory
    for fref in files_to_combine:
        for i in range(n_instances):
            with open(os.path.join(config['save_to']['path'], str(i)+'_'+fref), 'rb') as f:
                accumulator[fref].append(pickle.load(f))

    accumulator_combined = {k: concat_method_outputs(preds) for k, preds in accumulator.items()}

    for fname, outs in accumulator_combined.items():
        # aggregate the needed outputs
        with open(os.path.join(config['save_to']['path'], fname), 'wb') as f:
            pickle.dump(outs, f)

    print(f'done, files combined in {config["save_to"]["path"]}')
import os
import pickle

import yaml

from typing import Dict, Union, List
from functools import reduce
from .constants import GIT_CURRENT_HEAD_SHA
import re

import numpy as np
import torch


digitize_table_high = [i/100 for i in range(101)] + list(np.logspace(-0.5, -10, num=157))
digitize_table_high = np.asarray(list(reversed(sorted(digitize_table_high)))).astype(np.float32)


def safe_numpy(t):
    if t.dtype==torch.bfloat16:
      return t.to(dtype=torch.float).numpy()
    else:
      return t.numpy()


def digitize_nparray(arr: np.ndarray):
    assert np.alltrue(arr <= 1.) and np.alltrue(arr >= 0)
    return np.digitize(arr, digitize_table_high).astype(np.uint8)


def undigitize_nparray(arr: np.ndarray):
    return digitize_table_high[arr]


def save_results(fname: str, content_dict: dict):
    if fname.endswith('.npy'):
        assert all([isinstance(a, torch.Tensor) for a in content_dict.values()]), 'only allowed to dump tensors into npy.npz'
        content_dict = {k: digitize_nparray(safe_numpy(v)) for k, v in content_dict.items()}
        np.savez_compressed(fname, **content_dict)
    else:
        with open(fname, 'wb') as f:
            pickle.dump(content_dict, f)
    return


def load_results(fname: str):
    if fname.endswith('.npy'):
        fname+='.npz'
        res = {k: v for k, v in np.load(fname).items()}
        assert all({isinstance(v, np.ndarray) for v in res.values()}), "must be loading python files"
        res = {k: torch.from_numpy(undigitize_nparray(v)) for k, v in res.items()}
        return res
    else:
        with open(fname, 'rb') as f:
            buff = pickle.load(f)
        return buff


def dict_to_env(d: dict, prefix=''):
    for k, v in d.items():
        if k in os.environ:
            # stash the old value
            os.environ['__'+prefix+k] = os.environ[prefix+k]
        os.environ[prefix+k] = yaml.safe_dump(v)[:-5]  # to make sure that floats are safe


def dict_to_env_unset(d: dict, prefix=''):
    for k, v in d.items():
        if '__'+prefix+k in os.environ:
            # stash the old value
            os.environ[prefix+k] = os.environ['__'+prefix+k]
            os.environ.pop('__' + prefix + k)
        else:
            os.environ.pop(prefix+k)


def flatten_dicts(d: dict, context = []):
    retdict = {}
    for k, v in d.items():
        if isinstance(v, Dict):
            retdict = {**retdict, **flatten_dicts(v, context+[k])}
        else:
            retdict['_'.join(context+[k])] = v
    return retdict


def read_config(
    name: str,
    subject: str = "",
    abspath = False
):

    fpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         'config', subject, name+'.yaml')) if not abspath else os.path.abspath(name)

    with open(fpath, 'rt') as f:
        config = yaml.load(f, yaml.SafeLoader)

    return config


def load_master_config(
    name: str,
    abspath = False,
    substitute_vars = True,
):
    '''
    :param name:
        name of the configuration file
    :param abspath:
        whether the name is an absolute path
    :param substitute_vars:
        whether to substitute variables. This substitutes substrings like ${varname|default_value} to a corresponding
        environment variable
    :return:
    master config fully assembled
    '''
    master_cfg = read_config(name, '', abspath)
    full_cfg = {k: read_config(v, subject=k, abspath=abspath) if isinstance(v, str) else v
                for k, v in master_cfg.items()}  # if we have a dict, dont try to get a config

    full_cfg['quam_repo_version'] = GIT_CURRENT_HEAD_SHA

    # extract the variables, thy shall not nest
    if 'env' in full_cfg.keys():
        dict_to_env(full_cfg['env'])

    if substitute_vars:
        config_sub_recursive_dict(full_cfg)

    return full_cfg


def sub_str_parameter(p: str):
    # Use REGEX to substitute the parameters
    var_pattern = "\$\{.*?\}" # "(?<=\$\{).*?(?=\})"
    p_var = re.findall(var_pattern, p)
    if len(p_var) == 0:
        return p
    else:
        p_static = re.split(var_pattern, p)
        # the following will produce an error if neither the environmental variable is set nor the default value mentioned
        try:
            p_var = [os.environ[vname[0]] if vname[0] in os.environ else vname[1] for vname in [vname_raw[2:-1].split('|') for vname_raw in p_var]] + ['']
        except IndexError as err:
            print(p_var)
            for v in p_var:
                if not v[2:-1].split('|')[0] in os.environ:
                    print(f"Missing env {v}")
            raise err
        all_pieces = reduce(lambda a,b: a+b, list(zip(p_static, p_var)))
        return yaml.safe_load(''.join(all_pieces))  # reinterpret the yaml line to allow numbers/floats insertion


def config_sub_recursive_dict(cfg_rec: Union[Dict, List]):
    if isinstance(cfg_rec, List):
        for i, v in enumerate(cfg_rec):
            if isinstance(v, str):
                cfg_rec[i] = sub_str_parameter(v)
            elif isinstance(v, dict) or isinstance(v, list):
                config_sub_recursive_dict(v)
    elif isinstance(cfg_rec, Dict):
        for k, v in cfg_rec.items():
            if isinstance(v, str):
                cfg_rec[k] = sub_str_parameter(v)
            elif isinstance(v, dict) or isinstance(v, list):
                config_sub_recursive_dict(v)
    else:
        raise Exception(f"Error in configuration, type {type(cfg_rec)} is not supported with subsitutions.")


import datetime


def timestamp_str():
    tz = datetime.timezone.utc
    ft = "%m_%dT%H_%M"
    t = datetime.datetime.now(tz=tz).strftime(ft)
    return t

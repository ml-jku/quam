# [ x ] MoG
# [   ] Random walk dataset
# [ x ] Two moon
# [   ] Iris
# etc

import torch
from torch.utils.data import Dataset as tDataset

from sklearn.datasets import make_moons

import numpy as np
from numpy.random import RandomState

from typing import Union


class TwoMoonDataset(tDataset):
    def __init__(self,
                 n_samples,
                 noise=None,
                 random_state=None):
        super(TwoMoonDataset, self).__init__()
        self.X, self.y = make_moons(n_samples, noise=noise, random_state=random_state)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item, :], self.y[item]
        # return torch.from_numpy(self.X[item, :]), torch.from_numpy(self.y[item])

    def get_raw_data(self):
        return self.X, self.y


def get_gmm_sample(
        random: Union[int, RandomState],
        n_samples: int,
        means: np.ndarray,


        covars: np.ndarray,
        weights: np.ndarray,
):
    '''

    :param random:
        random state for reproducibility
    :param n_samples:
        number of samples to draw
    :param means:
        [n_gaussians, n_dimensions]
    :param covars:
        [n_gaussians, n_dimension, n_dimensions]
    :param weights:
        array/list with weights for the distribution
    :return:
        samples from the gaussians [n_samples, n_dimensions], labels of samples [n_samples,]
    '''
    assert means.shape[0] == covars.shape[0] and means.shape[1] == covars.shape[1] and covars.shape[1] == covars.shape[2], "Matrix sizes do not add up"

    if isinstance(random, int):
        random = RandomState(random)

    n_dim = means.shape[1]
    n_gaussians = means.shape[0]

    sample_dist = random.choice(
        np.arange(0, n_gaussians),
        size=n_samples,
        p=weights)
    sample = ((random.randn(n_samples, n_dim) @ covars)*np.eye(n_gaussians)[sample_dist].T[:,:, None]).sum(0) + means[sample_dist, :]

    return sample, sample_dist


class GeneratedGMMData(tDataset):
    # INFO: The old code has a few more functions to compute pdf, base aleatoric uncertainty, etc.
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.data, self.labels = get_gmm_sample(*args, **kwargs)
        # save the parameters
        self.means = args[2]
        self.covars = args[3]
        self.weights = args[4]

        # self.noise_probability = noise_probability
        # self.noise_range = noise_range
        self.n_classes = len(self.means)
        self.true_data_len = self.data.shape[0]

    def __len__(self):
        return self.true_data_len

    def __getitem__(self, item):
        return self.data[item, :], self.labels[item]

    def get_raw_data(self):
        return self.data, self.labels

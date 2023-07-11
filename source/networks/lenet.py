import os

import torch
import torch.nn as nn

import copy


def get_lenet(out_features: int = 10, p_dropout: float = 0.2, mnist: bool = True, checkpoint: str = None):
    net = LeNet(out_features, p_dropout, mnist)
    if checkpoint is not None:
        net.load_state_dict(torch.load(os.path.abspath(checkpoint)))
    return net


class LeNet(nn.Module):

    def __init__(self, out_features: int, p_drop: float = 0.2, mnist=False):
        super(LeNet, self).__init__()

        self.first_layer_fm = None
        self.second_layer_fm = None

        activation = nn.ReLU()

        dim = 4 if mnist else 5

        self.first_layer = nn.Conv2d(in_channels=1 if mnist else 3, out_channels=6, kernel_size=5)

        self.second_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            activation,
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5))

        self.projection = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            activation,
            nn.Flatten(),
            nn.Linear(in_features=16*dim**2, out_features=120),
            activation,
            nn.Linear(in_features=120, out_features=84),
            activation,
            nn.Dropout(p=p_drop)
        )

        self.out = nn.Linear(in_features=84, out_features=out_features)

    def project(self, x, store_fm=False):
        x = self.first_layer(x)
        if store_fm:
            self.first_layer_fm = copy.deepcopy(x.detach())
        x = self.second_layer(x)
        if store_fm:
            self.second_layer_fm = copy.deepcopy(x.detach())
        return self.projection(x)

    def forward(self, x, store_fm=False):
        return self.out(self.project(x, store_fm))

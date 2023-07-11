import os
import torch
import torch.nn as nn
import torchvision.models as models


def get_efficientnet(version: str = "s", out_features: int = 1000, p_dropout: float = 0.2, init_last_layer = False, checkpoint: str = None, use_projections: bool = False, use_bfloat16: bool = True):
    net = EfficientNet(version, out_features, p_dropout, init_last_layer, use_projections, use_bfloat16)
    if checkpoint is not None:
        net.load_state_dict(torch.load(os.path.abspath(checkpoint)))
    return net


def get_efficientnet_ll(p_drop: float = 0.2, n_projected_features: int = 1280, n_out_features: int = 1000, checkpoint: str = None, use_bfloat = True, unpack_linear = False):
    out_layer = nn.Sequential(nn.Dropout(p=p_drop), nn.Linear(n_projected_features, n_out_features))
    if checkpoint is not None:
        if checkpoint == "stock":
            network = models.efficientnet_v2_s(weights='DEFAULT')
            out_layer.load_state_dict(network.classifier.state_dict())
            print("loading torch default last layer for model size s")
        else:
            out_layer.load_state_dict(torch.load(os.path.abspath(checkpoint)))
    if use_bfloat:
        out_layer = out_layer.to(dtype=torch.bfloat16)
    else:
        out_layer = out_layer.to(dtype=torch.float32)

    if unpack_linear:
        out_layer = out_layer[-1]

    return out_layer


class EfficientNet(nn.Module):
    """
    PyTorch: https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b4.html#torchvision.models.EfficientNet_B4_Weights
    """
    def __init__(self, version, out_features: int = 1000, p_drop: float = 0.2, init_last_layer: bool = False, use_projections: bool = False, use_bfloat16: bool = True):
        super(EfficientNet, self).__init__()

        if version.lower() == "small" or version.lower() == "s":
            network = models.efficientnet_v2_s(weights='DEFAULT')
        elif version.lower() == "medium" or version.lower() == "m":
            network = models.efficientnet_v2_m(weights='DEFAULT')
        elif version.lower() == "large" or version.lower() == "l":
            network = models.efficientnet_v2_l(weights='DEFAULT')
        elif version.lower() == "b0":
            network = models.efficientnet_b0(weights='DEFAULT')
        elif version.lower() == "b1":
            network = models.efficientnet_b1(weights='DEFAULT')
        elif version.lower() == "b2":
            network = models.efficientnet_b2(weights='DEFAULT')
        elif version.lower() == "b3":
            network = models.efficientnet_b3(weights='DEFAULT')
        elif version.lower() == "b4":
            network = models.efficientnet_b4(weights='DEFAULT')
        elif version.lower() == "b5":
            network = models.efficientnet_b5(weights='DEFAULT')
        elif version.lower() == "b6":
            network = models.efficientnet_b6(weights='DEFAULT')
        elif version.lower() == "b7":
            network = models.efficientnet_b7(weights='DEFAULT')
        else:
            raise ValueError(f"EfficientNet version '{version}' not defined!")

        self.use_projections = use_projections

        if init_last_layer:
            self.out = nn.Sequential(nn.Dropout(p=p_drop, inplace=True),
                                     nn.Linear(network.classifier[1].in_features, out_features))
        else:
            assert network.classifier[1].out_features == out_features, f"{network.classifier[1].in_features} out_features of pre-trained network don't match specified {out_features} out_features"
            self.out = network.classifier
            if use_bfloat16:
                self.out = self.out.to(dtype=torch.bfloat16)

        if not self.use_projections:
            self.projection = nn.Sequential(*list(network.children())[:-1])
            if use_bfloat16:
                self.projection = self.projection.to(dtype=torch.bfloat16)
        else:
            del network


    def project(self, x):
        if not self.use_projections:
            return self.projection(x)
        else:
            raise ValueError("when 'use_projection' is set to True, 'projection' function should not be called")

    def forward(self, x):

        if not self.use_projections:
            return self.out(self.project(x).squeeze())
        else: 
            return self.out(x)


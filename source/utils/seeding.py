import os
import torch
import hamiltorch
from typing import Iterable


def fix_seeds(seed: int = 42):
    # for reproducibility on GPU with cudatoolkit >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if not torch.are_deterministic_algorithms_enabled():
        torch.use_deterministic_algorithms(True)

    hamiltorch.set_random_seed(seed)

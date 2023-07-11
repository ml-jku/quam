import torch

import numpy as np

from typing import Union

from torchvision import transforms
from torchvision.transforms import functional as transfun


@torch.no_grad()
def apply_image_augmentations(
    image: torch.Tensor,    # [N, C, X, Y]
    rng: Union[int, np.random.RandomState],
    n_samples: int,
    max_rotation_angle = 1.,
    flip_horizontal = 0.5,
    crop = 3,
):
    if isinstance(rng, int):
        rng = np.random.RandomState(seed=rng)

    images = image.repeat(n_samples, 1, 1, 1).cpu()

    # first image is left unchanged
    for i in range(1, n_samples):
        if max_rotation_angle is not None:
            angle = (rng.random()*2 - 1)*max_rotation_angle
            images[i] = transfun.rotate(images[i], angle, transforms.InterpolationMode.BILINEAR, center=None, expand=False, fill=0)
        if rng.random() < flip_horizontal:
            images[i] = transfun.hflip(images[i])
        if crop:
            # all crops are central
            x,y = rng.randint(1, crop + 1, size=(2,), dtype=int).tolist()
            max_size = min(images.shape[2]-x, images.shape[3]-y)
            w = images.shape[2]-x
            h = images.shape[3]-y
            images[i] = transfun.resized_crop(images[i], y, x, h, w, list(images.shape[2:]))

    return images

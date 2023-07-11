import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from torch.utils.data import DataLoader
from source.data import get_dataset

if __name__ == "__main__":

    print(os.path.join(os.path.dirname(__file__), ".."))

    network_names = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "s", "m"]

    ds_ids = ["imagenet", "imagenet-o", "imagenet-a"]

    for idx in ds_ids:
        for n in network_names:
            dataset = get_dataset(id=idx, 
                                  subset="test", 
                                  properties= {'use_projections': True, 'efficientnet_version': n, 'use_train_ds_subset': None} if idx == "imagenet" else {'use_projections': True, 'efficientnet_version': n})

            dataloader = DataLoader(dataset=dataset,
                                    batch_size=2,
                                    num_workers=1)
            for x, y in dataloader:
                assert x.shape == [2, 1280]
                assert y.shape == [2]
                break

import torch
import numpy as np

from torch.utils.data import Dataset as tDataset
from torch.utils.data import DataLoader as tDataLoader
from torch.utils.data import Subset as tSubset

from collections.abc import Iterable, Mapping

from typing import Callable, Union, List, Dict, Tuple

from ..networks.batched_modules import feed_batch_model_image_multiple

from tqdm import tqdm


def wrap_encode(ds: tDataset, **kwargs):
    '''Add the encoded wrapper, where all the inputs are encoded by a model first'''
    raise NotImplementedError


def wrap_device(dl: tDataLoader, device: str):
    '''Cache the content of a dataloader on a device'''
    wrapped = CachedDataLoader(dl, device=device)
    return wrapped


def wrap_subset(ds: tDataset, subset_str: str):
    '''Only provide a subset of the dataset'''
    # parse the simple range string
    range_start, range_end = [int(s) for s in subset_str.split(':')]
    wrapped = tSubset(ds, list(range(range_start, range_end)))
    return wrapped


def wrap_indexify(ds: tDataset):
    '''Append index as the last element of the return value to keep better track of values produced.'''
    return IndexedDatasetWrapper(ds)


def wrap_looping(dl: tDataLoader, max_batches: int = -1):
    # leaving default max batches would make it yield one batch!
    return InfiniteDataLoaderWrapper(dl, max_n=max_batches)


def wrap_consistent_shuffle(ds: tDataset, rng_seed: int = 42):
    return ConsistentlyShuffledDataset(ds, rng_seed=rng_seed)


wrapper_registry = {
    "wrap_encode": wrap_encode,
    "wrap_index": wrap_indexify,
    "wrap_subset": wrap_subset,
    "wrap_device": wrap_device,
    "wrap_shuffle": wrap_consistent_shuffle,
    "wrap_looping": wrap_looping,
}


def wrap_dataset_or_loader(ds: Union[tDataset, tDataLoader], ds_wrappers: List[Dict]):
    for wdict in ds_wrappers:
        # wdict would normally contain one item
        for wrapper, wrapper_params in wdict.items():
            wrapper_fn = wrapper_registry[wrapper]
            # print(wrapper_fn, wrapper_params)
            ds = wrapper_fn(ds, **wrapper_params)
    return ds


class ConsistentlyShuffledDataset(tDataset):
    def __init__(self, original_data: tDataset, rng_seed: int = 42):
        super(ConsistentlyShuffledDataset, self).__init__()

        self.unwrapped = original_data

        # generate the dataset index as per the seed
        rng_state = np.random.RandomState(seed=rng_seed)
        self.indices = rng_state.choice(np.arange(0, len(original_data), dtype=np.uint64), size=len(self.unwrapped), replace=False) # sample indices without replacement into an array

    def __getitem__(self, item):
        return self.unwrapped[self.indices[item].item()]

    def __len__(self):
        return len(self.unwrapped)


class EncodedDatasetWrapper(tDataset):
    def __init__(self, original_data: tDataset):
        super(EncodedDatasetWrapper, self).__init__()

        self.unwrapped = original_data
        self.transformed_data = None
        # if the encoder is a tuple, the 'encoder' lambda should be aware of that
        self.encoder = None

    def encode_data(
        self,
        encoder: Callable
    ):
        self.transformed_data = [
            encoder(d) for d in self.unwrapped
        ]

    def encode_data_lazy(
        self,
        encoder: Callable
    ):
        self.encoder = encoder

    def __getattr__(self, item):
        if self.transformed_data:
            return self.transformed_data[item]
        elif self.encoder_net:
            return self.encoder(self.unwrapped[item])
        else:
            raise AttributeError('The data is neither statically transformed nor a transformation function is specified.')

    def __len__(self):
        if self.transformed_data:
            return len(self.transformed_data)
        elif self.encoder_net:
            return len(self.unwrapped)
        else:
            raise AttributeError(
                'The data is neither statically transformed nor a transformation function is specified.')


class CachingDatasetWrapper(tDataset):
    def __init__(self, original_data: tDataset, device = None, dtype = None, max_cache_size = -1, verbose=False):
        super(CachingDatasetWrapper, self).__init__()

        self.unwrapped = original_data
        self.cache = {}
        # self.overflow_list = []
        # self.max_size = max_cache_size
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self.unwrapped)

    def __getitem__(self, item):
        if item not in self.cache.keys():
            # fetch the entry, convert to the required dtype and store on the required device
            dentry = self.unwrapped[item]
            if isinstance(dentry, torch.Tensor):
                self.cache[item] = dentry.to(device=self.device, dtype=self.dtype)
            elif isinstance(dentry, Mapping):
                raise NotImplementedError()
            elif isinstance(dentry, Iterable):
                self.cache[item] = [e.to(device=self.device, dtype=self.dtype) if isinstance(e, torch.Tensor) else e for e in dentry]
            else:
                raise NotImplementedError()
        return self.cache[item]


class CachedDataLoader(tDataLoader):
    def __init__(self, dl: tDataLoader, device = None, dtype = None, verbose=False):
        self.unwrapped = dl
        self.device = device
        self.dtype = dtype
        self.cache = [[t.to(device=self.device, dtype=self.dtype) if t.dtype.is_floating_point else t.to(device=self.device) for t in e] if isinstance(e, Iterable) else e for e in (tqdm(self.unwrapped) if verbose else self.unwrapped)]

    def __iter__(self):
        return iter(self.cache)

    def __getattr__(self, item):
        return self.unwrapped.__getattribute__(item)


class EnsembleImageDataLoader(tDataLoader):
    def __init__(
        self,
        dl: tDataLoader,
        ensemble_model_count: int,
        convertion_fns: list = [feed_batch_model_image_multiple, lambda a, n: a],
    ):
        self.unwrapped = dl
        self.convertion_fns = convertion_fns
        self.model_n = ensemble_model_count
        self.iter = None

    def __iter__(self):
        self.iter = iter(self.unwrapped)
        return self

    def __next__(self):
        try:
            d = next(self.iter)
        except StopIteration:
            raise

        return (fn(subd, self.model_n) for subd, fn in zip(d, self.convertion_fns))

    def __getattr__(self, item):
        return self.unwrapped.__getattribute__(item)


class InfiniteDataLoaderWrapper(tDataLoader):
    '''
    DO NOT STACK BEFORE CACHING WRAPPER!
    '''
    def __init__(self, source_dl: tDataLoader, max_n = -1):
        self.dl = source_dl
        self.max_n = max_n

    def __iter__(self):
        return self.gen()

    def __len__(self):
        return self.max_n

    def gen(self):
        iters_made = 0
        while True:
            for batch in self.dl:
                if iters_made < self.max_n:
                    yield [b for b in batch]
                    iters_made += 1
                else:
                    return


class IndexedDatasetWrapper(tDataset):
    def __init__(self, original_data: tDataset):
        self.unwrapped = original_data

    def __len__(self):
        return len(self.unwrapped)

    def __getitem__(self, item):
        # get the item from the parent dataset
        dpoint = self.unwrapped[item]

        if isinstance(dpoint, List) or isinstance(dpoint, Tuple):
            return list(dpoint)+[item,]
        elif isinstance(dpoint, Dict):
            dpoint['ds_index'] = item
            return dpoint
        else:
            raise NotImplementedError("only datasets providing dictionaries or lists/tuples are supported")

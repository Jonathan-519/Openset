# Function to get different types of data loader instances.

from .basedata import BaseDataLoader
from .hierdata import HierDataLoader
from .taxosafe_data import TaxoSafeDataLoader
from .utils import get_prompt_template


def get_dataloader(cfg, splits, batch_size):
    loader = _get_loader_instance(cfg["loader"])
    return loader(cfg, splits, batch_size)


def _get_loader_instance(name):
    loaders = {
        "BaseDataLoader": BaseDataLoader,
        "HierDataLoader": HierDataLoader,
        "TaxoSafeDataLoader": TaxoSafeDataLoader,
    }
    if name not in loaders:
        raise ValueError("Loader type {} not available".format(name))
    return loaders[name]

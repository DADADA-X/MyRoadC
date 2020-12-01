import torch
import random
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

import data.datautils as util


class BaseDataset(Dataset):
    """
    Base class for all datasets.
    """

    def __init__(self, seed):
        """Initialize file paths or a list of file names. """
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def __len__(self):
        """Return the total size of the dataset."""
        return 0

    def __getitem__(self, item):
        """Return a data pair."""
        pass
from itertools import chain
import torch
import numpy as np
from torch.utils.data import Dataset

###############################################
#              map style dataset              #
###############################################
class CustomizeDataset(Dataset):

    def __init__(self, data: dict, shuffle=True, processGDT=True):
        self.data = {}
        gdt, noisy = data["gdt"], data["noisy"]

        # noisy, nmean, nstd = CustomizeDataset.standardize(noisy)
        noisy, nmin, nmax = CustomizeDataset.normalize(noisy)
        if processGDT:
            # gdt, _, _ = CustomizeDataset.standardize(gdt, nmean, nstd)
            gdt, _, _ = CustomizeDataset.normalize(gdt)

    # shuffle to mix signal present and signal absent
        if shuffle:
            index = np.arange(gdt.shape[0])
            np.random.shuffle(index)
            gdt = gdt[index, ...]
            noisy = noisy[index, ...]

        self.data["noisy"] = noisy
        self.data["gdt"] = gdt


    def __len__(self):
        """Return number of data"""
        return self.data["gdt"].shape[0]


    def __getitem__(self, idx):
        return torch.stack([torch.from_numpy(self.data["gdt"][idx, ...]).float(),
                            torch.from_numpy(self.data["noisy"][idx, ...]).float()], dim=0)


    @staticmethod
    def standardize(x, m=None, s=None):
        """
        standardize x in shape of (N, H, W) to mean=0, std=1
        """
        if m is None:
            m = np.mean(x, axis=(1, 2), keepdims=True)

        if s is None:
            s = np.std(x, axis=(1, 2), keepdims=True)

        return (x - m) / s, m, s



    @staticmethod
    def normalize(x, min=None, max=None):
        """
        normalize x in shape of (N, H, W) to min=0, max=1
        """
        if min is None:
            min = np.min(x, axis=(1, 2), keepdims=True)
        if max is None:
            max = np.max(x, axis=(1, 2), keepdims=True)
        return (x - min) / (max - min), min, max

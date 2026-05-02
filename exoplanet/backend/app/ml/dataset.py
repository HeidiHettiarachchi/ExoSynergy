import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

ALL_GASES = [
    "H2O", "CO2", "CH4", "CO", "NH3",
    "H2",  "He",  "N2",  "O2", "O3", "SO2", "H2S"
]


class ABCDataset(Dataset):
    def __init__(self, file_path, stats=None):
        with h5py.File(file_path, "r") as f:
            X_data = np.array(f['X'], dtype=np.float32)
            y_data = np.array(f['y'], dtype=np.float32)

        # PROCESSING (LOG SPACE)
        y_data = np.clip(y_data, 1e-12, None)
        y_data = np.log10(y_data)

        # FEATURE NORMALIZATION
        if stats is None:
            self.mean = X_data.mean(axis=0, keepdims=True)
            self.std = X_data.std(axis=0, keepdims=True) + 1e-8
        else:
            self.mean = stats['mean']
            self.std = stats['std']

        X_data = (X_data - self.mean) / self.std

        # CLIP EXTREME VALUES
        X_data = np.clip(X_data, -5, 5)

        # CONVERT TO TENSORS
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)

    def get_stats(self):
        return {
            'mean': self.mean,
            'std': self.std
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
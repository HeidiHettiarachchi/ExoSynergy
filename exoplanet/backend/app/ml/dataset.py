import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

ALL_GASES = [
    "H2O", "CO2", "CH4", "CO", "NH3",
    "H2",  "He",  "N2",  "O2", "O3", "SO2", "H2S"
]


class ABCDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, "r") as f:
            X_data = np.array(f['X'], dtype=np.float32)
            y_data = np.array(f['y'], dtype=np.float32)

        # Sanity check labels
        y_data = np.clip(y_data, 0, None)           # no negatives
        row_sums = y_data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        y_data = y_data / row_sums * 100.0           # ensure every row sums to 100%

        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
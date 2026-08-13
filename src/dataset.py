"""PyTorch ``Dataset`` wrappers around the preprocessed NumPy arrays."""

import numpy as np
import torch
from torch.utils.data import Dataset


class NIDSDataset(Dataset):
    """Wrap a feature matrix ``X`` and target vector ``y`` as tensors.

    Args:
        X: float32 feature matrix, shape ``(n_samples, n_features)``.
        y: targets. For the multi-class model pass int64 class indices; for the
            binary model pass float32 0/1 labels.
        binary: if True, targets are reshaped to ``(n, 1)`` float tensors so
            they line up with a single-logit BCE output.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, binary: bool = False) -> None:
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        if binary:
            self.y = torch.from_numpy(np.asarray(y, dtype=np.float32)).view(-1, 1)
        else:
            self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

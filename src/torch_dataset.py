"""PyTorch dataset utilities for the learned cost-map model."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CostMapDataset(Dataset):
    """
    PyTorch dataset backed by a generated .npz dataset split.

    Each sample contains:
        input:  (5, H, W)
        target: (1, H, W)
    """

    def __init__(
        self,
        npz_path: str | Path,
    ) -> None:
        npz_path = Path(npz_path)

        if not npz_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {npz_path}"
            )

        with np.load(
            npz_path,
            allow_pickle=False,
        ) as archive:
            inputs = archive["inputs"]
            soft_labels = archive["soft_labels"]

        if inputs.ndim != 4:
            raise ValueError(
                "inputs must have shape (N, C, H, W)."
            )

        if soft_labels.ndim != 4:
            raise ValueError(
                "soft_labels must have shape (N, 1, H, W)."
            )

        if inputs.shape[0] != soft_labels.shape[0]:
            raise ValueError(
                "inputs and soft_labels must contain "
                "the same number of samples."
            )

        self.inputs = torch.from_numpy(
            inputs.astype(np.float32)
        )

        self.targets = torch.from_numpy(
            soft_labels.astype(np.float32)
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.inputs.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return one input-target pair.
        """
        return (
            self.inputs[index],
            self.targets[index],
        )
"""Causal tactile windows and aligned dynamic pretraining targets."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from encoder.clean_v2 import decode


def causal_history_indices(
    anchor: int, sequence_length: int, temporal_stride: int
) -> tuple[int, ...]:
    """Return an ordered causal window, repeating frame zero when needed."""
    if anchor < 0:
        raise ValueError("anchor must be non-negative")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if temporal_stride < 1:
        raise ValueError("temporal_stride must be positive")
    return tuple(
        max(0, anchor - offset * temporal_stride)
        for offset in range(sequence_length - 1, -1, -1)
    )


def marker_displacement(marker_pair: np.ndarray) -> np.ndarray:
    marker_pair = np.asarray(marker_pair, dtype=np.float32)
    if marker_pair.shape != (2, 1200, 2):
        raise ValueError(f"invalid marker pair shape: {marker_pair.shape}")
    return marker_pair[1] - marker_pair[0]


def dynamic_marker_delta(marker: h5py.Dataset, anchor: int, temporal_stride: int):
    previous = max(0, anchor - temporal_stride)
    return marker_displacement(marker[anchor]) - marker_displacement(marker[previous])


def _normalize(raw: np.ndarray, statistics: dict, target_name: str) -> torch.Tensor:
    value = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite {target_name} target")
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.asarray(statistics["std"], dtype=np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(f"Invalid {target_name} normalization")
    return torch.from_numpy((value - mean) / std)


class DynamicCleanDataset(Dataset):
    """Load one causal image window and current-frame supervision per sample."""

    def __init__(
        self,
        root,
        files,
        normalization,
        image_size=(256, 256),
        stride=1,
        sequence_length=4,
        temporal_stride=1,
    ):
        self.root = Path(root)
        self.normalization = normalization
        self.image_size = tuple(image_size)
        self.sequence_length = sequence_length
        self.temporal_stride = temporal_stride
        self.index = []
        causal_history_indices(0, sequence_length, temporal_stride)
        if stride < 1:
            raise ValueError("stride must be positive")
        required_statistics = {"depth", "marker", "delta_marker"}
        if set(normalization) < required_statistics:
            raise ValueError(
                f"normalization missing {sorted(required_statistics - set(normalization))}"
            )
        for name in files:
            with h5py.File(self.root / name, "r") as handle:
                for side in ("left", "right"):
                    key = f"tactile/{side}_gsmini"
                    sensor = handle[key]
                    frame_count = len(sensor["rgb_marker"])
                    if len(sensor["depth"]) != frame_count or len(sensor["marker"]) != frame_count:
                        raise ValueError(f"Length mismatch {name}/{key}")
                    if sensor["depth"].shape[1:] != (240, 320) and sensor["depth"].shape[1:] != self.image_size:
                        raise ValueError(f"Invalid depth shape {name}/{key}")
                    if sensor["marker"].shape[1:] != (2, 1200, 2):
                        raise ValueError(f"Invalid marker shape {name}/{key}")
                    self.index.extend(
                        (name, key, anchor)
                        for anchor in range(0, frame_count, stride)
                    )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        name, key, anchor = self.index[item]
        history = causal_history_indices(
            anchor, self.sequence_length, self.temporal_stride
        )
        if max(history) != anchor:
            raise RuntimeError("causal history must end at its anchor")
        with h5py.File(self.root / name, "r") as handle:
            sensor = handle[key]
            images = torch.stack(
                [
                    TF.resize(
                        TF.to_tensor(decode(sensor["rgb_marker"][frame])),
                        self.image_size,
                        antialias=True,
                    )
                    for frame in history
                ]
            )
            depth = np.asarray(sensor["depth"][anchor], dtype=np.float32)[None]
            marker = marker_displacement(sensor["marker"][anchor])
            delta_marker = dynamic_marker_delta(
                sensor["marker"], anchor, self.temporal_stride
            )
        if not torch.isfinite(images).all():
            raise ValueError(f"Nonfinite image {name}/{key}/{anchor}")
        targets = {
            "depth": _normalize(depth, self.normalization["depth"], "depth"),
            "marker": _normalize(marker, self.normalization["marker"], "marker"),
            "delta_marker": _normalize(
                delta_marker,
                self.normalization["delta_marker"],
                "delta_marker",
            ),
        }
        return images, targets


__all__ = [
    "DynamicCleanDataset",
    "causal_history_indices",
    "dynamic_marker_delta",
    "marker_displacement",
]

"""Audit Dynamic Details V2 clean data and compute train-only normalization."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from encoder.clean_v2 import PREPROCESS, decode, manifest_hash
from encoder.dynamic_data import dynamic_marker_delta, marker_displacement


def _accumulate(accumulator, values):
    values = np.asarray(values, dtype=np.float64)
    accumulator[0] += values.size
    accumulator[1] += values.sum()
    accumulator[2] += np.square(values).sum()


def _all_finite(dataset, chunk_size=64):
    return all(
        np.isfinite(dataset[start : start + chunk_size]).all()
        for start in range(0, len(dataset), chunk_size)
    )


def _normalization(accumulator, statistics_frame_stride):
    count, total, squared_total = accumulator
    if not count:
        raise ValueError("normalization received no train samples")
    mean = total / count
    variance = max(squared_total / count - mean * mean, 0.0)
    return {
        "mean": mean,
        "std": max(float(np.sqrt(variance)), 1e-6),
        "count": count,
        "statistics_frame_stride": statistics_frame_stride,
        "source": "train_only",
    }


def audit_dynamic_dataset(
    root,
    manifest,
    *,
    sequence_length=4,
    temporal_stride=1,
    statistics_frame_stride=10,
):
    root = Path(root)
    if sequence_length < 1 or temporal_stride < 1:
        raise ValueError("history parameters must be positive")
    if statistics_frame_stride < 1:
        raise ValueError("statistics_frame_stride must be positive")
    if not manifest.get("train") or not manifest.get("val"):
        raise ValueError("manifest must contain non-empty train and val trajectories")
    if set(manifest["train"]) & set(manifest["val"]):
        raise ValueError("train and val trajectories must be disjoint")

    accumulators = {name: [0, 0.0, 0.0] for name in ("depth", "marker", "delta_marker")}
    trajectory_audit = []
    for split_name in ("train", "val"):
        for name in manifest[split_name]:
            with h5py.File(root / name, "r") as handle:
                row = {"file": name, "split": split_name, "sensors": {}}
                for side in ("left", "right"):
                    key = f"tactile/{side}_gsmini"
                    sensor = handle[key]
                    frame_count = len(sensor["rgb_marker"])
                    if frame_count == 0:
                        raise ValueError(f"Empty trajectory {name}/{key}")
                    depth = sensor["depth"]
                    marker_pairs = sensor["marker"]
                    if len(depth) != frame_count or len(marker_pairs) != frame_count:
                        raise ValueError(f"Length mismatch {name}/{key}")
                    if depth.ndim != 3:
                        raise ValueError(f"Invalid depth shape {name}/{key}: {depth.shape}")
                    if marker_pairs.shape[1:] != (2, 1200, 2):
                        raise ValueError(f"Invalid marker shape {name}/{key}: {marker_pairs.shape}")
                    if not _all_finite(depth) or not _all_finite(marker_pairs):
                        raise ValueError(f"Nonfinite target in {split_name} {name}/{key}")
                    first_image = decode(sensor["rgb_marker"][0])
                    last_image = decode(sensor["rgb_marker"][-1])
                    if first_image.ndim != 3 or last_image.shape != first_image.shape:
                        raise ValueError(f"Invalid image shape {name}/{key}")
                    if tuple(depth.shape[1:]) != tuple(first_image.shape[:2]):
                        raise ValueError(f"Depth/image shape mismatch {name}/{key}")

                    sampled = range(0, frame_count, statistics_frame_stride)
                    sampled = tuple(sampled)
                    displacement = np.stack(
                        [marker_displacement(marker_pairs[anchor]) for anchor in sampled]
                    )
                    delta = np.stack(
                        [
                            dynamic_marker_delta(
                                sensor["marker"], anchor, temporal_stride
                            )
                            for anchor in sampled
                        ]
                    )
                    row["sensors"][side] = {
                        "frames": frame_count,
                        "sampled_frames": len(sampled),
                        "image_shape": list(first_image.shape),
                        "finite": True,
                    }
                    if split_name == "train":
                        _accumulate(accumulators["depth"], depth[::statistics_frame_stride])
                        _accumulate(accumulators["marker"], displacement)
                        _accumulate(accumulators["delta_marker"], delta)
                trajectory_audit.append(row)

    normalization = {
        name: _normalization(values, statistics_frame_stride)
        for name, values in accumulators.items()
    }
    return {
        "split_hash": manifest_hash(manifest),
        "history": {
            "sequence_length": sequence_length,
            "temporal_stride": temporal_stride,
        },
        "inspected_splits": {
            "train": len(manifest["train"]),
            "val": len(manifest["val"]),
        },
        "normalization": normalization,
        "preprocess": PREPROCESS,
        "trajectory_audit": trajectory_audit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--temporal-stride", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads(Path(args.split).read_text())
    report = audit_dynamic_dataset(
        args.clean,
        manifest,
        sequence_length=args.sequence_length,
        temporal_stride=args.temporal_stride,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "trajectory_audit"}, indent=2))


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "encoder"))

from dataloader import HDF5Dataset, discover_hdf5_paths  # noqa: E402


def check_dataset(data_root):
    paths = discover_hdf5_paths("gsmini", data_root)
    dataset = HDF5Dataset(paths[:1], schema="gsmini")
    sample = dataset[0]
    expected_shapes = {
        "rgb": (3, 256, 256),
        "marked_rgb": (3, 256, 256),
        "depth": (1, 256, 256),
        "marker": (63, 2),
        "pose": (7,),
    }
    actual_shapes = {key: tuple(value.shape) for key, value in sample.items()}
    assert actual_shapes == expected_shapes, actual_shapes
    assert all(torch.isfinite(value).all().item() for value in sample.values())
    print(
        f"gsmini/{Path(data_root).parent.name}: files={len(paths)}, "
        f"one_file_samples={len(dataset)}, shapes={actual_shapes}"
    )


def main():
    check_dataset(PROJECT_ROOT / "data/insert_HDMI/clean")
    check_dataset(PROJECT_ROOT / "data/lift_bottle/clean")
    try:
        discover_hdf5_paths("legacy_contact_gs", PROJECT_ROOT / "data/contact-gs")
    except FileNotFoundError as exc:
        print("legacy_missing_data_error:", exc)
    else:
        raise AssertionError("legacy missing-data guard did not trigger")
    print("DATA_ADAPTER_SMOKE_PASSED")


if __name__ == "__main__":
    main()

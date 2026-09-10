import cv2
import h5py
import numpy as np
import pytest
import torch


def _write_trajectory(path, *, frames=21, displacement_scale=1.0, depth_offset=0.0):
    with h5py.File(path, "w") as handle:
        for side_index, side in enumerate(("left", "right")):
            sensor = handle.create_group(f"tactile/{side}_gsmini")
            images = []
            marker = np.empty((frames, 2, 1200, 2), dtype=np.float32)
            depth = np.empty((frames, 24, 32), dtype=np.float32)
            for frame in range(frames):
                pixels = np.full(
                    (24, 32, 3), side_index * 80 + frame * 5, dtype=np.uint8
                )
                ok, encoded = cv2.imencode(".png", pixels)
                assert ok
                images.append(encoded.tobytes())
                displacement = np.full(
                    (1200, 2), displacement_scale * frame, dtype=np.float32
                )
                # Reference slots intentionally move across frames. This catches
                # cross-frame plane subtraction instead of true frame-local
                # displacement differencing.
                reference = np.full(
                    (1200, 2), side_index * 100.0 + frame * 7.0, dtype=np.float32
                )
                marker[frame, 0] = reference
                marker[frame, 1] = reference + displacement
                depth[frame] = depth_offset + frame
            max_length = max(map(len, images))
            sensor.create_dataset(
                "rgb_marker", data=np.asarray(images, dtype=f"S{max_length}")
            )
            sensor.create_dataset("marker", data=marker)
            sensor.create_dataset("depth", data=depth)


def test_causal_indices_repeat_first_and_apply_stride():
    from encoder.dynamic_data import causal_history_indices

    assert causal_history_indices(0, 4, 1) == (0, 0, 0, 0)
    assert causal_history_indices(1, 4, 1) == (0, 0, 0, 1)
    assert causal_history_indices(3, 4, 1) == (0, 1, 2, 3)
    assert causal_history_indices(3, 4, 2) == (0, 0, 1, 3)
    assert max(causal_history_indices(17, 4, 3)) == 17
    with pytest.raises(ValueError):
        causal_history_indices(-1, 4, 1)
    with pytest.raises(ValueError):
        causal_history_indices(1, 0, 1)
    with pytest.raises(ValueError):
        causal_history_indices(1, 4, 0)


def test_dataset_rejects_missing_normalization_even_with_unrelated_extra_key(tmp_path):
    from encoder.dynamic_data import DynamicCleanDataset

    incomplete = {
        "depth": {"mean": 0.0, "std": 1.0},
        "marker": {"mean": 0.0, "std": 1.0},
        "rgb": {"mean": 0.0, "std": 1.0},
    }
    with pytest.raises(ValueError, match="delta_marker"):
        DynamicCleanDataset(tmp_path, [], incomplete)


def test_dataset_keeps_history_within_trajectory_and_side_and_aligns_targets(tmp_path):
    from encoder.dynamic_data import DynamicCleanDataset

    _write_trajectory(tmp_path / "train.hdf5", frames=5)
    normalization = {
        "depth": {"mean": 0.0, "std": 1.0},
        "marker": {"mean": 0.0, "std": 1.0},
        "delta_marker": {"mean": 0.0, "std": 1.0},
    }
    dataset = DynamicCleanDataset(
        tmp_path,
        ["train.hdf5"],
        normalization,
        image_size=(24, 32),
        sequence_length=4,
        temporal_stride=1,
    )

    # The fourth left-side sample is anchored at t=3. Its pixel values identify
    # the exact causal frame sequence without sharing state across sensors/files.
    images, targets = dataset[3]
    assert dataset.index[3] == ("train.hdf5", "tactile/left_gsmini", 3)
    assert images.shape == (4, 3, 24, 32)
    torch.testing.assert_close(
        images[:, 0, 0, 0], torch.tensor([0.0, 5.0, 10.0, 15.0]) / 255.0
    )
    assert set(targets) == {"depth", "marker", "delta_marker"}
    torch.testing.assert_close(targets["depth"], torch.full((1, 24, 32), 3.0))
    torch.testing.assert_close(targets["marker"], torch.full((1200, 2), 3.0))
    torch.testing.assert_close(
        targets["delta_marker"], torch.ones((1200, 2))
    )

    padded_images, padded_targets = dataset[1]
    torch.testing.assert_close(
        padded_images[:, 0, 0, 0], torch.tensor([0.0, 0.0, 0.0, 5.0]) / 255.0
    )
    torch.testing.assert_close(
        padded_targets["delta_marker"], torch.ones((1200, 2))
    )
    _, first_targets = dataset[0]
    torch.testing.assert_close(
        first_targets["delta_marker"], torch.zeros((1200, 2))
    )

    # First right-side sample must start a fresh history and cannot reuse left t=4.
    right_images, _ = dataset[5]
    assert dataset.index[5] == ("train.hdf5", "tactile/right_gsmini", 0)
    torch.testing.assert_close(
        right_images[:, 0, 0, 0], torch.full((4,), 80.0 / 255.0)
    )

    stride_two = DynamicCleanDataset(
        tmp_path,
        ["train.hdf5"],
        normalization,
        image_size=(24, 32),
        sequence_length=4,
        temporal_stride=2,
    )
    stride_images, stride_targets = stride_two[3]
    torch.testing.assert_close(
        stride_images[:, 0, 0, 0], torch.tensor([0.0, 0.0, 5.0, 15.0]) / 255.0
    )
    torch.testing.assert_close(
        stride_targets["delta_marker"], torch.full((1200, 2), 2.0)
    )


def test_dynamic_audit_uses_only_train_trajectories_for_all_statistics(tmp_path):
    from scripts.audit_dynamic_v2 import audit_dynamic_dataset

    _write_trajectory(tmp_path / "train.hdf5", displacement_scale=1.0)
    _write_trajectory(
        tmp_path / "validation.hdf5",
        displacement_scale=1000.0,
        depth_offset=10000.0,
    )
    manifest = {
        "seed": 42,
        "grouping": "one_file_one_trajectory_both_sensors",
        "train": ["train.hdf5"],
        "val": ["validation.hdf5"],
    }

    report = audit_dynamic_dataset(
        tmp_path,
        manifest,
        sequence_length=4,
        temporal_stride=1,
        statistics_frame_stride=10,
    )

    assert report["history"] == {"sequence_length": 4, "temporal_stride": 1}
    assert report["inspected_splits"] == {"train": 1, "val": 1}
    expected_count = 2 * 3 * 1200 * 2
    assert report["normalization"]["marker"] == {
        "mean": 10.0,
        "std": pytest.approx(np.sqrt(200.0 / 3.0)),
        "count": expected_count,
        "statistics_frame_stride": 10,
        "source": "train_only",
    }
    assert report["normalization"]["delta_marker"] == {
        "mean": pytest.approx(2.0 / 3.0),
        "std": pytest.approx(np.sqrt(2.0) / 3.0),
        "count": expected_count,
        "statistics_frame_stride": 10,
        "source": "train_only",
    }
    depth_stats = report["normalization"]["depth"]
    assert depth_stats["mean"] == 10.0
    assert depth_stats["count"] == 2 * 3 * 24 * 32
    assert depth_stats["statistics_frame_stride"] == 10
    assert depth_stats["source"] == "train_only"


def test_dynamic_audit_checks_validation_finiteness_without_using_it(tmp_path):
    from scripts.audit_dynamic_v2 import audit_dynamic_dataset

    _write_trajectory(tmp_path / "train.hdf5")
    _write_trajectory(tmp_path / "validation.hdf5")
    with h5py.File(tmp_path / "validation.hdf5", "r+") as handle:
        handle["tactile/right_gsmini/depth"][7, 0, 0] = np.nan
    manifest = {"train": ["train.hdf5"], "val": ["validation.hdf5"]}

    with pytest.raises(ValueError, match="Nonfinite.*validation"):
        audit_dynamic_dataset(tmp_path, manifest)

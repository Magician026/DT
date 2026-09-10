"""Contracts for integrating causal Dynamic Details V2 history with ACT."""

from __future__ import annotations

import json
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACT_ROOT = PROJECT_ROOT / "policy" / "ACT"
DETR_ROOT = ACT_ROOT / "detr"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACT_ROOT))
sys.path.insert(0, str(DETR_ROOT))

from encoder.clean_v2 import PREPROCESS
from encoder.detail_v2 import (
    DynamicDetailV2Encoder,
    load_encoder_checkpoint,
    save_encoder_checkpoint,
)
import act_policy
from models.backbone import TactileBackbone
from scripts import train_policy_v2
from utils import TacArenaDataset, worker_clear_fn


@pytest.fixture(autouse=True)
def _limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    worker_clear_fn()
    torch.set_num_threads(previous)


def _write_episode(path: Path, length: int = 5) -> None:
    with h5py.File(path, "w") as root:
        root.create_dataset("/observations/qpos", data=np.arange(length * 2).reshape(length, 2))
        root.create_dataset("/action", data=np.arange(length * 2).reshape(length, 2) + 100)
        images = root.create_group("/observations/images")
        images.create_dataset(
            "cam_high",
            data=np.stack([np.full((6, 6, 3), 10 + index, np.uint8) for index in range(length)]),
        )
        images.create_dataset(
            "tac_left",
            data=np.stack([np.full((6, 6, 3), 20 + index, np.uint8) for index in range(length)]),
        )
        images.create_dataset(
            "tac_right",
            data=np.stack([np.full((6, 6, 3), 40 + index, np.uint8) for index in range(length)]),
        )


def _dataset(tmp_path: Path, *, sequence_length: int, temporal_stride: int):
    _write_episode(tmp_path / "episode_0.hdf5")
    stats = {
        "qpos_mean": torch.tensor([1.5, 2.5]),
        "qpos_std": torch.tensor([2.0, 4.0]),
        "action_mean": torch.tensor([101.0, 102.0]),
        "action_std": torch.tensor([2.0, 5.0]),
    }
    return TacArenaDataset(
        np.array([0]),
        tmp_path,
        ["cam_high"],
        ["tac_left", "tac_right"],
        stats,
        chunk_size=3,
        sequence_length=sequence_length,
        temporal_stride=temporal_stride,
    )


def test_training_history_is_causal_and_only_changes_tactile(tmp_path):
    static = _dataset(tmp_path, sequence_length=1, temporal_stride=1)
    dynamic = _dataset(tmp_path, sequence_length=4, temporal_stride=1)

    static_sample = static[3]
    dynamic_sample = dynamic[3]
    static_camera, static_tactile, static_qpos, static_action, static_pad = static_sample
    dynamic_camera, dynamic_tactile, dynamic_qpos, dynamic_action, dynamic_pad = dynamic_sample

    assert static_tactile.shape == (2, 3, 256, 256)
    assert dynamic_tactile.shape == (2, 4, 3, 256, 256)
    torch.testing.assert_close(dynamic_tactile[:, -1], static_tactile, rtol=0, atol=0)
    torch.testing.assert_close(dynamic_camera, static_camera, rtol=0, atol=0)
    torch.testing.assert_close(dynamic_qpos, static_qpos, rtol=0, atol=0)
    torch.testing.assert_close(dynamic_action, static_action, rtol=0, atol=0)
    torch.testing.assert_close(dynamic_pad, static_pad, rtol=0, atol=0)

    left_values = dynamic_tactile[0, :, 0, 0, 0] * 255
    right_values = dynamic_tactile[1, :, 0, 0, 0] * 255
    torch.testing.assert_close(left_values, torch.tensor([20.0, 21.0, 22.0, 23.0]))
    torch.testing.assert_close(right_values, torch.tensor([40.0, 41.0, 42.0, 43.0]))


def test_training_history_repeat_first_padding_and_stride(tmp_path):
    dynamic = _dataset(tmp_path, sequence_length=4, temporal_stride=2)

    first = dynamic[0][1][0, :, 0, 0, 0] * 255
    third = dynamic[3][1][0, :, 0, 0, 0] * 255

    torch.testing.assert_close(first, torch.tensor([20.0, 20.0, 20.0, 20.0]))
    torch.testing.assert_close(third, torch.tensor([20.0, 20.0, 21.0, 23.0]))


@pytest.fixture
def dynamic_checkpoint(tmp_path):
    torch.manual_seed(29)
    encoder = DynamicDetailV2Encoder().eval()
    path = tmp_path / "dynamic.pt"
    save_encoder_checkpoint(path, encoder, PREPROCESS)
    return path


def _batch_norms(module):
    return [part for part in module.modules() if isinstance(part, nn.BatchNorm2d)]


def test_dynamic_backbone_strictly_loads_and_preserves_act_shape_and_frozen_bn(
    dynamic_checkpoint,
):
    model = TactileBackbone(
        "resnet18",
        str(dynamic_checkpoint),
        ["tac_left", "tac_right"],
        True,
        False,
        nn.Identity(),
        tactile_type="feat",
        tactile_encoder_type="detail_v2_dynamic",
    ).train()
    canonical, metadata = load_encoder_checkpoint(
        dynamic_checkpoint,
        expected_encoder_type="detail_v2_dynamic",
        expected_preprocess=PREPROCESS,
    )
    batch_norms = _batch_norms(model.backbone)
    running_before = [(bn.running_mean.clone(), bn.running_var.clone()) for bn in batch_norms]

    inputs = torch.randn(2, 4, 3, 64, 64)
    features, positions = model(inputs)
    features[0].square().mean().backward()

    assert isinstance(model.backbone, DynamicDetailV2Encoder)
    assert model.checkpoint_metadata == metadata
    assert features[0].shape == (2, 512, 1, 1)
    assert positions[0].shape == (1, 512, 1, 1)
    assert model.backbone.cross_attention.in_proj_weight.grad.norm() > 0
    assert model.backbone.dynamic_attention.in_proj_weight.grad.norm() > 0
    assert model.backbone.gate_logit.grad.abs() > 0
    assert all(not batch_norm.training for batch_norm in batch_norms)
    assert all(not batch_norm.weight.requires_grad for batch_norm in batch_norms)
    assert all(not batch_norm.bias.requires_grad for batch_norm in batch_norms)
    for batch_norm, (mean_before, variance_before) in zip(batch_norms, running_before):
        torch.testing.assert_close(batch_norm.running_mean, mean_before, rtol=0, atol=0)
        torch.testing.assert_close(batch_norm.running_var, variance_before, rtol=0, atol=0)

    with pytest.raises(ValueError, match="five-dimensional"):
        model(torch.randn(2, 3, 64, 64))


class _CapturePolicy(nn.Module):
    def __init__(self, _args, _robot_config=None):
        super().__init__()
        self.calls = []

    def forward(self, qpos, camera, tactile):
        self.calls.append((qpos.clone(), camera.clone(), tactile.clone()))
        return torch.zeros((qpos.shape[0], 1, qpos.shape[1]), device=qpos.device)


def _deployment(tmp_path: Path, monkeypatch, *, sequence_length=3, temporal_stride=2):
    checkpoint_dir = tmp_path / "policy"
    checkpoint_dir.mkdir()
    torch.save({}, checkpoint_dir / "policy_last.ckpt")
    with (checkpoint_dir / "dataset_stats.pkl").open("wb") as stream:
        pickle.dump(
            {
                "qpos_mean": np.zeros(1, dtype=np.float32),
                "qpos_std": np.ones(1, dtype=np.float32),
                "action_mean": np.zeros(1, dtype=np.float32),
                "action_std": np.ones(1, dtype=np.float32),
            },
            stream,
        )
    monkeypatch.setattr(act_policy, "ACTPolicy", _CapturePolicy)
    return act_policy.ACT(
        {
            "ckpt_dir": str(checkpoint_dir),
            "device": "cpu",
            "chunk_size": 1,
            "state_dim": 1,
            "camera_names": ["cam_high"],
            "tactile_names": ["tac_left", "tac_right"],
            "temporal_agg": False,
            "tactile_sequence_length": sequence_length,
            "tactile_temporal_stride": temporal_stride,
        }
    )


def _observation(value: int):
    return {
        "qpos": np.array([value], dtype=np.float32),
        "cam_high": torch.full((3, 2, 2), float(100 + value)),
        "tac_left": torch.full((3, 2, 2), float(value)),
        "tac_right": torch.full((3, 2, 2), float(10 + value)),
    }


def test_deployment_history_applies_stride_and_reset_isolates_episodes(tmp_path, monkeypatch):
    deployment = _deployment(tmp_path, monkeypatch)

    for value in range(5):
        deployment.get_action(_observation(value))

    calls = deployment.policy.calls
    assert calls[0][2].shape == (1, 2, 3, 3, 2, 2)
    torch.testing.assert_close(calls[0][2][0, 0, :, 0, 0, 0], torch.tensor([0.0, 0.0, 0.0]))
    torch.testing.assert_close(calls[1][2][0, 0, :, 0, 0, 0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(calls[2][2][0, 0, :, 0, 0, 0], torch.tensor([0.0, 0.0, 2.0]))
    torch.testing.assert_close(calls[4][2][0, 0, :, 0, 0, 0], torch.tensor([0.0, 2.0, 4.0]))
    torch.testing.assert_close(calls[4][2][0, 1, :, 0, 0, 0], torch.tensor([10.0, 12.0, 14.0]))
    assert calls[4][1].shape == (1, 1, 3, 2, 2)
    assert calls[4][0].item() == 4

    deployment.reset()
    deployment.get_action(_observation(9))
    assert deployment.t == 1
    torch.testing.assert_close(
        deployment.policy.calls[-1][2][0, 0, :, 0, 0, 0],
        torch.tensor([9.0, 9.0, 9.0]),
    )


def test_formal_dynamic_config_preserves_act_and_optimizer_contract():
    static = yaml.safe_load((PROJECT_ROOT / "configs/policy_v2.yml").read_text())
    dynamic = yaml.safe_load((PROJECT_ROOT / "configs/policy_dynamic_v2_hdmi.yml").read_text())
    changed = {
        key
        for key in static.keys() | dynamic.keys()
        if static.get(key) != dynamic.get(key)
    }
    assert changed == {
        "tactile_encoder_type",
        "tactile_sequence_length",
        "tactile_temporal_stride",
    }
    assert dynamic["tactile_encoder_type"] == "detail_v2_dynamic"
    assert dynamic["tactile_sequence_length"] == 4
    assert dynamic["tactile_temporal_stride"] == 1

    schedule = train_policy_v2.validate_training_contract(
        dynamic,
        smoke=False,
        smoke_batch=2,
        optimizer_group_lrs=[1e-5, 1e-5, 1e-5],
    )
    assert schedule == {
        "optimizer_updates": 4000,
        "physical_batch": 32,
        "gradient_accumulation": 2,
        "effective_batch": 64,
    }

    for key in ("lr", "lr_vision_backbone", "lr_tactile_backbone"):
        broken = dict(dynamic, **{key: 2e-5})
        with pytest.raises(ValueError, match="1e-5"):
            train_policy_v2.validate_training_contract(broken, smoke=False, smoke_batch=2)
    with pytest.raises(ValueError, match="three optimizer groups"):
        train_policy_v2.validate_training_contract(
            dynamic,
            smoke=False,
            smoke_batch=2,
            optimizer_group_lrs=[1e-5, 1e-5],
        )
    with pytest.raises(ValueError, match="three optimizer groups"):
        train_policy_v2.validate_training_contract(
            dynamic,
            smoke=True,
            smoke_batch=2,
            optimizer_group_lrs=[1e-5, 1e-5],
        )


def test_stride2_variant_changes_only_temporal_stride():
    encoder_base = json.loads(
        (PROJECT_ROOT / "configs/encoder_dynamic_v2_hdmi.json").read_text()
    )
    encoder_stride2 = json.loads(
        (PROJECT_ROOT / "configs/encoder_dynamic_v2_hdmi_stride2.json").read_text()
    )
    encoder_changed = {
        key
        for key in encoder_base.keys() | encoder_stride2.keys()
        if encoder_base.get(key) != encoder_stride2.get(key)
    }
    assert encoder_changed == {"temporal_stride"}
    assert encoder_stride2["sequence_length"] == 4
    assert encoder_stride2["temporal_stride"] == 2

    policy_base = yaml.safe_load(
        (PROJECT_ROOT / "configs/policy_dynamic_v2_hdmi.yml").read_text()
    )
    policy_stride2 = yaml.safe_load(
        (PROJECT_ROOT / "configs/policy_dynamic_v2_hdmi_stride2.yml").read_text()
    )
    policy_changed = {
        key
        for key in policy_base.keys() | policy_stride2.keys()
        if policy_base.get(key) != policy_stride2.get(key)
    }
    assert policy_changed == {"tactile_temporal_stride"}
    assert policy_stride2["tactile_sequence_length"] == 4
    assert policy_stride2["tactile_temporal_stride"] == 2
    assert train_policy_v2.validate_training_contract(
        policy_stride2,
        smoke=False,
        smoke_batch=2,
        optimizer_group_lrs=[1e-5, 1e-5, 1e-5],
    ) == {
        "optimizer_updates": 4000,
        "physical_batch": 32,
        "gradient_accumulation": 2,
        "effective_batch": 64,
    }

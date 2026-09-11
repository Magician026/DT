"""Minimal TRA ACT, causal-window, and deployment contract checks."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml
from torch import nn

from test_dynamic_policy import _dataset, _deployment, _observation, _write_episode
from test_eval_v2 import _runtime_fixture, _sha
from encoder.clean_v2 import PREPROCESS
from encoder.detail_v2 import build_encoder, save_encoder_checkpoint
from models.backbone import TactileBackbone
from scripts.prepare_eval_v2 import prepare_runtime
from utils import TacArenaDataset, worker_clear_fn


@pytest.fixture(autouse=True)
def limit_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    worker_clear_fn()
    torch.set_num_threads(old)


def test_tra_backbone_preserves_act_interface_and_bn(tmp_path):
    path = tmp_path / 'tra.pt'
    save_encoder_checkpoint(path, build_encoder('detail_v2_tra'), PREPROCESS)
    model = TactileBackbone('resnet18', str(path), ['tac_left', 'tac_right'],
                           True, False, nn.Identity(), tactile_type='feat',
                           tactile_encoder_type='detail_v2_tra').train()
    feat, pos = model(torch.randn(2, 4, 3, 64, 64))
    assert feat[0].shape == (2, 512, 1, 1)
    assert pos[0].shape == (1, 512, 1, 1)
    feat[0].square().mean().backward()
    assert model.backbone.gru.weight_ih_l0.grad.norm() > 0
    assert model.backbone.alpha.grad.abs() > 0
    assert all(not part.training and not part.weight.requires_grad
               for part in model.backbone.modules() if isinstance(part, nn.BatchNorm2d))


def test_t4_training_padding_and_episode_boundary(tmp_path):
    dataset = _dataset(tmp_path, sequence_length=4, temporal_stride=1)
    for i, expected in enumerate(([20]*4, [20,20,20,21], [20,20,21,22], [20,21,22,23])):
        torch.testing.assert_close(dataset[i][1][0,:,0,0,0]*255, torch.tensor(expected).float())
    _write_episode(tmp_path / 'episode_1.hdf5')
    with h5py.File(tmp_path / 'episode_1.hdf5', 'r+') as root:
        root['/observations/images/tac_left'][:] += 100
    combined = TacArenaDataset(np.array([0,1]), tmp_path, ['cam_high'],
        ['tac_left','tac_right'], dataset.norm_stats, 3, 4, 1)
    torch.testing.assert_close(combined[5][1][0,:,0,0,0]*255, torch.tensor([120.]*4))


def test_t4_live_history_has_no_future_and_reset_clears_episode(tmp_path, monkeypatch):
    deployment = _deployment(tmp_path, monkeypatch, sequence_length=4, temporal_stride=1)
    for i, expected in enumerate(([0]*4, [0,0,0,1], [0,0,1,2], [0,1,2,3])):
        deployment.get_action(_observation(i))
        actual = deployment.policy.calls[-1][2][0,0,:,0,0,0]
        torch.testing.assert_close(actual, torch.tensor(expected).float())
    deployment.reset()
    deployment.get_action(_observation(9))
    torch.testing.assert_close(deployment.policy.calls[-1][2][0,0,:,0,0,0], torch.tensor([9.]*4))


def test_eval_accepts_tra_and_rejects_stride_drift(tmp_path):
    external, source, policy, seeds = _runtime_fixture(tmp_path, encoder_type='detail_v2_dynamic')
    cfgpath, donepath = policy/'train_config.yml', policy/'completion.json'
    cfg = yaml.safe_load(cfgpath.read_text())
    done = json.loads(donepath.read_text())
    ckpt = Path(cfg['tactile_ckpt'])
    payload = torch.load(ckpt, weights_only=True)
    payload['encoder_type'] = cfg['tactile_encoder_type'] = done['tactile_encoder_type'] = 'detail_v2_tra'
    torch.save(payload, ckpt)
    done['encoder_sha256'] = _sha(ckpt)
    cfgpath.write_text(yaml.safe_dump(cfg)); donepath.write_text(json.dumps(done))
    result = prepare_runtime(external_runtime=external, source=source, policy_run=policy,
        out=tmp_path/'eval', gpu='2', seeds_path=seeds)
    assert result['tactile_encoder_type'] == 'detail_v2_tra'
    assert result['tactile_sequence_length'] == 4
    cfg['tactile_temporal_stride'] = 2
    cfgpath.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='temporal metadata'):
        prepare_runtime(external_runtime=external, source=source, policy_run=policy,
            out=tmp_path/'bad_eval', gpu='2', seeds_path=seeds)

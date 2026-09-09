"""Policy-side contract tests for the canonical Details V2 encoder."""

from __future__ import annotations

import hashlib
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETR_ROOT = PROJECT_ROOT / "policy" / "ACT" / "detr"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DETR_ROOT))

from encoder.clean_v2 import PREPROCESS
from encoder.detail_v2 import (
    DetailV2Encoder,
    OriginalEncoder,
    load_encoder_checkpoint,
    save_encoder_checkpoint,
)
from models.backbone import TactileBackbone, build_tactile_backbone
from models import network as tactile_network


@pytest.fixture(autouse=True)
def _limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def detail_v2_checkpoint(tmp_path):
    torch.manual_seed(19)
    encoder = DetailV2Encoder().eval()
    path = tmp_path / "detail_v2.pt"
    save_encoder_checkpoint(path, encoder, PREPROCESS)
    return path


def _build_policy_encoder(path, *, train_backbone=True):
    return TactileBackbone(
        name="resnet18",
        ckpt=str(path),
        tac_names=["tac_left", "tac_right"],
        train_backbone=train_backbone,
        return_interm_layers=False,
        position_embedding=nn.Identity(),
        tactile_type="feat",
        tactile_encoder_type="detail_v2",
    )


def _batch_norms(module):
    return [part for part in module.modules() if isinstance(part, nn.BatchNorm2d)]


def test_policy_uses_canonical_encoder_and_strict_checkpoint_contract(
    detail_v2_checkpoint,
):
    policy_encoder = _build_policy_encoder(detail_v2_checkpoint)
    canonical, metadata = load_encoder_checkpoint(
        detail_v2_checkpoint,
        expected_encoder_type="detail_v2",
        expected_preprocess=PREPROCESS,
    )

    assert isinstance(policy_encoder.backbone, DetailV2Encoder)
    assert type(policy_encoder.backbone) is type(canonical)
    assert policy_encoder.checkpoint_metadata == metadata
    assert policy_encoder.checkpoint_load_coverage == 1.0
    assert policy_encoder.checkpoint_sha256 == hashlib.sha256(
        detail_v2_checkpoint.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("batch_size", [1, 3])
def test_detail_v2_feat_keeps_act_input_shape_and_eval_embedding_alignment(
    detail_v2_checkpoint, batch_size
):
    policy_encoder = _build_policy_encoder(detail_v2_checkpoint).eval()
    canonical, _ = load_encoder_checkpoint(
        detail_v2_checkpoint,
        expected_encoder_type="detail_v2",
        expected_preprocess=PREPROCESS,
    )
    canonical.eval()
    inputs = torch.randn(batch_size, 3, 64, 64)

    with torch.no_grad():
        features, positions = policy_encoder(inputs)
        expected = canonical(inputs)

    assert len(features) == len(positions) == 1
    assert features[0].shape == (batch_size, 512, 1, 1)
    assert positions[0].shape == (1, 512, 1, 1)
    assert torch.isfinite(features[0]).all()
    torch.testing.assert_close(features[0].flatten(1), expected, rtol=0, atol=0)


def test_detail_v2_rejects_missing_checkpoint_and_full_mode(tmp_path):
    with pytest.raises(FileNotFoundError, match="tactile checkpoint"):
        _build_policy_encoder(tmp_path / "missing.pt")

    encoder = DetailV2Encoder().eval()
    path = tmp_path / "detail_v2.pt"
    save_encoder_checkpoint(path, encoder, PREPROCESS)
    with pytest.raises(ValueError, match="only supports tactile_type='feat'"):
        TactileBackbone(
            "resnet18",
            str(path),
            ["tac_left"],
            True,
            False,
            nn.Identity(),
            tactile_type="full",
            tactile_encoder_type="detail_v2",
        )


def test_finetune_freezes_bn_affine_and_stats_but_updates_trunk_and_attention(
    detail_v2_checkpoint,
):
    model = _build_policy_encoder(detail_v2_checkpoint, train_backbone=True)
    model.train()
    batch_norms = _batch_norms(model.backbone)
    assert batch_norms
    assert all(not batch_norm.training for batch_norm in batch_norms)
    assert all(not batch_norm.weight.requires_grad for batch_norm in batch_norms)
    assert all(not batch_norm.bias.requires_grad for batch_norm in batch_norms)
    assert model.backbone.trunk.conv1.weight.requires_grad
    assert model.backbone.cross_attention.in_proj_weight.requires_grad

    running_before = [(bn.running_mean.clone(), bn.running_var.clone()) for bn in batch_norms]
    trunk_before = model.backbone.trunk.conv1.weight.detach().clone()
    attention_before = model.backbone.cross_attention.in_proj_weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-5
    )
    features, _ = model(torch.randn(2, 3, 64, 64))
    features[0].square().mean().backward()
    optimizer.step()

    assert not torch.equal(model.backbone.trunk.conv1.weight, trunk_before)
    assert not torch.equal(model.backbone.cross_attention.in_proj_weight, attention_before)
    for batch_norm, (mean_before, variance_before) in zip(batch_norms, running_before):
        torch.testing.assert_close(batch_norm.running_mean, mean_before, rtol=0, atol=0)
        torch.testing.assert_close(batch_norm.running_var, variance_before, rtol=0, atol=0)


def test_frozen_encoder_disables_all_gradients_and_stays_in_eval_mode(
    detail_v2_checkpoint,
):
    model = _build_policy_encoder(detail_v2_checkpoint, train_backbone=False)
    model.train()

    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert not model.backbone.training
    assert all(not batch_norm.training for batch_norm in _batch_norms(model.backbone))

    inputs = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        before = model(inputs)[0][0]
        after = model(inputs)[0][0]
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_factory_selects_detail_v2_and_preserves_512_channels(detail_v2_checkpoint):
    args = Namespace(
        lr_tactile_backbone=1e-5,
        tactile_masks=False,
        tactile_type="feat",
        tactile_encoder_type="detail_v2",
        tactile_backbone="resnet18",
        tactile_ckpt=str(detail_v2_checkpoint),
        tactile_names=["tac_left", "tac_right"],
        hidden_dim=512,
        position_embedding="sine",
    )

    model = build_tactile_backbone(args)

    assert isinstance(model.backbone, DetailV2Encoder)
    assert model.num_channels == 512


def test_original_encoder_only_checkpoint_uses_canonical_loader_and_bn_contract(
    tmp_path,
):
    torch.manual_seed(23)
    source = OriginalEncoder().eval()
    path = tmp_path / "original_encoder_only.pt"
    save_encoder_checkpoint(path, source, PREPROCESS)

    model = TactileBackbone(
        "resnet18",
        str(path),
        ["tac_left", "tac_right"],
        True,
        False,
        nn.Identity(),
        tactile_type="feat",
        tactile_encoder_type="original",
    )
    model.train()
    inputs = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        actual = model(inputs)[0][0].flatten(1)
        expected = source(inputs)

    assert isinstance(model.backbone, OriginalEncoder)
    assert model.checkpoint_metadata["encoder_type"] == "original"
    assert model.checkpoint_metadata["preprocess"] == PREPROCESS
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(not batch_norm.training for batch_norm in _batch_norms(model.backbone))
    assert all(
        not parameter.requires_grad
        for batch_norm in _batch_norms(model.backbone)
        for parameter in (batch_norm.weight, batch_norm.bias)
    )


def test_original_legacy_state_dict_remains_strictly_loadable(tmp_path, monkeypatch):
    class TinyLegacyTactile(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(3, 512),
            )

    monkeypatch.setattr(tactile_network, "Tactile", TinyLegacyTactile)
    source = TinyLegacyTactile()
    path = tmp_path / "legacy_state_dict.pt"
    torch.save(source.state_dict(), path)

    model = TactileBackbone(
        "resnet18",
        str(path),
        ["tac_left"],
        True,
        False,
        nn.Identity(),
        tactile_type="feat",
        tactile_encoder_type="original",
    )

    assert model.checkpoint_metadata["format"] == "legacy_tactile_state_dict"
    assert model.checkpoint_load_coverage == 1.0
    output = model(torch.randn(2, 3, 8, 8))[0][0]
    assert output.shape == (2, 512, 1, 1)

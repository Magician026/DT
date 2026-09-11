"""Training contracts for Spatial-anchored Details V2-TRA."""

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from encoder.tra_v2 import TRADetailV2Encoder
from scripts import train_policy_v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DummyTRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Linear(4, 4)
        self.spatial_bn = nn.BatchNorm1d(4)
        self.gru = nn.GRU(4, 4, batch_first=True)
        self.temporal_mlp = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))
        self.alpha = nn.Parameter(torch.tensor(-2.1972246))
        self.fusion_norm = nn.LayerNorm(4)

        # Canonical encoder adapters freeze BN affine before optimizer creation.
        self.spatial_bn.eval()
        self.spatial_bn.weight.requires_grad_(False)
        self.spatial_bn.bias.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.spatial_bn.eval()
        return self

    def forward(self, value):
        detail = self.spatial_bn(self.spatial(value))
        hidden, _ = self.gru(detail.unsqueeze(1).expand(-1, 4, -1))
        residual = self.temporal_mlp(hidden[:, -1])
        return self.fusion_norm(detail + torch.sigmoid(self.alpha) * residual)


def _optimizer_for(model):
    head = nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": 1e-5},
            {"params": [], "lr": 1e-5},
            {"params": [parameter for parameter in model.parameters() if parameter.requires_grad], "lr": 1e-5},
        ],
        lr=1e-5,
    )
    return head, optimizer


def test_tra_config_changes_only_temporal_contract_and_preserves_schedule():
    spatial = yaml.safe_load((PROJECT_ROOT / "configs/policy_v2.yml").read_text())
    tra = yaml.safe_load((PROJECT_ROOT / "configs/policy_tra_v2_hdmi.yml").read_text())

    changed = {
        key for key in spatial.keys() | tra.keys() if spatial.get(key) != tra.get(key)
    }
    assert changed == {
        "tactile_encoder_type",
        "tactile_sequence_length",
        "tactile_temporal_stride",
        "tactile_spatial_freeze_updates",
    }
    assert tra["tactile_encoder_type"] == "detail_v2_tra"
    assert tra["tactile_sequence_length"] == 4
    assert tra["tactile_temporal_stride"] == 1
    assert tra["tactile_spatial_freeze_updates"] == 400
    assert train_policy_v2.validate_training_contract(
        tra,
        smoke=False,
        smoke_batch=2,
        optimizer_group_lrs=[1e-5, 1e-5, 1e-5],
    ) == {
        "optimizer_updates": 4000,
        "physical_batch": 32,
        "gradient_accumulation": 2,
        "effective_batch": 64,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tactile_sequence_length", 3, "length=4 and stride=1"),
        ("tactile_temporal_stride", 2, "length=4 and stride=1"),
        ("tactile_spatial_freeze_updates", 401, "400 Spatial freeze updates"),
    ],
)
def test_tra_formal_contract_rejects_deviations(field, value, message):
    config = yaml.safe_load((PROJECT_ROOT / "configs/policy_tra_v2_hdmi.yml").read_text())
    config[field] = value
    with pytest.raises(ValueError, match=message):
        train_policy_v2.validate_training_contract(
            config,
            smoke=False,
            smoke_batch=2,
            optimizer_group_lrs=[1e-5, 1e-5, 1e-5],
        )


def test_optimizer_keeps_spatial_parameters_while_warmup_freezes_updates():
    torch.manual_seed(7)
    model = _DummyTRA()
    head, optimizer = _optimizer_for(model)
    base_trainable = train_policy_v2.capture_trainable_state(model)
    managed = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert id(model.spatial.weight) in managed

    stage = train_policy_v2.set_tra_training_stage(
        model, step=0, freeze_updates=400, base_trainable=base_trainable
    )
    assert stage == "temporal_warmup"
    assert not model.spatial.weight.requires_grad
    assert model.gru.weight_ih_l0.requires_grad
    assert model.temporal_mlp[0].weight.requires_grad
    assert model.alpha.requires_grad
    assert model.fusion_norm.weight.requires_grad
    assert not model.spatial_bn.weight.requires_grad

    spatial_before = model.spatial.weight.detach().clone()
    alpha_before = model.alpha.detach().clone()
    output = head(model(torch.randn(3, 4))).square().mean()
    optimizer.zero_grad(set_to_none=True)
    output.backward()
    gradients = train_policy_v2.tra_gradient_report(model)
    assert gradients["gru"] > 0
    assert gradients["alpha"] > 0
    optimizer.step()
    torch.testing.assert_close(model.spatial.weight, spatial_before, rtol=0, atol=0)
    assert not torch.equal(model.alpha, alpha_before)

    stage = train_policy_v2.set_tra_training_stage(
        model, step=400, freeze_updates=400, base_trainable=base_trainable
    )
    assert stage == "joint_finetune"
    assert model.spatial.weight.requires_grad
    assert not model.spatial_bn.weight.requires_grad
    model.train()
    assert not model.spatial_bn.training


def test_canonical_tra_module_names_follow_training_stage_contract():
    encoder = TRADetailV2Encoder()
    base_trainable = train_policy_v2.capture_trainable_state(encoder)
    stage = train_policy_v2.set_tra_training_stage(
        encoder, step=0, freeze_updates=400, base_trainable=base_trainable
    )
    assert stage == "temporal_warmup"
    temporal_names = {
        name
        for name in base_trainable
        if train_policy_v2._is_tra_temporal_parameter(name)
    }
    assert temporal_names
    assert all(
        parameter.requires_grad == (base_trainable[name] and name in temporal_names)
        for name, parameter in encoder.named_parameters()
    )
    assert train_policy_v2.alpha_snapshot(encoder)["sigmoid"] == pytest.approx(0.1)

    stage = train_policy_v2.set_tra_training_stage(
        encoder, step=400, freeze_updates=400, base_trainable=base_trainable
    )
    assert stage == "joint_finetune"
    assert all(
        parameter.requires_grad == base_trainable[name]
        for name, parameter in encoder.named_parameters()
    )


def test_tra_forward_backward_checkpoint_reload(tmp_path):
    torch.manual_seed(19)
    model = _DummyTRA()
    base_trainable = train_policy_v2.capture_trainable_state(model)
    train_policy_v2.set_tra_training_stage(
        model, step=400, freeze_updates=400, base_trainable=base_trainable
    )
    sample = torch.randn(2, 4)
    output = model(sample)
    output.square().mean().backward()
    assert model.spatial.weight.grad.norm() > 0
    assert model.gru.weight_ih_l0.grad.norm() > 0
    assert model.alpha.grad.abs() > 0

    checkpoint = tmp_path / "policy_last.ckpt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = _DummyTRA()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    reloaded.eval()
    model.eval()
    torch.testing.assert_close(reloaded(sample), model(sample), rtol=0, atol=0)


def test_trainability_map_mismatch_is_rejected():
    model = _DummyTRA()
    base_trainable = train_policy_v2.capture_trainable_state(model)
    del base_trainable["spatial.weight"]
    with pytest.raises(ValueError, match="trainability map"):
        train_policy_v2.set_tra_training_stage(
            model, step=0, freeze_updates=400, base_trainable=base_trainable
        )

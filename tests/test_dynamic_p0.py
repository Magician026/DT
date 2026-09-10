import hashlib

import pytest
import torch
from torch import nn


class _FakeDynamicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(512))
        self.dynamic = nn.Parameter(torch.zeros(256))

    def forward_with_details(self, inputs):
        batch = inputs.shape[0]
        return (
            self.embedding.expand(batch, -1),
            self.dynamic.expand(batch, -1),
        )

    def forward(self, inputs):
        return self.forward_with_details(inputs)[0]


def test_dynamic_p0_reports_unweighted_delta_and_applies_half_weight():
    from encoder.pretrain_v2 import P0Model

    model = P0Model(
        _FakeDynamicEncoder(), ["depth", "marker"], dynamic_weight=0.5
    )
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    model.encoder.embedding.data.fill_(11.0)
    model.encoder.dynamic.data.fill_(-3.0)
    observed = {}

    def capture_dynamic_input(_module, inputs):
        observed["dynamic_decoder_input"] = inputs[0].detach().clone()

    handle = model.dynamic_decoder.register_forward_pre_hook(capture_dynamic_input)
    inputs = torch.zeros(2, 4, 3, 16, 16)
    targets = {
        "depth": torch.ones(2, 1, 24, 32),
        "marker": torch.full((2, 1200, 2), 2.0),
        "delta_marker": torch.full((2, 1200, 2), 3.0),
    }

    try:
        loss, parts = model.loss(inputs, targets)
    finally:
        handle.remove()

    assert model.dynamic_decoder(torch.zeros(2, 256)).shape == (2, 1200, 2)
    torch.testing.assert_close(
        observed["dynamic_decoder_input"], torch.full((2, 256), -3.0)
    )
    assert not torch.equal(
        observed["dynamic_decoder_input"],
        model.encoder.embedding.detach()[:256].expand(2, -1),
    )
    assert set(parts) == {"depth", "marker", "delta_marker", "total"}
    torch.testing.assert_close(parts["depth"], torch.tensor(1.0))
    torch.testing.assert_close(parts["marker"], torch.tensor(4.0))
    torch.testing.assert_close(parts["delta_marker"], torch.tensor(9.0))
    torch.testing.assert_close(loss, torch.tensor(9.5))
    torch.testing.assert_close(parts["total"], torch.tensor(9.5))


def test_dynamic_auxiliary_loss_reaches_head_attention_gate_and_shared_trunk():
    from encoder.detail_v2 import build_encoder
    from encoder.pretrain_v2 import P0Model

    torch.manual_seed(17)
    encoder = build_encoder("detail_v2_dynamic")
    model = P0Model(encoder, ["depth", "marker"], dynamic_weight=0.5).train()
    inputs = torch.randn(2, 4, 3, 64, 64)
    targets = {
        "depth": torch.randn(2, 1, 16, 16),
        "marker": torch.randn(2, 1200, 2),
        "delta_marker": torch.randn(2, 1200, 2),
    }

    loss, _ = model.loss(inputs, targets)
    loss.backward()

    gradients = {
        "dynamic_head": model.dynamic_decoder[0].weight.grad,
        "dynamic_attention": encoder.dynamic_attention.in_proj_weight.grad,
        "gate": encoder.gate_logit.grad,
        "trunk": encoder.trunk.conv1.weight.grad,
    }
    for name, gradient in gradients.items():
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum().item() > 0, name


def test_dynamic_initialization_requires_and_records_strict_spatial_provenance(tmp_path):
    from encoder.detail_v2 import build_encoder, save_encoder_checkpoint
    from encoder.clean_v2 import PREPROCESS
    from scripts.train_encoder_v2 import initialize_encoder

    source = tmp_path / "spatial.pt"
    save_encoder_checkpoint(source, build_encoder("detail_v2"), PREPROCESS)
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    config = {
        "encoder_type": "detail_v2_dynamic",
        "sequence_length": 4,
        "temporal_stride": 1,
        "initial_gate": 0.15,
    }

    with pytest.raises(ValueError, match="--spatial-init"):
        initialize_encoder(config, spatial_init=None, spatial_source_commit="abc123")
    with pytest.raises(ValueError, match="--spatial-source-commit"):
        initialize_encoder(config, spatial_init=source, spatial_source_commit=None)

    encoder, provenance = initialize_encoder(
        config,
        spatial_init=source,
        spatial_source_commit="abc123",
    )

    assert encoder.encoder_type == "detail_v2_dynamic"
    assert provenance["spatial_init"] == str(source.resolve())
    assert provenance["spatial_init_sha256"] == expected_sha
    assert provenance["spatial_source_commit"] == "abc123"
    assert provenance["warm_start_report"]["source_encoder_type"] == "detail_v2"
    assert provenance["warm_start_report"]["target_encoder_type"] == "detail_v2_dynamic"
    assert provenance["warm_start_report"]["compatible_key_count"] > 0
    assert provenance["warm_start_report"]["new_key_count"] > 0
    spatial_state = torch.load(source, weights_only=True)["encoder_state"]
    dynamic_state = encoder.state_dict()
    for name, value in spatial_state.items():
        torch.testing.assert_close(dynamic_state[name], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [("channels", "rgb"), ("scale", 1.0)],
)
def test_dynamic_initialization_rejects_spatial_preprocess_drift(
    tmp_path, field, value
):
    from encoder.clean_v2 import PREPROCESS
    from encoder.detail_v2 import build_encoder, save_encoder_checkpoint
    from scripts.train_encoder_v2 import initialize_encoder

    source = tmp_path / f"bad-{field}.pt"
    altered = {**PREPROCESS, field: value}
    save_encoder_checkpoint(source, build_encoder("detail_v2"), altered)
    config = {
        "encoder_type": "detail_v2_dynamic",
        "sequence_length": 4,
        "temporal_stride": 1,
        "initial_gate": 0.15,
    }

    with pytest.raises(ValueError, match="preprocess mismatch"):
        initialize_encoder(
            config,
            spatial_init=source,
            spatial_source_commit="abc123",
        )


def test_static_initialization_remains_backward_compatible():
    from scripts.train_encoder_v2 import initialize_encoder

    encoder, provenance = initialize_encoder(
        {"encoder_type": "detail_v2"},
        spatial_init=None,
        spatial_source_commit=None,
    )
    assert encoder.encoder_type == "detail_v2"
    assert provenance == {}


def test_dynamic_training_rejects_non_train_or_mismatched_audit_contract():
    from scripts.train_encoder_v2 import validate_dynamic_audit

    config = {
        "encoder_type": "detail_v2_dynamic",
        "sequence_length": 4,
        "temporal_stride": 1,
    }
    audit = {
        "history": {"sequence_length": 4, "temporal_stride": 1},
        "normalization": {
            name: {
                "mean": 0.0,
                "std": 1.0,
                "count": 2400,
                "statistics_frame_stride": 10,
                "source": "train_only",
            }
            for name in ("depth", "marker", "delta_marker")
        },
    }
    validate_dynamic_audit(config, audit)

    wrong_history = {**audit, "history": {"sequence_length": 3, "temporal_stride": 1}}
    with pytest.raises(ValueError, match="history"):
        validate_dynamic_audit(config, wrong_history)
    wrong_source = {
        **audit,
        "normalization": {
            **audit["normalization"],
            "delta_marker": {
                **audit["normalization"]["delta_marker"],
                "source": "train_and_val",
            },
        },
    }
    with pytest.raises(ValueError, match="train-only"):
        validate_dynamic_audit(config, wrong_source)

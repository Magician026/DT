import copy
from pathlib import Path

import pytest
import torch

from encoder.detail_v2 import (
    DynamicDetailV2Encoder,
    build_encoder,
    export_encoder_checkpoint,
    load_encoder_checkpoint,
    save_encoder_checkpoint,
    warm_start_dynamic_encoder,
)


PREPROCESS = {
    "source_key": "rgb_marker",
    "decode": "cv2_preserve_native_channels",
    "dtype": "float32",
    "scale": "uint8_div_255",
    "resize": [256, 256],
}


@pytest.fixture(autouse=True)
def _limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_dynamic_encoder_returns_finite_embedding_and_pre_gate_detail():
    encoder = build_encoder("detail_v2_dynamic").eval()
    inputs = torch.randn(2, 4, 3, 64, 64)

    with torch.no_grad():
        embedding = encoder(inputs)
        detailed_embedding, detail_dynamic = encoder.forward_with_details(inputs)

    assert isinstance(encoder, DynamicDetailV2Encoder)
    assert embedding.shape == (2, 512)
    assert detail_dynamic.shape == (2, 256)
    assert torch.isfinite(embedding).all()
    assert torch.isfinite(detail_dynamic).all()
    torch.testing.assert_close(detailed_embedding, embedding, rtol=0, atol=0)


def test_dynamic_attention_receives_three_full_layer2_transition_grids():
    encoder = DynamicDetailV2Encoder().eval()
    observed = {}

    def capture_attention_inputs(_module, inputs, _output):
        observed["query"] = inputs[0].shape
        observed["key"] = inputs[1].shape
        observed["value"] = inputs[2].shape

    handle = encoder.dynamic_attention.register_forward_hook(capture_attention_inputs)
    try:
        with torch.no_grad():
            encoder(torch.randn(1, 4, 3, 256, 256))
    finally:
        handle.remove()

    # ResNet-18 layer2 retains a 32x32 grid at 256 input. Four frames create
    # three adjacent transitions, so dynamic attention must see 3 * 1024 sites.
    assert observed == {
        "query": torch.Size([1, 1, 256]),
        "key": torch.Size([1, 3 * 1024, 256]),
        "value": torch.Size([1, 3 * 1024, 256]),
    }
    assert encoder.temporal_encoding.shape == (3, 256)


def test_projection_precedes_temporal_subtraction_for_nonlinear_tokens():
    torch.manual_seed(5)
    encoder = DynamicDetailV2Encoder().eval()
    observed = {}

    def capture_projection(_module, inputs, output):
        observed["projection_input"] = inputs[0].detach().clone()
        observed["projected"] = output.detach().clone()

    def capture_dynamic_norm_input(_module, inputs):
        observed["dynamic_norm_input"] = inputs[0].detach().clone()

    projection_handle = encoder.local_projection.register_forward_hook(
        capture_projection
    )
    norm_handle = encoder.dynamic_token_norm.register_forward_pre_hook(
        capture_dynamic_norm_input
    )
    base = torch.randn(1, 1, 3, 64, 64)
    frame_scales = torch.tensor([0.5, 1.0, 1.75, 2.5]).view(1, 4, 1, 1, 1)
    inputs = base * frame_scales
    try:
        with torch.no_grad():
            encoder(inputs)
    finally:
        projection_handle.remove()
        norm_handle.remove()

    projection_input = observed["projection_input"]
    projected = observed["projected"]
    assert projection_input.shape == (1, 4, 64, 128)
    assert projected.shape == (1, 4, 64, 256)
    expected_delta = projected[:, 1:] - projected[:, :-1]
    torch.testing.assert_close(
        observed["dynamic_norm_input"], expected_delta, rtol=0, atol=0
    )

    # Linear + LayerNorm is nonlinear as a whole. Applying it after subtraction
    # yields a measurably different tensor and must not describe the dynamic path.
    with torch.no_grad():
        projected_after_subtraction = encoder.local_projection(
            projection_input[:, 1:] - projection_input[:, :-1]
        )
    assert not torch.allclose(expected_delta, projected_after_subtraction)


def test_dynamic_gate_starts_at_sigmoid_point_fifteen():
    encoder = DynamicDetailV2Encoder()

    torch.testing.assert_close(
        torch.sigmoid(encoder.gate_logit.detach()),
        torch.tensor(0.15),
        rtol=0,
        atol=1e-7,
    )


def test_backward_reaches_static_dynamic_gate_and_shared_backbone():
    torch.manual_seed(17)
    encoder = DynamicDetailV2Encoder().train()
    inputs = torch.randn(2, 4, 3, 64, 64)

    embedding = encoder(inputs)
    embedding_weight = torch.randn_like(embedding)
    loss = (embedding * embedding_weight).mean()
    loss.backward()

    expected_nonzero = {
        "static_attention": encoder.cross_attention.in_proj_weight.grad,
        "static_tokens": encoder.local_projection[0].weight.grad,
        "dynamic_attention": encoder.dynamic_attention.in_proj_weight.grad,
        "dynamic_tokens": encoder.dynamic_token_norm.weight.grad,
        "temporal_encoding": encoder.temporal_encoding.grad,
        "gate": encoder.gate_logit.grad,
        "backbone_early": encoder.trunk.conv1.weight.grad,
        "backbone_late": encoder.trunk.layer4[-1].conv2.weight.grad,
    }
    for name, gradient in expected_nonzero.items():
        assert gradient is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(gradient).all(), f"{name} gradient is not finite"
        assert torch.count_nonzero(gradient), f"{name} gradient is identically zero"


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (torch.randn(2, 3, 64, 64), "five-dimensional"),
        (torch.randn(2, 3, 3, 64, 64), "exactly 4 frames"),
        (torch.randn(2, 4, 1, 64, 64), "exactly 3 channels"),
    ],
)
def test_dynamic_encoder_rejects_invalid_sequences(inputs, match):
    with pytest.raises(ValueError, match=match):
        DynamicDetailV2Encoder()(inputs)


def test_spatial_v2_warm_start_consumes_every_source_key_exactly():
    torch.manual_seed(23)
    spatial = build_encoder("detail_v2")
    source_checkpoint = export_encoder_checkpoint(spatial, PREPROCESS)

    dynamic, report = warm_start_dynamic_encoder(source_checkpoint)

    source_state = spatial.state_dict()
    dynamic_state = dynamic.state_dict()
    compatible_keys = sorted(source_state)
    new_keys = [
        "dynamic_attention.in_proj_bias",
        "dynamic_attention.in_proj_weight",
        "dynamic_attention.out_proj.bias",
        "dynamic_attention.out_proj.weight",
        "dynamic_projection.0.bias",
        "dynamic_projection.0.weight",
        "dynamic_token_norm.bias",
        "dynamic_token_norm.weight",
        "fusion_norm.bias",
        "fusion_norm.weight",
        "gate_logit",
        "temporal_encoding",
    ]
    assert report == {
        "source_encoder_type": "detail_v2",
        "target_encoder_type": "detail_v2_dynamic",
        "compatible_keys": compatible_keys,
        "new_keys": new_keys,
        "compatible_key_count": len(compatible_keys),
        "new_key_count": len(new_keys),
    }
    assert compatible_keys
    assert new_keys
    assert set(source_state).issubset(dynamic_state)
    for key, source_tensor in source_state.items():
        assert dynamic_state[key].shape == source_tensor.shape
        assert dynamic_state[key].dtype == source_tensor.dtype
        torch.testing.assert_close(dynamic_state[key], source_tensor, rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["shape", "dtype"])
def test_spatial_v2_warm_start_rejects_incompatible_tensor_contract(mutation):
    checkpoint = export_encoder_checkpoint(build_encoder("detail_v2"), PREPROCESS)
    checkpoint = copy.deepcopy(checkpoint)
    key = "local_projection.0.weight"
    if mutation == "shape":
        checkpoint["encoder_state"][key] = checkpoint["encoder_state"][key][:-1]
    else:
        checkpoint["encoder_state"][key] = checkpoint["encoder_state"][key].to(
            torch.float64
        )

    with pytest.raises(RuntimeError, match=f"{mutation} mismatch.*{key}"):
        warm_start_dynamic_encoder(checkpoint)


def test_warm_start_rejects_non_spatial_v2_checkpoint():
    checkpoint = export_encoder_checkpoint(build_encoder("original"), PREPROCESS)

    with pytest.raises(ValueError, match="detail_v2"):
        warm_start_dynamic_encoder(checkpoint)


def test_dynamic_checkpoint_roundtrips_strictly(tmp_path: Path):
    torch.manual_seed(29)
    encoder = build_encoder("detail_v2_dynamic").eval()
    inputs = torch.randn(2, 4, 3, 64, 64)
    with torch.no_grad():
        expected_embedding, expected_dynamic = encoder.forward_with_details(inputs)

    path = tmp_path / "dynamic_encoder.pt"
    save_encoder_checkpoint(path, encoder, PREPROCESS)
    loaded, metadata = load_encoder_checkpoint(
        path,
        expected_encoder_type="detail_v2_dynamic",
        expected_preprocess=PREPROCESS,
    )
    loaded.eval()
    with torch.no_grad():
        actual_embedding, actual_dynamic = loaded.forward_with_details(inputs)

    torch.testing.assert_close(actual_embedding, expected_embedding, rtol=0, atol=0)
    torch.testing.assert_close(actual_dynamic, expected_dynamic, rtol=0, atol=0)
    assert metadata == {
        "encoder_type": "detail_v2_dynamic",
        "structure": {
            "backbone": "resnet18",
            "input_channels": 3,
            "latent_dims": 512,
            "token_dim": 256,
            "num_heads": 4,
            "attention_dropout": 0.0,
            "sequence_length": 4,
            "temporal_stride": 1,
            "initial_gate": 0.15,
        },
        "preprocess": PREPROCESS,
    }

    malformed = export_encoder_checkpoint(encoder, PREPROCESS)
    malformed["encoder_state"].pop("gate_logit")
    with pytest.raises(RuntimeError, match="state_dict"):
        load_encoder_checkpoint(malformed)

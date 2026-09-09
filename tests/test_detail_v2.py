import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from encoder.detail_v2 import (
    DetailV2Encoder,
    OriginalEncoder,
    build_2d_sincos_position_embedding,
    build_encoder,
    export_encoder_checkpoint,
    load_encoder_checkpoint,
    save_encoder_checkpoint,
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


@pytest.mark.parametrize("encoder_type", ["original", "detail_v2"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_encoder_outputs_finite_512_vector_for_supported_batches(
    encoder_type, batch_size
):
    encoder = build_encoder(encoder_type).eval()
    inputs = torch.randn(batch_size, 3, 64, 64)

    with torch.no_grad():
        outputs = encoder(inputs)

    assert outputs.shape == (batch_size, 512)
    assert torch.isfinite(outputs).all()


def test_detail_v2_preserves_layer2_grid_for_attention_tokens():
    encoder = build_encoder("detail_v2").eval()
    observed = {}

    def capture_projection_shape(_module, inputs, output):
        observed["input"] = inputs[0].shape
        observed["output"] = output.shape

    handle = encoder.local_projection.register_forward_hook(capture_projection_shape)
    try:
        with torch.no_grad():
            encoder(torch.randn(1, 3, 64, 64))
    finally:
        handle.remove()

    # ResNet-18 layer2 is an 8x8 grid for a 64x64 input. Every one of its
    # 64 sites must reach the token projection; a V1-style 4x4 pool would fail.
    assert observed["input"] == torch.Size([1, 64, 128])
    assert observed["output"] == torch.Size([1, 64, 256])


def test_position_embedding_is_fixed_2d_and_coordinate_sensitive():
    position = build_2d_sincos_position_embedding(
        height=2,
        width=3,
        embed_dim=256,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    repeated = build_2d_sincos_position_embedding(
        height=2,
        width=3,
        embed_dim=256,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert position.shape == (1, 6, 256)
    torch.testing.assert_close(position, repeated, rtol=0, atol=0)
    assert not position.requires_grad
    assert not torch.equal(position[:, 0], position[:, 1])  # x coordinate
    assert not torch.equal(position[:, 0], position[:, 3])  # y coordinate


def test_backward_reaches_semantic_detail_qkv_and_shared_backbone():
    torch.manual_seed(7)
    encoder = DetailV2Encoder().train()
    outputs = encoder(torch.randn(2, 3, 64, 64))

    outputs.square().mean().backward()

    expected_nonzero = {
        "semantic": encoder.semantic_projection[0].weight.grad,
        "detail_tokens": encoder.local_projection[0].weight.grad,
        "global_query": encoder.global_query.weight.grad,
        "backbone_early": encoder.trunk.conv1.weight.grad,
        "backbone_late": encoder.trunk.layer4[-1].conv2.weight.grad,
    }
    for name, gradient in expected_nonzero.items():
        assert gradient is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(gradient).all(), f"{name} gradient is not finite"
        assert torch.count_nonzero(gradient), f"{name} gradient is identically zero"

    qkv_gradient = encoder.cross_attention.in_proj_weight.grad
    assert qkv_gradient is not None
    assert qkv_gradient.shape == (3 * 256, 256)
    for name, gradient_slice in zip("qkv", qkv_gradient.chunk(3, dim=0)):
        assert torch.isfinite(gradient_slice).all(), f"{name} gradient is not finite"
        assert torch.count_nonzero(gradient_slice), f"{name} gradient is zero"


def test_original_and_detail_v2_can_reuse_identical_shared_trunk_initialization():
    original = OriginalEncoder()
    shared_state = original.export_trunk_state()
    detail_v2 = build_encoder("detail_v2", trunk_state=shared_state)

    assert shared_state
    assert all(not key.startswith("fc.") for key in shared_state)
    for key, tensor in detail_v2.export_trunk_state().items():
        torch.testing.assert_close(tensor, shared_state[key], rtol=0, atol=0)
    assert isinstance(original.trunk.fc, nn.Linear)
    assert original.trunk.fc.in_features == 512
    assert original.trunk.fc.out_features == 512


def test_shared_trunk_initialization_rejects_incomplete_state():
    shared_state = OriginalEncoder().export_trunk_state()
    shared_state.pop(next(iter(shared_state)))

    with pytest.raises(RuntimeError, match="shared trunk state"):
        build_encoder("detail_v2", trunk_state=shared_state)


def test_checkpoint_contract_roundtrips_exact_eval_embedding(tmp_path: Path):
    torch.manual_seed(11)
    encoder = build_encoder("detail_v2").eval()
    inputs = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected = encoder(inputs)

    path = tmp_path / "encoder.pth"
    save_encoder_checkpoint(path, encoder, PREPROCESS)
    loaded, metadata = load_encoder_checkpoint(
        path,
        expected_encoder_type="detail_v2",
        expected_preprocess=PREPROCESS,
    )
    loaded.eval()
    with torch.no_grad():
        actual = loaded(inputs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert metadata == {
        "encoder_type": "detail_v2",
        "structure": {
            "backbone": "resnet18",
            "input_channels": 3,
            "latent_dims": 512,
            "token_dim": 256,
            "num_heads": 4,
            "attention_dropout": 0.0,
        },
        "preprocess": PREPROCESS,
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda checkpoint: checkpoint.pop("preprocess"), "checkpoint fields"),
        (lambda checkpoint: checkpoint.__setitem__("decoder_state", {}), "checkpoint fields"),
        (
            lambda checkpoint: checkpoint["structure"].__setitem__("num_heads", 8),
            "unsupported detail_v2 structure",
        ),
        (
            lambda checkpoint: checkpoint["encoder_state"].pop(
                next(iter(checkpoint["encoder_state"]))
            ),
            "state_dict",
        ),
    ],
)
def test_checkpoint_loader_rejects_malformed_or_non_encoder_payload(mutation, match):
    checkpoint = export_encoder_checkpoint(build_encoder("detail_v2"), PREPROCESS)
    mutation(checkpoint)

    with pytest.raises((KeyError, ValueError, RuntimeError), match=match):
        load_encoder_checkpoint(checkpoint)


def test_checkpoint_loader_rejects_expected_contract_mismatch():
    checkpoint = export_encoder_checkpoint(build_encoder("detail_v2"), PREPROCESS)

    with pytest.raises(ValueError, match="encoder_type mismatch"):
        load_encoder_checkpoint(checkpoint, expected_encoder_type="original")

    wrong_preprocess = copy.deepcopy(PREPROCESS)
    wrong_preprocess["scale"] = "none"
    with pytest.raises(ValueError, match="preprocess mismatch"):
        load_encoder_checkpoint(checkpoint, expected_preprocess=wrong_preprocess)


def test_factory_rejects_unknown_encoder_type():
    with pytest.raises(ValueError, match="unsupported encoder_type"):
        build_encoder("detail_v1")

import copy
import math
from pathlib import Path

import pytest
import torch

from encoder.detail_v2 import (
    DetailV2Encoder,
    build_encoder,
    export_encoder_checkpoint,
    load_encoder_checkpoint,
    save_encoder_checkpoint,
)
from encoder.tra_v2 import (
    TRADetailV2Encoder,
    measure_spatial_warm_start_sanity,
    warm_start_tra_encoder,
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


def _spatial_checkpoint(seed: int = 101):
    torch.manual_seed(seed)
    spatial = DetailV2Encoder().eval()
    return spatial, export_encoder_checkpoint(spatial, PREPROCESS)


def test_factory_builds_canonical_tra_without_dynamic_v2_modules():
    encoder = build_encoder("detail_v2_tra")

    assert isinstance(encoder, TRADetailV2Encoder)
    assert encoder.structure == {
        "backbone": "resnet18",
        "input_channels": 3,
        "latent_dims": 512,
        "token_dim": 256,
        "num_heads": 4,
        "attention_dropout": 0.0,
        "sequence_length": 4,
        "temporal_stride": 1,
        "initial_temporal_gate": 0.1,
    }
    assert encoder.gru.input_size == 256
    assert encoder.gru.hidden_size == 256
    assert encoder.gru.num_layers == 1
    assert encoder.gru.dropout == 0
    assert not any(
        hasattr(encoder, name)
        for name in (
            "dynamic_attention",
            "dynamic_projection",
            "dynamic_token_norm",
            "temporal_encoding",
            "gate_logit",
        )
    )
    assert torch.sigmoid(encoder.alpha.detach()).item() == pytest.approx(0.1)


def test_tra_uses_current_global_query_for_four_spatial_attention_calls():
    encoder = TRADetailV2Encoder().eval()
    observed_queries = []
    layer3_batches = []

    def capture_attention(_module, inputs, _output):
        observed_queries.append(inputs[0].detach().clone())

    def capture_layer3(_module, inputs, _output):
        layer3_batches.append(inputs[0].shape[0])

    attention_handle = encoder.cross_attention.register_forward_hook(capture_attention)
    layer3_handle = encoder.trunk.layer3.register_forward_hook(capture_layer3)
    try:
        with torch.no_grad():
            output, details = encoder.forward_with_details(torch.randn(2, 4, 3, 64, 64))
    finally:
        attention_handle.remove()
        layer3_handle.remove()

    assert output.shape == (2, 512)
    assert details["spatial_sequence"].shape == (2, 4, 256)
    assert len(observed_queries) == 4
    for query in observed_queries[1:]:
        torch.testing.assert_close(query, observed_queries[0], rtol=0, atol=0)
    assert layer3_batches == [2]


def test_forward_spatial_details_exposes_semantic_and_four_current_query_details():
    encoder = TRADetailV2Encoder().eval()
    inputs = torch.randn(2, 4, 3, 64, 64)

    with torch.no_grad():
        semantic, spatial_details = encoder.forward_spatial_details(inputs)
        _, introspection = encoder.forward_with_details(inputs)

    assert semantic.shape == (2, 256)
    assert spatial_details.shape == (2, 4, 256)
    torch.testing.assert_close(semantic, introspection["semantic"], rtol=0, atol=0)
    torch.testing.assert_close(
        spatial_details, introspection["spatial_sequence"], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (torch.randn(2, 3, 64, 64), "five-dimensional"),
        (torch.randn(2, 3, 3, 64, 64), "exactly 4 frames"),
        (torch.randn(2, 4, 1, 64, 64), "exactly 3 channels"),
    ],
)
def test_tra_rejects_noncanonical_input_windows(inputs, match):
    with pytest.raises(ValueError, match=match):
        TRADetailV2Encoder()(inputs)


def test_warm_start_consumes_all_spatial_keys_and_only_leaves_tra_keys_new():
    spatial, checkpoint = _spatial_checkpoint()

    tra, report = warm_start_tra_encoder(checkpoint)

    source_state = spatial.state_dict()
    target_state = tra.state_dict()
    expected_new_keys = [
        "alpha",
        "fusion_norm.bias",
        "fusion_norm.weight",
        "gru.bias_hh_l0",
        "gru.bias_ih_l0",
        "gru.weight_hh_l0",
        "gru.weight_ih_l0",
        "temporal_mlp.0.bias",
        "temporal_mlp.0.weight",
        "temporal_mlp.2.bias",
        "temporal_mlp.2.weight",
    ]
    assert report == {
        "source_encoder_type": "detail_v2",
        "target_encoder_type": "detail_v2_tra",
        "compatible_keys": sorted(source_state),
        "new_keys": expected_new_keys,
        "compatible_key_count": len(source_state),
        "new_key_count": len(expected_new_keys),
    }
    for key, expected in source_state.items():
        torch.testing.assert_close(target_state[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["shape", "dtype"])
def test_tra_warm_start_rejects_incompatible_spatial_tensor(mutation):
    _, checkpoint = _spatial_checkpoint()
    checkpoint = copy.deepcopy(checkpoint)
    key = "local_projection.0.weight"
    if mutation == "shape":
        checkpoint["encoder_state"][key] = checkpoint["encoder_state"][key][:-1]
    else:
        checkpoint["encoder_state"][key] = checkpoint["encoder_state"][key].double()

    with pytest.raises(RuntimeError, match=f"{mutation} mismatch.*{key}"):
        warm_start_tra_encoder(checkpoint)


def test_gate_zero_sanity_is_exact_before_fusion_and_quantifies_layernorm_delta():
    spatial, checkpoint = _spatial_checkpoint(seed=103)
    tra, _ = warm_start_tra_encoder(checkpoint)
    inputs = torch.randn(2, 4, 3, 64, 64)

    report = measure_spatial_warm_start_sanity(spatial, tra, inputs)

    assert report["semantic_max_abs_error"] <= 1e-7
    assert report["current_detail_max_abs_error"] <= 1e-7
    assert 0.0 <= report["final_detail_cosine_similarity"] <= 1.0
    assert math.isfinite(report["final_detail_relative_l2_error"])
    assert report["final_detail_relative_l2_error"] > 0.0
    assert report["gate_value"] < 1e-8


def test_gru_and_alpha_receive_gradients_and_alpha_updates():
    torch.manual_seed(107)
    encoder = TRADetailV2Encoder().train()
    optimizer = torch.optim.SGD([encoder.alpha, *encoder.gru.parameters()], lr=0.1)
    alpha_before = encoder.alpha.detach().clone()

    output = encoder(torch.randn(2, 4, 3, 64, 64))
    loss = (output * torch.randn_like(output)).mean()
    loss.backward()

    assert encoder.gru.weight_ih_l0.grad is not None
    assert torch.count_nonzero(encoder.gru.weight_ih_l0.grad)
    assert encoder.alpha.grad is not None
    assert torch.count_nonzero(encoder.alpha.grad)
    optimizer.step()
    assert not torch.equal(encoder.alpha.detach(), alpha_before)


def test_tra_checkpoint_roundtrips_with_strict_metadata(tmp_path: Path):
    torch.manual_seed(109)
    encoder = build_encoder("detail_v2_tra").eval()
    inputs = torch.randn(2, 4, 3, 64, 64)
    with torch.no_grad():
        expected = encoder(inputs)

    path = tmp_path / "tra_encoder.pt"
    save_encoder_checkpoint(path, encoder, PREPROCESS)
    loaded, metadata = load_encoder_checkpoint(
        path,
        expected_encoder_type="detail_v2_tra",
        expected_preprocess=PREPROCESS,
    )
    loaded.eval()
    with torch.no_grad():
        actual = loaded(inputs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert metadata == {
        "encoder_type": "detail_v2_tra",
        "structure": encoder.structure,
        "preprocess": PREPROCESS,
    }

    malformed = export_encoder_checkpoint(encoder, PREPROCESS)
    malformed["structure"]["initial_temporal_gate"] = 0.15
    with pytest.raises(ValueError, match="unsupported detail_v2_tra structure"):
        load_encoder_checkpoint(malformed)

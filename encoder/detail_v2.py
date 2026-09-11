"""Canonical single-frame tactile encoders and their checkpoint contract.

This module intentionally contains encoders only. Pretraining decoders belong to
the training pipeline and must never be serialized as part of an encoder
checkpoint.
"""

from __future__ import annotations

import copy
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchvision import models


_CHECKPOINT_FIELDS = {
    "encoder_type",
    "structure",
    "preprocess",
    "encoder_state",
}
_DETAIL_V2_STRUCTURE = {
    "backbone": "resnet18",
    "input_channels": 3,
    "latent_dims": 512,
    "token_dim": 256,
    "num_heads": 4,
    "attention_dropout": 0.0,
}
_DYNAMIC_DETAIL_V2_STRUCTURE = {
    **_DETAIL_V2_STRUCTURE,
    "sequence_length": 4,
    "temporal_stride": 1,
    "initial_gate": 0.15,
}
_DYNAMIC_TEMPORAL_STRIDES = (1, 2)
_DYNAMIC_NEW_STATE_KEYS = {
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
}
_ORIGINAL_STRUCTURE = {
    "backbone": "resnet18",
    "input_channels": 3,
    "latent_dims": 512,
}


def _validate_base_structure(backbone: str, input_channels: int, latent_dims: int) -> None:
    if backbone != "resnet18" or input_channels != 3 or latent_dims != 512:
        raise ValueError(
            "unsupported encoder structure: expected resnet18, three input "
            "channels, and a 512-D output"
        )


def _validate_dynamic_structure(structure: Mapping[str, Any]) -> None:
    candidate = dict(structure)
    temporal_stride = candidate.get("temporal_stride")
    initial_gate = candidate.get("initial_gate")
    candidate["temporal_stride"] = _DYNAMIC_DETAIL_V2_STRUCTURE["temporal_stride"]
    candidate["initial_gate"] = _DYNAMIC_DETAIL_V2_STRUCTURE["initial_gate"]
    if (
        type(temporal_stride) is not int
        or temporal_stride not in _DYNAMIC_TEMPORAL_STRIDES
        or not isinstance(initial_gate, (int, float))
        or not math.isfinite(float(initial_gate))
        or not 0.0 < float(initial_gate) < 1.0
        or candidate != _DYNAMIC_DETAIL_V2_STRUCTURE
    ):
        raise ValueError(
            "unsupported detail_v2_dynamic structure; expected the canonical "
            "structure with temporal_stride in (1, 2) and 0 < initial_gate < 1, got "
            f"{dict(structure)}"
        )


def _make_resnet18(*, weights: Any, latent_dims: int) -> nn.Module:
    # Do not supply a custom norm layer: the project protocol requires ordinary
    # torchvision BatchNorm2d, including its running statistics.
    if weights is None:
        return models.resnet18(weights=None, num_classes=latent_dims)
    trunk = models.resnet18(weights=weights)
    trunk.fc = nn.Linear(trunk.fc.in_features, latent_dims)
    return trunk


class _SharedResNetTrunkEncoder(nn.Module):
    """Common strict import/export behavior for the convolutional ResNet trunk."""

    trunk: nn.Module

    def export_trunk_state(self) -> dict[str, Tensor]:
        return {
            key: tensor.detach().clone()
            for key, tensor in self.trunk.state_dict().items()
            if not key.startswith("fc.")
        }

    def _load_shared_trunk_state(self, state: Mapping[str, Tensor]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("shared trunk state must be a mapping")

        current = self.trunk.state_dict()
        expected_keys = {key for key in current if not key.startswith("fc.")}
        supplied_keys = set(state)
        if supplied_keys != expected_keys:
            missing = sorted(expected_keys - supplied_keys)
            unexpected = sorted(supplied_keys - expected_keys)
            raise RuntimeError(
                "shared trunk state keys do not match exactly; "
                f"missing={missing}, unexpected={unexpected}"
            )

        merged = dict(current)
        merged.update(state)
        try:
            self.trunk.load_state_dict(merged, strict=True)
        except RuntimeError as error:
            raise RuntimeError(f"invalid shared trunk state: {error}") from error


class OriginalEncoder(_SharedResNetTrunkEncoder):
    """B0 ResNet-18 with its original 512-output fully connected layer."""

    encoder_type = "original"

    def __init__(
        self,
        *,
        backbone: str = "resnet18",
        input_channels: int = 3,
        latent_dims: int = 512,
        weights: Any = None,
        trunk_state: Mapping[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        _validate_base_structure(backbone, input_channels, latent_dims)
        self.structure = copy.deepcopy(_ORIGINAL_STRUCTURE)
        self.trunk = _make_resnet18(weights=weights, latent_dims=latent_dims)
        if trunk_state is not None:
            self._load_shared_trunk_state(trunk_state)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.trunk(inputs)


def _sincos_1d(positions: Tensor, embed_dim: int) -> Tensor:
    if embed_dim % 2:
        raise ValueError("one-dimensional sin/cos embedding size must be even")
    frequencies = torch.arange(
        embed_dim // 2,
        device=positions.device,
        dtype=torch.float32,
    )
    frequencies = 1.0 / (10000 ** (frequencies / (embed_dim // 2)))
    angles = positions.to(torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((angles.sin(), angles.cos()), dim=1)


def build_2d_sincos_position_embedding(
    *,
    height: int,
    width: int,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a deterministic row-major [1, H*W, D] 2-D sin/cos grid."""
    if height <= 0 or width <= 0:
        raise ValueError("position grid dimensions must be positive")
    if embed_dim % 4:
        raise ValueError("2-D sin/cos embedding size must be divisible by four")

    y_coordinates, x_coordinates = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    axis_dim = embed_dim // 2
    y_embedding = _sincos_1d(y_coordinates.reshape(-1), axis_dim)
    x_embedding = _sincos_1d(x_coordinates.reshape(-1), axis_dim)
    return torch.cat((y_embedding, x_embedding), dim=1).to(dtype=dtype).unsqueeze(0)


class DetailV2Encoder(_SharedResNetTrunkEncoder):
    """ResNet-18 encoder with full-grid layer2 cross-scale attention."""

    encoder_type = "detail_v2"

    def __init__(
        self,
        *,
        backbone: str = "resnet18",
        input_channels: int = 3,
        latent_dims: int = 512,
        token_dim: int = 256,
        num_heads: int = 4,
        attention_dropout: float = 0.0,
        weights: Any = None,
        trunk_state: Mapping[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        structure = {
            "backbone": backbone,
            "input_channels": input_channels,
            "latent_dims": latent_dims,
            "token_dim": token_dim,
            "num_heads": num_heads,
            "attention_dropout": float(attention_dropout),
        }
        if structure != _DETAIL_V2_STRUCTURE:
            raise ValueError(
                "unsupported detail_v2 structure; expected "
                f"{_DETAIL_V2_STRUCTURE}, got {structure}"
            )

        self.structure = copy.deepcopy(structure)
        self.trunk = _make_resnet18(weights=weights, latent_dims=latent_dims)
        self.trunk.fc = nn.Identity()

        self.local_projection = nn.Sequential(
            nn.Linear(128, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.semantic_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.semantic_projection = nn.Sequential(
            nn.Linear(512, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.global_query = nn.Linear(512, token_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.detail_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )

        if trunk_state is not None:
            self._load_shared_trunk_state(trunk_state)

    def forward_features(self, inputs: Tensor) -> dict[str, Tensor]:
        features = self.trunk.conv1(inputs)
        features = self.trunk.bn1(features)
        features = self.trunk.relu(features)
        features = self.trunk.maxpool(features)
        layer1 = self.trunk.layer1(features)
        layer2 = self.trunk.layer2(layer1)
        layer3 = self.trunk.layer3(layer2)
        layer4 = self.trunk.layer4(layer3)
        return {"layer2": layer2, "layer4": layer4}

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.forward_features(inputs)
        layer2 = features["layer2"]
        batch_size, _, height, width = layer2.shape
        local_tokens = layer2.flatten(2).transpose(1, 2)
        local_tokens = self.local_projection(local_tokens)
        local_tokens = local_tokens + build_2d_sincos_position_embedding(
            height=height,
            width=width,
            embed_dim=self.structure["token_dim"],
            device=local_tokens.device,
            dtype=local_tokens.dtype,
        )

        global_feature = self.semantic_pool(features["layer4"]).reshape(batch_size, 512)
        semantic = self.semantic_projection(global_feature)
        query = self.global_query(global_feature).unsqueeze(1)
        detail, _ = self.cross_attention(
            query,
            local_tokens,
            local_tokens,
            need_weights=False,
        )
        detail = self.detail_projection(detail.squeeze(1))
        return torch.cat((semantic, detail), dim=1)


class DynamicDetailV2Encoder(DetailV2Encoder):
    """Details V2 with a shared-trunk, causal local-dynamics branch."""

    encoder_type = "detail_v2_dynamic"

    def __init__(
        self,
        *,
        backbone: str = "resnet18",
        input_channels: int = 3,
        latent_dims: int = 512,
        token_dim: int = 256,
        num_heads: int = 4,
        attention_dropout: float = 0.0,
        sequence_length: int = 4,
        temporal_stride: int = 1,
        initial_gate: float = 0.15,
        weights: Any = None,
        trunk_state: Mapping[str, Tensor] | None = None,
    ) -> None:
        structure = {
            "backbone": backbone,
            "input_channels": input_channels,
            "latent_dims": latent_dims,
            "token_dim": token_dim,
            "num_heads": num_heads,
            "attention_dropout": float(attention_dropout),
            "sequence_length": sequence_length,
            "temporal_stride": temporal_stride,
            "initial_gate": float(initial_gate),
        }
        _validate_dynamic_structure(structure)

        super().__init__(
            backbone=backbone,
            input_channels=input_channels,
            latent_dims=latent_dims,
            token_dim=token_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            weights=weights,
            trunk_state=trunk_state,
        )
        self.structure = copy.deepcopy(structure)
        transition_count = sequence_length - 1
        self.dynamic_token_norm = nn.LayerNorm(token_dim)
        self.temporal_encoding = nn.Parameter(
            torch.zeros(transition_count, token_dim)
        )
        self.dynamic_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dynamic_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.gate_logit = nn.Parameter(
            torch.tensor(math.log(initial_gate / (1.0 - initial_gate)))
        )
        self.fusion_norm = nn.LayerNorm(token_dim)

    def _validate_sequence(self, inputs: Tensor) -> None:
        if inputs.ndim != 5:
            raise ValueError(
                "dynamic tactile input must be five-dimensional [B,T,3,H,W]"
            )
        if inputs.shape[1] != self.structure["sequence_length"]:
            raise ValueError(
                "dynamic tactile input must contain exactly 4 frames"
            )
        if inputs.shape[2] != self.structure["input_channels"]:
            raise ValueError(
                "dynamic tactile input frames must contain exactly 3 channels"
            )

    def forward_with_details(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_sequence(inputs)
        batch_size, sequence_length, channels, height, width = inputs.shape
        features = inputs.reshape(batch_size * sequence_length, channels, height, width)
        features = self.trunk.conv1(features)
        features = self.trunk.bn1(features)
        features = self.trunk.relu(features)
        features = self.trunk.maxpool(features)
        layer1 = self.trunk.layer1(features)
        layer2_flat = self.trunk.layer2(layer1)
        _, layer2_channels, grid_height, grid_width = layer2_flat.shape
        layer2_sequence = layer2_flat.reshape(
            batch_size,
            sequence_length,
            layer2_channels,
            grid_height,
            grid_width,
        )

        current_layer2 = layer2_sequence[:, -1]
        layer3 = self.trunk.layer3(current_layer2)
        layer4 = self.trunk.layer4(layer3)
        global_feature = self.semantic_pool(layer4).reshape(batch_size, 512)
        semantic = self.semantic_projection(global_feature)
        query = self.global_query(global_feature).unsqueeze(1)

        position = build_2d_sincos_position_embedding(
            height=grid_height,
            width=grid_width,
            embed_dim=self.structure["token_dim"],
            device=layer2_flat.device,
            dtype=layer2_flat.dtype,
        )
        layer2_tokens = layer2_sequence.flatten(3).permute(0, 1, 3, 2)
        projected_tokens = self.local_projection(layer2_tokens)
        static_tokens = projected_tokens[:, -1] + position
        detail_static, _ = self.cross_attention(
            query,
            static_tokens,
            static_tokens,
            need_weights=False,
        )
        detail_static = self.detail_projection(detail_static.squeeze(1))

        residual_tokens = projected_tokens[:, 1:] - projected_tokens[:, :-1]
        residual_tokens = self.dynamic_token_norm(residual_tokens)
        residual_tokens = residual_tokens + position.unsqueeze(1)
        residual_tokens = residual_tokens + self.temporal_encoding.view(
            1,
            sequence_length - 1,
            1,
            self.structure["token_dim"],
        )
        residual_tokens = residual_tokens.flatten(1, 2)
        detail_dynamic, _ = self.dynamic_attention(
            query,
            residual_tokens,
            residual_tokens,
            need_weights=False,
        )
        detail_dynamic = self.dynamic_projection(detail_dynamic.squeeze(1))

        detail = self.fusion_norm(
            detail_static + torch.sigmoid(self.gate_logit) * detail_dynamic
        )
        return torch.cat((semantic, detail), dim=1), detail_dynamic

    def forward(self, inputs: Tensor) -> Tensor:
        embedding, _ = self.forward_with_details(inputs)
        return embedding


def build_encoder(
    encoder_type: str,
    *,
    backbone: str = "resnet18",
    input_channels: int = 3,
    latent_dims: int = 512,
    token_dim: int = 256,
    num_heads: int = 4,
    attention_dropout: float = 0.0,
    sequence_length: int = 4,
    temporal_stride: int = 1,
    initial_gate: float = 0.15,
    initial_temporal_gate: float = 0.1,
    weights: Any = None,
    trunk_state: Mapping[str, Tensor] | None = None,
) -> OriginalEncoder | DetailV2Encoder | DynamicDetailV2Encoder:
    """Build a canonical B0, Spatial Details V2, or Dynamic Details V2 encoder."""
    if encoder_type == "original":
        return OriginalEncoder(
            backbone=backbone,
            input_channels=input_channels,
            latent_dims=latent_dims,
            weights=weights,
            trunk_state=trunk_state,
        )
    if encoder_type == "detail_v2":
        return DetailV2Encoder(
            backbone=backbone,
            input_channels=input_channels,
            latent_dims=latent_dims,
            token_dim=token_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            weights=weights,
            trunk_state=trunk_state,
        )
    if encoder_type == "detail_v2_dynamic":
        return DynamicDetailV2Encoder(
            backbone=backbone,
            input_channels=input_channels,
            latent_dims=latent_dims,
            token_dim=token_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            sequence_length=sequence_length,
            temporal_stride=temporal_stride,
            initial_gate=initial_gate,
            weights=weights,
            trunk_state=trunk_state,
        )
    if encoder_type == "detail_v2_tra":
        from .tra_v2 import TRADetailV2Encoder

        return TRADetailV2Encoder(
            backbone=backbone,
            input_channels=input_channels,
            latent_dims=latent_dims,
            token_dim=token_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            sequence_length=sequence_length,
            temporal_stride=temporal_stride,
            initial_temporal_gate=initial_temporal_gate,
            weights=weights,
            trunk_state=trunk_state,
        )
    raise ValueError(f"unsupported encoder_type: {encoder_type}")


def warm_start_dynamic_encoder(
    source: str | os.PathLike[str] | Mapping[str, Any],
    **structure: Any,
) -> tuple[DynamicDetailV2Encoder, dict[str, Any]]:
    """Warm-start Dynamic Details V2 from a strictly validated Spatial V2 checkpoint."""
    checkpoint = _read_checkpoint(source, "cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    actual_fields = set(checkpoint)
    if actual_fields != _CHECKPOINT_FIELDS:
        missing = sorted(_CHECKPOINT_FIELDS - actual_fields)
        unexpected = sorted(actual_fields - _CHECKPOINT_FIELDS)
        raise KeyError(
            "checkpoint fields do not match contract; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if checkpoint["encoder_type"] != "detail_v2":
        raise ValueError("dynamic warm-start source must be a detail_v2 checkpoint")
    source_structure = checkpoint["structure"]
    if (
        not isinstance(source_structure, Mapping)
        or dict(source_structure) != _DETAIL_V2_STRUCTURE
    ):
        raise ValueError(
            "dynamic warm-start source must use the canonical detail_v2 structure"
        )
    if not isinstance(checkpoint["preprocess"], Mapping):
        raise TypeError("checkpoint preprocess must be a mapping")
    source_state = checkpoint["encoder_state"]
    if not isinstance(source_state, Mapping):
        raise TypeError("encoder_state must be a state_dict mapping")

    spatial_reference = DetailV2Encoder()
    reference_state = spatial_reference.state_dict()
    source_keys = set(source_state)
    reference_keys = set(reference_state)
    if source_keys != reference_keys:
        missing = sorted(reference_keys - source_keys)
        unexpected = sorted(source_keys - reference_keys)
        raise RuntimeError(
            "Spatial V2 state_dict keys do not match exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key in sorted(reference_keys):
        source_tensor = source_state[key]
        reference_tensor = reference_state[key]
        if not isinstance(source_tensor, Tensor):
            raise RuntimeError(f"state_dict value for {key} is not a tensor")
        if source_tensor.shape != reference_tensor.shape:
            raise RuntimeError(
                f"shape mismatch for {key}: expected {tuple(reference_tensor.shape)}, "
                f"got {tuple(source_tensor.shape)}"
            )
        if source_tensor.dtype != reference_tensor.dtype:
            raise RuntimeError(
                f"dtype mismatch for {key}: expected {reference_tensor.dtype}, "
                f"got {source_tensor.dtype}"
            )
    spatial_reference.load_state_dict(source_state, strict=True)

    encoder = build_encoder("detail_v2_dynamic", **structure)
    if not isinstance(encoder, DynamicDetailV2Encoder):
        raise RuntimeError("dynamic encoder factory returned the wrong encoder class")
    destination_state = encoder.state_dict()
    destination_keys = set(destination_state)
    compatible_keys = source_keys & destination_keys
    new_keys = destination_keys - source_keys
    incompatible_source_keys = source_keys - destination_keys
    if compatible_keys != source_keys or incompatible_source_keys:
        raise RuntimeError(
            "Dynamic V2 does not consume every Spatial V2 key; "
            f"incompatible={sorted(incompatible_source_keys)}"
        )
    if new_keys != _DYNAMIC_NEW_STATE_KEYS:
        raise RuntimeError(
            "Dynamic V2 new state keys do not match the explicit contract; "
            f"missing={sorted(_DYNAMIC_NEW_STATE_KEYS - new_keys)}, "
            f"unexpected={sorted(new_keys - _DYNAMIC_NEW_STATE_KEYS)}"
        )
    for key in sorted(compatible_keys):
        source_tensor = source_state[key]
        destination_tensor = destination_state[key]
        if source_tensor.shape != destination_tensor.shape:
            raise RuntimeError(
                f"shape mismatch for compatible key {key}: "
                f"source={tuple(source_tensor.shape)}, "
                f"destination={tuple(destination_tensor.shape)}"
            )
        if source_tensor.dtype != destination_tensor.dtype:
            raise RuntimeError(
                f"dtype mismatch for compatible key {key}: "
                f"source={source_tensor.dtype}, destination={destination_tensor.dtype}"
            )

    merged_state = dict(destination_state)
    merged_state.update(source_state)
    encoder.load_state_dict(merged_state, strict=True)
    compatible_key_list = sorted(compatible_keys)
    new_key_list = sorted(new_keys)
    report = {
        "source_encoder_type": "detail_v2",
        "target_encoder_type": "detail_v2_dynamic",
        "compatible_keys": compatible_key_list,
        "new_keys": new_key_list,
        "compatible_key_count": len(compatible_key_list),
        "new_key_count": len(new_key_list),
    }
    return encoder, report


def export_encoder_checkpoint(
    encoder: OriginalEncoder | DetailV2Encoder | DynamicDetailV2Encoder,
    preprocess: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the strict encoder-only checkpoint payload."""
    if not isinstance(
        encoder,
        (OriginalEncoder, DetailV2Encoder, DynamicDetailV2Encoder),
    ):
        raise TypeError(
            "encoder must be OriginalEncoder, DetailV2Encoder, or "
            "DynamicDetailV2Encoder"
        )
    if not isinstance(preprocess, Mapping):
        raise TypeError("preprocess must be a mapping")
    return {
        "encoder_type": encoder.encoder_type,
        "structure": copy.deepcopy(encoder.structure),
        "preprocess": copy.deepcopy(dict(preprocess)),
        "encoder_state": copy.deepcopy(encoder.state_dict()),
    }


def save_encoder_checkpoint(
    path: str | os.PathLike[str],
    encoder: OriginalEncoder | DetailV2Encoder | DynamicDetailV2Encoder,
    preprocess: Mapping[str, Any],
) -> None:
    """Atomically save a strict encoder-only checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = export_encoder_checkpoint(encoder, preprocess)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _read_checkpoint(
    source: str | os.PathLike[str] | Mapping[str, Any],
    map_location: str | torch.device,
) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    return torch.load(source, map_location=map_location, weights_only=True)


def load_encoder_checkpoint(
    source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
    expected_encoder_type: str | None = None,
    expected_preprocess: Mapping[str, Any] | None = None,
) -> tuple[
    OriginalEncoder | DetailV2Encoder | DynamicDetailV2Encoder,
    dict[str, Any],
]:
    """Strictly validate and restore an encoder plus its public metadata."""
    checkpoint = _read_checkpoint(source, map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    actual_fields = set(checkpoint)
    if actual_fields != _CHECKPOINT_FIELDS:
        missing = sorted(_CHECKPOINT_FIELDS - actual_fields)
        unexpected = sorted(actual_fields - _CHECKPOINT_FIELDS)
        raise KeyError(
            "checkpoint fields do not match contract; "
            f"missing={missing}, unexpected={unexpected}"
        )

    encoder_type = checkpoint["encoder_type"]
    if expected_encoder_type is not None and encoder_type != expected_encoder_type:
        raise ValueError(
            f"encoder_type mismatch: expected {expected_encoder_type!r}, "
            f"got {encoder_type!r}"
        )
    preprocess = checkpoint["preprocess"]
    if not isinstance(preprocess, Mapping):
        raise TypeError("checkpoint preprocess must be a mapping")
    if expected_preprocess is not None and dict(preprocess) != dict(expected_preprocess):
        raise ValueError(
            f"preprocess mismatch: expected {dict(expected_preprocess)!r}, "
            f"got {dict(preprocess)!r}"
        )

    structure = checkpoint["structure"]
    if not isinstance(structure, Mapping):
        raise TypeError("checkpoint structure must be a mapping")
    if encoder_type == "detail_v2_dynamic":
        _validate_dynamic_structure(structure)
    elif encoder_type == "detail_v2_tra":
        from .tra_v2 import validate_tra_structure

        validate_tra_structure(structure)
    else:
        expected_structure = {
            "original": _ORIGINAL_STRUCTURE,
            "detail_v2": _DETAIL_V2_STRUCTURE,
        }.get(encoder_type)
        if expected_structure is None:
            raise ValueError(f"unsupported encoder_type: {encoder_type}")
        if dict(structure) != expected_structure:
            raise ValueError(
                f"unsupported {encoder_type} structure: expected "
                f"{expected_structure}, got {dict(structure)}"
            )

    encoder = build_encoder(encoder_type, **dict(structure))
    encoder_state = checkpoint["encoder_state"]
    if not isinstance(encoder_state, Mapping):
        raise TypeError("encoder_state must be a state_dict mapping")
    encoder.load_state_dict(encoder_state, strict=True)
    metadata = {
        "encoder_type": encoder_type,
        "structure": copy.deepcopy(dict(structure)),
        "preprocess": copy.deepcopy(dict(preprocess)),
    }
    return encoder, metadata


def warm_start_tra_encoder(
    source: str | os.PathLike[str] | Mapping[str, Any],
    **structure: Any,
) -> tuple[DetailV2Encoder, dict[str, Any]]:
    """Lazily dispatch the strict Spatial V2 to TRA warm-start."""
    from .tra_v2 import warm_start_tra_encoder as _warm_start_tra_encoder

    return _warm_start_tra_encoder(source, **structure)


__all__ = [
    "DetailV2Encoder",
    "DynamicDetailV2Encoder",
    "OriginalEncoder",
    "build_2d_sincos_position_embedding",
    "build_encoder",
    "export_encoder_checkpoint",
    "load_encoder_checkpoint",
    "save_encoder_checkpoint",
    "warm_start_dynamic_encoder",
    "warm_start_tra_encoder",
]

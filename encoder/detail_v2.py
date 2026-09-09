"""Canonical single-frame tactile encoders and their checkpoint contract.

This module intentionally contains encoders only. Pretraining decoders belong to
the training pipeline and must never be serialized as part of an encoder
checkpoint.
"""

from __future__ import annotations

import copy
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


def build_encoder(
    encoder_type: str,
    *,
    backbone: str = "resnet18",
    input_channels: int = 3,
    latent_dims: int = 512,
    token_dim: int = 256,
    num_heads: int = 4,
    attention_dropout: float = 0.0,
    weights: Any = None,
    trunk_state: Mapping[str, Tensor] | None = None,
) -> OriginalEncoder | DetailV2Encoder:
    """Build a canonical B0 or Details V2 encoder."""
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
    raise ValueError(f"unsupported encoder_type: {encoder_type}")


def export_encoder_checkpoint(
    encoder: OriginalEncoder | DetailV2Encoder,
    preprocess: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the strict encoder-only checkpoint payload."""
    if not isinstance(encoder, (OriginalEncoder, DetailV2Encoder)):
        raise TypeError("encoder must be OriginalEncoder or DetailV2Encoder")
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
    encoder: OriginalEncoder | DetailV2Encoder,
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
) -> tuple[OriginalEncoder | DetailV2Encoder, dict[str, Any]]:
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


__all__ = [
    "DetailV2Encoder",
    "OriginalEncoder",
    "build_2d_sincos_position_embedding",
    "build_encoder",
    "export_encoder_checkpoint",
    "load_encoder_checkpoint",
    "save_encoder_checkpoint",
]

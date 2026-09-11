"""Spatial-first Temporal Residual Aggregator for Details V2."""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .detail_v2 import (
    DetailV2Encoder,
    _CHECKPOINT_FIELDS,
    _DETAIL_V2_STRUCTURE,
    _read_checkpoint,
    build_2d_sincos_position_embedding,
)


TRA_DETAIL_V2_STRUCTURE = {
    **_DETAIL_V2_STRUCTURE,
    "sequence_length": 4,
    "temporal_stride": 1,
    "initial_temporal_gate": 0.1,
}
TRA_NEW_STATE_KEYS = {
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
}


def validate_tra_structure(structure: Mapping[str, Any]) -> None:
    """Require the fixed first-version TRA architecture."""
    if dict(structure) != TRA_DETAIL_V2_STRUCTURE:
        raise ValueError(
            "unsupported detail_v2_tra structure; expected "
            f"{TRA_DETAIL_V2_STRUCTURE}, got {dict(structure)}"
        )


class TRADetailV2Encoder(DetailV2Encoder):
    """Apply a small GRU to four Spatial V2 detail vectors."""

    encoder_type = "detail_v2_tra"

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
        initial_temporal_gate: float = 0.1,
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
            "initial_temporal_gate": float(initial_temporal_gate),
        }
        validate_tra_structure(structure)
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
        self.gru = nn.GRU(
            input_size=token_dim,
            hidden_size=token_dim,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
        )
        self.temporal_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.alpha = nn.Parameter(
            torch.tensor(
                math.log(initial_temporal_gate / (1.0 - initial_temporal_gate))
            )
        )
        self.fusion_norm = nn.LayerNorm(token_dim)

    def _validate_sequence(self, inputs: Tensor) -> None:
        if inputs.ndim != 5:
            raise ValueError("TRA tactile input must be five-dimensional [B,T,3,H,W]")
        if inputs.shape[1] != self.structure["sequence_length"]:
            raise ValueError("TRA tactile input must contain exactly 4 frames")
        if inputs.shape[2] != self.structure["input_channels"]:
            raise ValueError("TRA tactile input frames must contain exactly 3 channels")

    def forward_spatial_details(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return current semantic context and four current-query detail vectors."""
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
        current_query = self.global_query(global_feature).unsqueeze(1)

        local_tokens = layer2_sequence.flatten(3).permute(0, 1, 3, 2)
        local_tokens = self.local_projection(local_tokens)
        position = build_2d_sincos_position_embedding(
            height=grid_height,
            width=grid_width,
            embed_dim=self.structure["token_dim"],
            device=local_tokens.device,
            dtype=local_tokens.dtype,
        )
        local_tokens = local_tokens + position.unsqueeze(1)

        spatial_details = []
        for frame_index in range(sequence_length):
            frame_tokens = local_tokens[:, frame_index]
            detail, _ = self.cross_attention(
                current_query,
                frame_tokens,
                frame_tokens,
                need_weights=False,
            )
            spatial_details.append(self.detail_projection(detail.squeeze(1)))
        spatial_sequence = torch.stack(spatial_details, dim=1)
        return semantic, spatial_sequence

    def forward_with_details(self, inputs: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the ACT embedding and tensors used for TRA sanity checks."""
        semantic, spatial_sequence = self.forward_spatial_details(inputs)

        _, last_hidden = self.gru(spatial_sequence)
        temporal_hidden = last_hidden[-1]
        temporal_residual = self.temporal_mlp(temporal_hidden)
        current_detail = spatial_sequence[:, -1]
        temporal_gate = torch.sigmoid(self.alpha)
        fused_detail = self.fusion_norm(
            current_detail + temporal_gate * temporal_residual
        )
        embedding = torch.cat((semantic, fused_detail), dim=1)
        details = {
            "semantic": semantic,
            "spatial_sequence": spatial_sequence,
            "current_detail": current_detail,
            "temporal_hidden": temporal_hidden,
            "temporal_residual": temporal_residual,
            "temporal_gate": temporal_gate,
            "fused_detail": fused_detail,
        }
        return embedding, details

    def forward(self, inputs: Tensor) -> Tensor:
        embedding, _ = self.forward_with_details(inputs)
        return embedding


def _validate_spatial_checkpoint(
    source: str | os.PathLike[str] | Mapping[str, Any],
) -> tuple[Mapping[str, Tensor], Mapping[str, Any]]:
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
        raise ValueError("TRA warm-start source must be a detail_v2 checkpoint")
    if (
        not isinstance(checkpoint["structure"], Mapping)
        or dict(checkpoint["structure"]) != _DETAIL_V2_STRUCTURE
    ):
        raise ValueError(
            "TRA warm-start source must use the canonical detail_v2 structure"
        )
    if not isinstance(checkpoint["preprocess"], Mapping):
        raise TypeError("checkpoint preprocess must be a mapping")
    source_state = checkpoint["encoder_state"]
    if not isinstance(source_state, Mapping):
        raise TypeError("encoder_state must be a state_dict mapping")

    reference_state = DetailV2Encoder().state_dict()
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
    return source_state, checkpoint["preprocess"]


def warm_start_tra_encoder(
    source: str | os.PathLike[str] | Mapping[str, Any],
    **structure: Any,
) -> tuple[TRADetailV2Encoder, dict[str, Any]]:
    """Warm-start TRA from every tensor in a strict Spatial V2 checkpoint."""
    source_state, _ = _validate_spatial_checkpoint(source)
    encoder = TRADetailV2Encoder(**structure)
    destination_state = encoder.state_dict()
    source_keys = set(source_state)
    destination_keys = set(destination_state)
    compatible_keys = source_keys & destination_keys
    new_keys = destination_keys - source_keys
    incompatible_source_keys = source_keys - destination_keys
    if compatible_keys != source_keys or incompatible_source_keys:
        raise RuntimeError(
            "TRA does not consume every Spatial V2 key; "
            f"incompatible={sorted(incompatible_source_keys)}"
        )
    if new_keys != TRA_NEW_STATE_KEYS:
        raise RuntimeError(
            "TRA new state keys do not match the explicit contract; "
            f"missing={sorted(TRA_NEW_STATE_KEYS - new_keys)}, "
            f"unexpected={sorted(new_keys - TRA_NEW_STATE_KEYS)}"
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
        "target_encoder_type": "detail_v2_tra",
        "compatible_keys": compatible_key_list,
        "new_keys": new_key_list,
        "compatible_key_count": len(compatible_key_list),
        "new_key_count": len(new_key_list),
    }
    return encoder, report


def measure_spatial_warm_start_sanity(
    spatial: DetailV2Encoder,
    tra: TRADetailV2Encoder,
    inputs: Tensor,
) -> dict[str, float]:
    """Measure raw warm-start equality and the mandated final LayerNorm delta."""
    spatial_was_training = spatial.training
    tra_was_training = tra.training
    alpha_before = tra.alpha.detach().clone()
    spatial.eval()
    tra.eval()
    try:
        with torch.no_grad():
            tra.alpha.fill_(-50.0)
            spatial_embedding = spatial(inputs[:, -1])
            tra_embedding, details = tra.forward_with_details(inputs)
            spatial_semantic = spatial_embedding[:, :256]
            spatial_detail = spatial_embedding[:, 256:]
            final_detail = tra_embedding[:, 256:]
            relative_l2 = torch.linalg.vector_norm(
                final_detail - spatial_detail
            ) / torch.linalg.vector_norm(spatial_detail).clamp_min(
                torch.finfo(spatial_detail.dtype).eps
            )
            cosine = F.cosine_similarity(final_detail, spatial_detail, dim=1).mean()
            return {
                "semantic_max_abs_error": float(
                    (details["semantic"] - spatial_semantic).abs().max().item()
                ),
                "current_detail_max_abs_error": float(
                    (details["current_detail"] - spatial_detail).abs().max().item()
                ),
                "final_detail_cosine_similarity": float(cosine.item()),
                "final_detail_relative_l2_error": float(relative_l2.item()),
                "gate_value": float(details["temporal_gate"].item()),
            }
    finally:
        with torch.no_grad():
            tra.alpha.copy_(alpha_before)
        spatial.train(spatial_was_training)
        tra.train(tra_was_training)


__all__ = [
    "TRA_DETAIL_V2_STRUCTURE",
    "TRADetailV2Encoder",
    "measure_spatial_warm_start_sanity",
    "validate_tra_structure",
    "warm_start_tra_encoder",
]

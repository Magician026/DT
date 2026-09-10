# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List, Literal
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

# Use the exact encoder class and preprocessing contract shared by pretraining.
# The repository root is not necessarily on sys.path when ACT is launched from
# policy/ACT, so add that fixed source root before importing the canonical code.
repository_root = Path(__file__).resolve().parents[4]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from encoder.clean_v2 import PREPROCESS
from encoder.detail_v2 import (
    DetailV2Encoder,
    DynamicDetailV2Encoder,
    OriginalEncoder,
    load_encoder_checkpoint,
)

import IPython

e = IPython.embed


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                             unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models,
                           name)(replace_stride_with_dilation=[False, False, dilation],
                                 pretrained=is_main_process(),
                                 norm_layer=FrozenBatchNorm2d)  # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_vision_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

class TactileBackbone(nn.Module):
    """ACT adapter for legacy tactile encoders and canonical Details V2."""

    def __init__(
        self,
        name: str,
        ckpt: str,
        tac_names: list[str],
        train_backbone: bool,
        return_interm_layers: bool,
        position_embedding,
        tactile_type: Literal['feat', 'full'] = 'feat',
        tactile_encoder_type: str = 'original',
    ):
        super().__init__()

        checkpoint_path = Path(ckpt).expanduser() if ckpt else None
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError(f"tactile checkpoint does not exist: {ckpt!r}")

        self.train_backbone = train_backbone
        self.tactile_type = tactile_type
        self.tactile_encoder_type = tactile_encoder_type
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        self.checkpoint_load_coverage = 1.0
        self.checkpoint_metadata = None
        self._uses_canonical_encoder = False
        self.tac_names = tac_names
        self.num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

        if tactile_encoder_type in ('detail_v2', 'detail_v2_dynamic'):
            if name != 'resnet18':
                raise ValueError(
                    f"{tactile_encoder_type} requires tactile_backbone='resnet18'"
                )
            if tactile_type != 'feat':
                raise ValueError(
                    f"{tactile_encoder_type} only supports tactile_type='feat'"
                )
            encoder, metadata = load_encoder_checkpoint(
                checkpoint_path,
                expected_encoder_type=tactile_encoder_type,
                expected_preprocess=PREPROCESS,
            )
            expected_class = (
                DetailV2Encoder
                if tactile_encoder_type == 'detail_v2'
                else DynamicDetailV2Encoder
            )
            if type(encoder) is not expected_class:
                raise TypeError(
                    f"{tactile_encoder_type} checkpoint loader returned a "
                    "non-canonical encoder"
                )
            self.backbone = encoder
            self.position_embedding = nn.Embedding(1, self.num_channels)
            self.checkpoint_metadata = metadata
            self._uses_canonical_encoder = True
            self._freeze_batch_norm_affine_and_stats()
        elif tactile_encoder_type == 'original':
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=True
            )
            if isinstance(checkpoint, Mapping) and 'encoder_state' in checkpoint:
                if name != 'resnet18':
                    raise ValueError("canonical original encoder requires resnet18")
                if tactile_type != 'feat':
                    raise ValueError(
                        "canonical original encoder only supports tactile_type='feat'"
                    )
                encoder, metadata = load_encoder_checkpoint(
                    checkpoint,
                    expected_encoder_type='original',
                    expected_preprocess=PREPROCESS,
                )
                if not isinstance(encoder, OriginalEncoder):
                    raise TypeError(
                        "original checkpoint loader returned a non-canonical encoder"
                    )
                self.backbone = encoder
                self.position_embedding = nn.Embedding(1, self.num_channels)
                self.checkpoint_metadata = metadata
                self._uses_canonical_encoder = True
                self._freeze_batch_norm_affine_and_stats()
            else:
                self._build_original_encoder(
                    name,
                    checkpoint,
                    return_interm_layers,
                    position_embedding,
                )
        else:
            raise ValueError(
                f"unsupported tactile_encoder_type: {tactile_encoder_type!r}"
            )

        if not train_backbone:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
        self.train(self.training)

    def _build_original_encoder(
        self,
        name,
        checkpoint,
        return_interm_layers,
        position_embedding,
    ):
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}

        from .network import Tactile

        backbone = Tactile(backbone='resnet18', supervise=[
            'marker', 'rgb', 'marked_rgb', 'pose', 'depth'
        ], latent_dims=self.num_channels, norm_layer=FrozenBatchNorm2d)
        backbone.load_state_dict(checkpoint, strict=True)
        self.checkpoint_metadata = {
            'encoder_type': 'original',
            'preprocess': None,
            'format': 'legacy_tactile_state_dict',
        }

        if self.tactile_type == 'feat':
            self.backbone = backbone.backbone
            self.position_embedding = nn.Embedding(1, 512)
        else:
            self.backbone = IntermediateLayerGetter(backbone.backbone, return_layers=return_layers)
            self.position_embedding = position_embedding

    def _freeze_batch_norm_affine_and_stats(self):
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                if module.affine:
                    module.weight.requires_grad_(False)
                    module.bias.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if not self.train_backbone:
            # Frozen encoders must also keep stochastic layers and BatchNorm state
            # fixed when the surrounding ACT policy switches to train mode.
            self.backbone.eval()
        elif self._uses_canonical_encoder:
            # Match the reference FrozenBatchNorm policy: ordinary BN parameters
            # retain checkpoint values and never update affine or running state.
            self._freeze_batch_norm_affine_and_stats()
        return self

    def forward(self, x):
        feat, pos = [], []
        if self.tactile_type == 'full':
            xs = self.backbone(x) # dict of feature maps [N, 512, 8, 8]
            for name, x in xs.items():
                feat.append(x)
                pos.append(self.position_embedding(x).to(x.dtype))
        else:
            xs = self.backbone(x) # [N, 512]
            feat.append(xs.unsqueeze(2).unsqueeze(3)) # [N, 512, 1, 1]
            pos.append(self.position_embedding.weight.unsqueeze(-1).unsqueeze(-1)) # [1, 512, 1, 1]
        return feat, pos

def build_tactile_backbone(args):
    train_backbone = args.lr_tactile_backbone > 0
    return_interm_layers = args.tactile_masks
    tactile_type = args.tactile_type if hasattr(args, 'tactile_type') else 'feat'
    tactile_encoder_type = (
        args.tactile_encoder_type
        if hasattr(args, 'tactile_encoder_type')
        else 'original'
    )
    position_embedding = build_position_encoding(args)
    backbone = TactileBackbone(
        args.tactile_backbone,
        args.tactile_ckpt,
        args.tactile_names,
        train_backbone,
        return_interm_layers,
        position_embedding,
        tactile_type,
        tactile_encoder_type,
    )
    return backbone

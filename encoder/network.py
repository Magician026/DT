import torch
import torch.nn as nn
from torchvision import models

from typing import Literal

class RGBDecoder(nn.Module):
    def __init__(self, latent_dims=512, output_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dims, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, stride=2, padding=1), # 32x32 -> 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, stride=2, padding=1), # 64x64 -> 128x128
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'), # 128x128 -> 256x256
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        x = self.deconv(x)
        return x

class MarkerDecoder(nn.Module):
    def __init__(self, latent_dims=512, marker_nums=63):
        super().__init__()
        self.marker_nums = marker_nums
        self.ffn = nn.Sequential(
            # nn.Linear(latent_dims, latent_dims * 2),
            # nn.GELU(),
            # nn.Linear(latent_dims * 2, marker_nums * 2)
            nn.Linear(latent_dims, 256),
            nn.GELU(),
            nn.Linear(256, marker_nums * 2)
        )

    def forward(self, x):
        x = self.ffn(x)
        x = x.view(-1, self.marker_nums, 2)
        return x 

class PoseDecoder(nn.Module):
    def __init__(self, latent_dims=512, pose_dims=7):
        super().__init__()
        self.ffn = nn.Sequential(
            # nn.Linear(latent_dims, latent_dims * 2),
            # nn.GELU(),
            # nn.Linear(latent_dims * 2, pose_dims)
            nn.Linear(latent_dims, 256),
            nn.GELU(),
            nn.Linear(256, pose_dims)
        )

    def forward(self, x):
        x = self.ffn(x)
        return x


class DetailPreservingEncoderV1(nn.Module):
    """A minimal multi-scale ResNet encoder with an explicit detail branch.

    The trunk keeps the original ResNet-18 downsampling schedule.  The detail
    branch reads layer2 and preserves a coarse 4x4 spatial layout, while the
    semantic branch reads layer4 with global average pooling.  Their projected
    256-D representations are concatenated into the original 512-D latent.
    """

    def __init__(
        self,
        latent_dims=512,
        detail_grid_size=4,
        detail_dim=256,
        semantic_dim=256,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        if latent_dims != detail_dim + semantic_dim:
            raise ValueError(
                "DetailPreservingEncoderV1 requires latent_dims == "
                "detail_dim + semantic_dim"
            )

        self.detail_grid_size = detail_grid_size
        self.trunk = models.resnet18(
            weights=None,
            num_classes=latent_dims,
            norm_layer=norm_layer,
        )
        # The original ResNet fc is not part of V1.  Keeping the convolutional
        # trunk explicit prevents the final 1x1 representation from replacing
        # the detail pathway.
        self.trunk.fc = nn.Identity()

        self.detail_projection = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            norm_layer(128),
            nn.GELU(),
        )
        self.detail_pool = nn.AdaptiveAvgPool2d(
            (detail_grid_size, detail_grid_size)
        )
        self.detail_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * detail_grid_size * detail_grid_size, detail_dim),
            nn.LayerNorm(detail_dim),
            nn.GELU(),
        )

        self.semantic_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.semantic_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
        )

    def forward_features(self, x):
        """Return intermediate feature maps for policy-side adapters."""
        x = self.trunk.conv1(x)
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x)
        layer1 = self.trunk.layer1(x)
        layer2 = self.trunk.layer2(layer1)
        layer3 = self.trunk.layer3(layer2)
        layer4 = self.trunk.layer4(layer3)
        return {
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
        }

    def forward(self, x):
        features = self.forward_features(x)
        detail = self.detail_head(
            self.detail_pool(self.detail_projection(features["layer2"]))
        )
        semantic = self.semantic_head(self.semantic_pool(features["layer4"]))
        return torch.cat([semantic, detail], dim=1)


def build_encoder(
    backbone="resnet18",
    latent_dims=512,
    encoder_type="original",
    norm_layer=nn.BatchNorm2d,
):
    """Build either the untouched baseline or the experimental V1 encoder."""
    if encoder_type == "original":
        if backbone == "resnet18":
            return models.resnet18(
                weights=None, num_classes=latent_dims, norm_layer=norm_layer
            )
        if backbone == "resnet34":
            return models.resnet34(
                weights=None, num_classes=latent_dims, norm_layer=norm_layer
            )
        if backbone == "resnet50":
            return models.resnet50(
                weights=None, num_classes=latent_dims, norm_layer=norm_layer
            )
        raise ValueError(f"Unsupported backbone: {backbone}")

    if encoder_type == "detail_v1":
        if backbone != "resnet18":
            raise ValueError("DetailPreservingEncoderV1 currently supports resnet18 only")
        return DetailPreservingEncoderV1(
            latent_dims=latent_dims,
            norm_layer=norm_layer,
        )

    raise ValueError(f"Unsupported encoder_type: {encoder_type}")

class Tactile(nn.Module):
    def __init__(
        self, backbone='resnet18', latent_dims=512,
        supervise:list[Literal['marker', 'rgb', 'marked_rgb', 'pose', 'depth']]=['marked_rgb'],
        encoder_type='original',
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.backbone = build_encoder(
            backbone=backbone,
            latent_dims=latent_dims,
            encoder_type=encoder_type,
            norm_layer=norm_layer,
        )
        
        self.supervise = supervise
        self.decoders = nn.ModuleDict()
        if 'rgb' in self.supervise:
            self.decoders['rgb'] = RGBDecoder(latent_dims=latent_dims, output_channels=3)
        if 'marked_rgb' in self.supervise:
            self.decoders['marked_rgb'] = RGBDecoder(latent_dims=latent_dims, output_channels=3)
        if 'depth' in self.supervise:
            self.decoders['depth'] = RGBDecoder(latent_dims=latent_dims, output_channels=1)
        if 'marker' in self.supervise:
            self.decoders['marker'] = MarkerDecoder(latent_dims=latent_dims, marker_nums=63)
        if 'pose' in self.supervise:
            self.decoders['pose'] = PoseDecoder(latent_dims=latent_dims, pose_dims=7)
    
    def reconstruct(self, img):
        latent = self.forward(img)
        outputs = {}
        for key in self.supervise:
            outputs[key] = self.decoders[key](latent)
        return outputs

    def forward(self, x):
        return self.backbone(x)
    
    def loss(self, outputs:dict, targets:dict, weights:dict=None):
        loss = 0
        loss_dict = {}
        criterion = nn.MSELoss()
        for key in self.supervise:
            if key in ['rgb', 'marked_rgb', 'depth'] and outputs[key].shape != targets[key].shape:
                resized = nn.functional.interpolate(
                    outputs[key], size=targets[key].shape[2:], mode='bilinear', align_corners=False)
                part_loss = criterion(resized, targets[key])
            else:
                part_loss = criterion(outputs[key], targets[key])
            loss += weights.get(key, 1.0) * part_loss
            loss_dict[key] = part_loss.item()
        loss_dict['total'] = loss.item()
        return loss, loss_dict

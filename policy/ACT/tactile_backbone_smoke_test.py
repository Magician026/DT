import io
import sys
from argparse import Namespace
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from policy.ACT.detr.models.backbone import TactileBackbone  # noqa: E402
from policy.ACT.detr.models.position_encoding import (  # noqa: E402
    build_position_encoding,
)


def make_args(tactile_type):
    return Namespace(
        hidden_dim=512,
        position_embedding="sine",
        tactile_type=tactile_type,
        tactile_masks=False,
    )


def build_tactile_backbone(encoder_type, tactile_type):
    args = make_args(tactile_type)
    return TactileBackbone(
        name="resnet18",
        ckpt=None,
        tac_names=["tac_left", "tac_right"],
        train_backbone=True,
        return_interm_layers=args.tactile_masks,
        position_embedding=build_position_encoding(args),
        tactile_type=tactile_type,
        tactile_encoder_type=encoder_type,
    )


def check_output(encoder_type, tactile_type, expected_feature_shape, expected_position_shape):
    model = build_tactile_backbone(encoder_type, tactile_type)
    model.eval()
    with torch.no_grad():
        features, positions = model(torch.randn(2, 3, 256, 256))
    assert len(features) == 1
    assert tuple(features[0].shape) == expected_feature_shape
    assert tuple(positions[0].shape) == expected_position_shape
    print(
        f"{encoder_type}/{tactile_type}: features={tuple(features[0].shape)}, "
        f"positions={tuple(positions[0].shape)}"
    )
    return model


def check_original_policy_checkpoint(model):
    checkpoint_path = (
        PROJECT_ROOT.parent
        / "UniVTAC"
        / "policy/ACT/act_ckpt/act-lift_bottle/demo-50/"
        "train_config_official_univtac/policy_last.ckpt"
    )
    if not checkpoint_path.is_file():
        print("original_policy_checkpoint: skipped; checkpoint not present")
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    prefix = "model.backbones.1."
    tactile_state = {
        key[len(prefix):]: value
        for key, value in checkpoint.items()
        if key.startswith(prefix)
    }
    model_state = model.state_dict()
    assert set(tactile_state) == set(model_state)
    assert all(tactile_state[key].shape == model_state[key].shape for key in model_state)
    print(
        "original_policy_checkpoint: tactile key/shape compatibility passed "
        f"({len(model_state)} keys)"
    )


def main():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    original_feat = check_output(
        "original", "feat", (2, 512, 1, 1), (1, 512, 1, 1)
    )
    check_output("detail_v1", "feat", (2, 512, 1, 1), (1, 512, 1, 1))
    check_output("original", "full", (2, 512, 8, 8), (1, 512, 8, 8))
    check_output("detail_v1", "full", (2, 512, 8, 8), (1, 512, 8, 8))
    check_original_policy_checkpoint(original_feat)

    detail_feat = build_tactile_backbone("detail_v1", "feat")
    buffer = io.BytesIO()
    torch.save(detail_feat.state_dict(), buffer)
    buffer.seek(0)
    reloaded = build_tactile_backbone("detail_v1", "feat")
    status = reloaded.load_state_dict(
        torch.load(buffer, map_location="cpu", weights_only=True), strict=True
    )
    print("detail_v1_policy_checkpoint_reload:", status)
    print("POLICY_SMOKE_TEST_PASSED")


if __name__ == "__main__":
    main()

import io
import sys
from pathlib import Path

import torch


ENCODER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENCODER_DIR.parent
sys.path.insert(0, str(ENCODER_DIR))

from network import Tactile  # noqa: E402


ALL_SUPERVISION = ["marker", "rgb", "marked_rgb", "pose", "depth"]


def assert_finite_gradients(model):
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients, "model has no trainable parameters"
    assert all(g is not None for g in gradients), "missing gradient"
    assert all(torch.isfinite(g).all() for g in gradients), "non-finite gradient"


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)

    original_full = Tactile(
        backbone="resnet18",
        latent_dims=512,
        supervise=ALL_SUPERVISION,
        encoder_type="original",
    )
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "encoder.pth"
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        status = original_full.load_state_dict(checkpoint, strict=True)
        print("original_checkpoint_load:", status)
    else:
        print("original_checkpoint_load: skipped; checkpoint not present")
    del original_full

    original = Tactile(
        backbone="resnet18", latent_dims=512, supervise=[], encoder_type="original"
    )
    detail = Tactile(
        backbone="resnet18", latent_dims=512, supervise=[], encoder_type="detail_v1"
    )
    x = torch.randn(2, 3, 256, 256)

    original.eval()
    detail.eval()
    with torch.no_grad():
        original_latent = original(x)
        detail_latent = detail(x)
        feature_maps = detail.backbone.forward_features(x)

    assert original_latent.shape == (2, 512)
    assert detail_latent.shape == (2, 512)
    assert feature_maps["layer2"].shape == (2, 128, 32, 32)
    assert feature_maps["layer4"].shape == (2, 512, 8, 8)
    print("original_forward:", tuple(original_latent.shape))
    print("detail_v1_forward:", tuple(detail_latent.shape))
    print("detail_v1_layer2:", tuple(feature_maps["layer2"].shape))
    print("detail_v1_layer4:", tuple(feature_maps["layer4"].shape))

    for name, model in (("original", original), ("detail_v1", detail)):
        model.train()
        model.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        assert torch.isfinite(loss), f"{name} loss is non-finite"
        assert_finite_gradients(model)
        print(f"{name}_backward: loss={loss.item():.6f}, gradients=finite")

    detail_with_decoders = Tactile(
        backbone="resnet18",
        latent_dims=512,
        supervise=ALL_SUPERVISION,
        encoder_type="detail_v1",
    )
    detail_with_decoders.eval()
    with torch.no_grad():
        decoded = detail_with_decoders.reconstruct(x[:1])
    expected_shapes = {
        "rgb": (1, 3, 256, 256),
        "marked_rgb": (1, 3, 256, 256),
        "depth": (1, 1, 256, 256),
        "marker": (1, 63, 2),
        "pose": (1, 7),
    }
    assert {key: tuple(value.shape) for key, value in decoded.items()} == expected_shapes
    print("detail_v1_decoder_outputs:", expected_shapes)
    del detail_with_decoders

    buffer = io.BytesIO()
    torch.save(detail.state_dict(), buffer)
    buffer.seek(0)
    reloaded = Tactile(
        backbone="resnet18", latent_dims=512, supervise=[], encoder_type="detail_v1"
    )
    reload_status = reloaded.load_state_dict(
        torch.load(buffer, map_location="cpu", weights_only=True), strict=True
    )
    print("detail_v1_checkpoint_reload:", reload_status)
    print(
        "parameter_counts:",
        {
            "original_backbone": sum(p.numel() for p in original.parameters()),
            "detail_v1_backbone": sum(p.numel() for p in detail.parameters()),
        },
    )
    print("SMOKE_TEST_PASSED")


if __name__ == "__main__":
    main()

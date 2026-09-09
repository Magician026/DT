# UniVTAC Details V2

Details V2 is an experimental single-frame tactile encoder, not a renamed V1.
The first run targets Insert HDMI: clean encoder pretraining → ACT demonstration
training → simulation success rate. No improvement claim is available yet.

## Implementation

`encoder/detail_v2.py` is the single encoder implementation used by pretraining
and ACT. ResNet18 layer2 retains its full spatial grid (32×32 at input 256×256).
128→256 local projection + LayerNorm + fixed 2-D sinusoidal coordinates form K/V.
Layer4 global average features produce a semantic 256-vector and one 256-query;
4-head attention (dropout zero) produces the detail 256-vector. Concatenation
returns `[B,512]`. The original ResNet18 encoder is retained for a same-data B0.

P0 enables only dense depth and all 1200 marker displacements. Sensor depth is
camera axial depth in mm, not a claim of calibrated indentation. Marker target is
current minus initial image-coordinate pixels (`marker[1] - marker[0]`). Both
use fixed train-only mean/std and mean squared error with weight 1. The decoders
read only the final embedding and have linear, unrestricted output ranges.
Pose, RGB reconstruction, augmentation and frame truncation are disabled.

100 clean trajectories split 90/10 with seed 42; both sensors stay in their
trajectory. Manifest is in `configs/encoder_split_seed42.json`. Training uses all
21,240 sensor frames; validation 2,360. Statistics use every tenth training frame
only, all pixels/markers. Depth mean/std: 32.5730768271 / 2.5551672946 mm;
marker displacement mean/std: 3.2583819008 / 1.6248295705 pixels.

Input: GSmini `rgb_marker`, native OpenCV decoded channel order without swapping,
uint8→CHW float /255, torchvision bilinear antialiased resize to 256×256. A real
clean frame and ACT episode tactile frame match exactly (max absolute error 0).
ACT has a separate 50-episode dataset and does not use the clean encoder loader.

## Runtime and dependencies

Encoder: Python 3.10, PyTorch 2.5.1, torchvision 0.20.1, h5py, numpy,
opencv-python-headless 4.10.0.84, pytest. Policy also needs einops, IPython, yaml.
The ACT code derives from UniVTAC `05bcd3e` and retains its license and upstream
Facebook notices. Existing repository history retains prior V1 implementations.

This repository does **not** bundle the simulator, assets, datasets or weights.
Simulation requires external [UniVTAC](https://github.com/univtac/UniVTAC)
`05bcd3e`, Isaac Sim 4.5.0, its compatible IsaacLab/TacEx/curobo installation and
assets, installed following upstream documentation. Machine paths are supplied
via CLI arguments/environment, never committed connection settings.

## Encoder usage

Run from the repository root with `PYTHONPATH=.`:

```bash
python -m pytest tests/test_detail_v2.py tests/test_clean_v2.py tests/test_p0.py
python scripts/audit_clean_v2.py --clean "$CLEAN_DATA" --act "$ACT_DATA" --out "$RUN/data"
python scripts/train_encoder_v2.py --smoke --config configs/encoder_p0.json \
  --clean "$CLEAN_DATA" --data-audit "$RUN/data/data_audit.json" \
  --split "$RUN/data/split.json" --out "$RUN/smoke"
python scripts/train_encoder_v2.py --config configs/encoder_p0.json \
  --source-commit "$(git rev-parse HEAD)" --clean "$CLEAN_DATA" \
  --data-audit "$RUN/data/data_audit.json" --split "$RUN/data/split.json" \
  --out "$RUN/encoder"
```

Use a fresh run directory; resume interrupted training with `--resume`.
`best.pt` and `last.pt` are atomic, encoder-only strict-format checkpoints;
`last_state.pt` retains optimizer/RNG/decoder state. Completion is published only
after strict reload and embedding equality. Downstream must verify completion,
run identity and SHA256; existence of a weight file alone is insufficient.

## Evidence and scope

M1: 16 encoder contract tests pass, active-target-only loader and P0 backward tests
pass, batch 1/4 full-size embeddings are finite. A real fixed batch overfits in
50 steps (first/last five loss mean 1.8490→0.9604). Three distinct real batch-32
steps pass backward/update and reload. These are engineering checks, not success
rate or generalization evidence. See `docs/details_v2_status.md` for stage status.

Historical HDMI 18/100 and released 15/100 use different recipes and are reference
only. A same-data original B0 is required before a structural improvement claim.
The initial budget is one encoder seed 42, five full epochs, one policy seed 42,
then fixed eval seeds 0–19 followed by 0–99 with no duplicate counting.

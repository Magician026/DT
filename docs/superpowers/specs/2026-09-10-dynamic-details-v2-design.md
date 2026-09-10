# Dynamic Details V2 Design

## Goal

Extend the working single-frame Details V2 tactile encoder with one lightweight causal local-dynamics branch, then retrain the encoder and the unchanged ACT policy for Insert HDMI. The downstream tactile interface remains `[B, 512]`. Formal evaluation uses exactly seeds `1000000..1000099`, with a first-20 quick triage before the full run.

## Locked experiment

- Keep ResNet18, the existing layer2 static cross-attention branch, the 256-D global context, ACT Transformer, action head, camera branch, qpos/action normalization, chunk definition, loss, demo split, and checkpoint selection rule.
- Use `sequence_length=4`, `temporal_stride=1`, `token_dim=256`, four heads, attention dropout zero, dynamic loss weight `0.5`, and initial sigmoid gate `0.15`.
- For anchor `t`, use history indices `max(0, t-3), max(0, t-2), max(0, t-1), t`; a non-unit stride multiplies the offsets. No index exceeds `t`, and history never crosses the current HDF5 trajectory or sensor side.
- Preserve all Spatial V2 artifacts. Dynamic runs live under a new `dynamic_v2_20260910` root.

## Encoder

`DynamicDetailV2Encoder` accepts `[B,T,3,H,W]`. A shared ResNet18 computes stem through layer2 for every frame; only the current frame continues through layer3/layer4. The current layer2 tokens use the existing `local_projection`, fixed 2-D positional embedding, `global_query`, `cross_attention`, and `detail_projection` parameter names so every compatible Spatial V2 parameter can be warm-started and explicitly verified.

Projected temporal residuals are computed at every layer2 grid location, normalized, augmented with the same fixed 2-D positional encoding and a learnable three-position temporal encoding, flattened to `[B,3072,256]`, and read by a separate four-head `dynamic_attention` using the current global query. The final detail is `LayerNorm(detail_static + sigmoid(gate_logit) * detail_dynamic)`. The encoder returns `concat(global_context, detail)` as `[B,512]`; an auxiliary method also exposes the pre-gate 256-D dynamic detail for pretraining.

Warm-start loads the canonical HDMI Spatial V2 encoder checkpoint, asserts the source checkpoint contract, source/destination key set, shape and dtype compatibility, consumes every Spatial V2 encoder key, and allows only an explicit set of new dynamic keys. The source path, SHA256, source commit and load report are recorded in the resolved config and completion record.

## Encoder pretraining data and loss

For each clean trajectory/sensor sample, the loader returns the causal image window and the current depth and marker-displacement targets. Marker displacement is `marker[:,1]-marker[:,0]`. The dynamic target is the change between the last two sampled history positions:

```text
delta_marker_t = displacement(t) - displacement(max(0, t-temporal_stride))
```

The audit computes `delta_marker` mean/std only from train-manifest trajectories and records the history parameters. `P0Model` keeps the existing normalized depth and marker MSEs and adds a 256-D dynamic head predicting normalized `[1200,2]` delta marker with weight `0.5`.

## ACT and rollout data flow

Training keeps camera, qpos and action supervision at anchor `t`; only tactile changes from `[sensors,3,H,W]` to `[sensors,T,3,H,W]`. The DETR call and encoder output remain unchanged after the tactile backbone.

Deployment stores transformed current tactile frames in an ACT-local causal buffer. `reset()` clears that buffer before the first observation of every episode. The same repeat-first padding and stride rule used by training constructs `[1,sensors,T,3,H,W]`. No simulator, camera, action or normalization logic changes.

## Training and evaluation

- Encoder: same HDMI clean split, batch 32, AdamW, LR `1e-4`, weight decay `1e-4`, five epochs, depth/marker heads unchanged, plus delta-marker weight `0.5`.
- Policy: exactly 4000 optimizer updates, physical batch 32, gradient accumulation 2, effective batch 64, AdamW, all three LR groups `1e-5`, same 50 demonstrations and seed/split semantics, final `policy_last.ckpt`.
- Sanity gates: causal/padded indices, sequence and output shapes, finite forward/backward, nonzero dynamic attention/head/gate gradients, exact warm-start report, strict checkpoint reload, frozen policy BN, unchanged camera/qpos/action tensors, deployment reset, and multi-step ACT smoke.
- Quick triage: seeds `1000000..1000019`. Continue to `1000000..1000099` when the result is materially better than the available B0/Spatial evidence. Existing HDMI `31/100` artifacts that encode seeds `0..99` are retained but not presented as a paired seed comparison.
- If the first version is below `33/100`, retain its artifacts and permit at most one isolated temporal-receptive-field change followed, if justified, by one isolated dynamic-strength change.


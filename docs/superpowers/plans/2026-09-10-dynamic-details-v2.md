# Dynamic Details V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, pretrain, integrate, train and fixed-seed evaluate the causal Dynamic Details V2 tactile encoder for Insert HDMI.

**Architecture:** Reuse every Spatial V2 static parameter and add layer2 feature residual tokens, a second global-query cross-attention, gated residual fusion and a delta-marker auxiliary head. Only tactile receives a four-frame causal window; the external encoder and ACT interfaces remain 512-D and otherwise unchanged.

**Tech Stack:** Python 3, PyTorch/torchvision, h5py, pytest, ACT, Isaac Sim runtime.

**Spec:** `docs/superpowers/specs/2026-09-10-dynamic-details-v2-design.md`

## Global Constraints

- Insert HDMI is the only first-round task.
- Preserve Spatial V2 and every existing artifact; use a new experiment directory.
- `sequence_length=4`, `temporal_stride=1`, `token_dim=256`, `num_heads=4`, `dropout=0`, initial sigmoid gate `0.15`, and `lambda_dynamic=0.5`.
- No future tactile frames and no history across HDF5 trajectories, sensor sides or rollout episodes; repeat the episode's first frame for missing history.
- Encoder output is exactly `[B,512]`; ACT, camera, qpos/action normalization, action chunk and ACT loss are unchanged.
- Policy training is exactly 4000 optimizer updates with physical batch 32, gradient accumulation 2, AdamW, three LR groups at `1e-5`, the existing split, and `policy_last.ckpt`.
- Formal evaluation uses exactly `1000000..1000099`, with `1000000..1000019` as quick triage.

---

### Task 1: Dynamic encoder and strict Spatial V2 warm-start

**Files:**
- Modify: `encoder/detail_v2.py`
- Test: `tests/test_dynamic_v2.py`

**Interfaces:**
- Produces: `DynamicDetailV2Encoder.forward([B,T,3,H,W]) -> [B,512]`.
- Produces: `DynamicDetailV2Encoder.forward_with_details(...) -> (embedding, detail_dynamic)`.
- Produces: `warm_start_dynamic_encoder(source, **structure) -> (encoder, report)` with exact compatible/new key accounting.
- Extends: `build_encoder`, encoder-only export/load checkpoint contract with `detail_v2_dynamic`.

- [ ] Write tests that fail because `detail_v2_dynamic` does not exist: 512-D shape, 1024 positions per transition at 256 input, three temporal transitions, gate value 0.15, static/dynamic/backbone nonzero gradients, invalid sequence rejection, exact Spatial V2 compatible-key warm-start, and strict checkpoint round-trip.
- [ ] Run `/opt/anaconda3/bin/python -m pytest -q tests/test_dynamic_v2.py` and confirm failures are missing-feature failures.
- [ ] Implement the shared-trunk dynamic encoder, residual tokens, temporal encoding, dynamic attention, gate/fusion, factory and checkpoint extensions without renaming the existing static modules.
- [ ] Run the focused test and `tests/test_detail_v2.py`; both must pass.
- [ ] Commit only Task 1 files with message `feat: add dynamic local tactile encoder`.

### Task 2: Causal encoder dataset, delta-marker audit and pretraining loss

**Files:**
- Create: `encoder/dynamic_data.py`
- Modify: `encoder/pretrain_v2.py`
- Create: `scripts/audit_dynamic_v2.py`
- Modify: `scripts/train_encoder_v2.py`
- Create: `configs/encoder_dynamic_v2_hdmi.json`
- Test: `tests/test_dynamic_data.py`
- Test: `tests/test_dynamic_p0.py`

**Interfaces:**
- Produces: `causal_history_indices(anchor, sequence_length, temporal_stride) -> tuple[int,...]`.
- Produces: `DynamicCleanDataset(...).__getitem__ -> ([T,3,H,W], targets)` where targets include current `depth`, current `marker`, and normalized `delta_marker`.
- Extends: `P0Model(..., dynamic_weight=0.5)` to consume `detail_dynamic` only for the auxiliary head.
- Extends: encoder training CLI with required Spatial V2 warm-start metadata for dynamic runs.

- [ ] Write failing HDF5 fixture tests for `t=0/1/3`, stride handling, repeat-first padding, maximum history index equal to anchor, same file/side, current-target alignment, raw delta formula and train-only delta statistics.
- [ ] Write failing P0 tests for the weighted three-loss equation, `[1200,2]` prediction and nonzero gradients in dynamic attention/head/gate/trunk.
- [ ] Run the two focused test files and confirm expected missing-feature failures.
- [ ] Implement the pure index helper, dynamic dataset, audit, auxiliary head/loss, warm-start CLI/config and provenance fields.
- [ ] Run the two focused tests plus `tests/test_clean_v2.py tests/test_p0.py tests/test_detail_v2.py`.
- [ ] Commit only Task 2 files with message `feat: pretrain dynamic tactile details`.

### Task 3: ACT training and deployment history integration

**Files:**
- Modify: `policy/ACT/utils.py`
- Modify: `policy/ACT/detr/models/backbone.py`
- Modify: `policy/ACT/act_policy.py`
- Modify: `scripts/train_policy_v2.py`
- Modify: `scripts/prepare_eval_v2.py`
- Create: `configs/policy_dynamic_v2_hdmi.yml`
- Test: `tests/test_dynamic_policy.py`
- Modify: `tests/test_eval_v2.py`

**Interfaces:**
- Training tactile tensor: `[B,sensors,T,3,H,W]`; camera/qpos/action tensors retain their existing shapes and values.
- Tactile backbone consumes `[B,T,3,H,W]` and returns one `[B,512,1,1]` feature.
- Deployment ACT owns its tactile history; `reset()` empties it and first-frame padding reconstructs a full window.

- [ ] Write failing tests proving anchor-aligned training windows, unchanged camera/qpos/action values, dynamic checkpoint selection, FrozenBatchNorm behavior, deployment repeat-first history, stride and reset isolation, and prepared runtime acceptance of `detail_v2_dynamic`.
- [ ] Run focused tests and confirm expected missing-feature failures.
- [ ] Implement optional temporal arguments with defaults that leave Spatial V2 behavior unchanged; add dynamic checkpoint loading and deployment buffering.
- [ ] Generalize the policy trainer and runtime validator without changing the 4000-update, batch32×2 or three-LR contracts.
- [ ] Run focused tests plus all existing `tests/` and compile modified Python files.
- [ ] Commit Task 3 files with message `feat: integrate dynamic tactile history with ACT`.

### Task 4: Remote sanity, formal HDMI run and fixed-seed triage

**Files:**
- Create remotely under experiment root: `dynamic_v2_20260910/{source,audit,runs,results,logs}`.
- Record tracked summaries in: `results/dynamic_v2_hdmi_status.md` and small JSON/CSV result files only.

**Interfaces:**
- Warm-start source: `/usr1/home/s126mdg41_04/details_v2_20260909/runs/encoder_v2_seed42/best.pt`, expected SHA256 `914d9d8315a8ccd7fd88ea853b20375a90010ac45968e88e9d2eb386abffe448`.
- Clean data: server Insert HDMI clean trajectories and existing split manifest.
- Policy data: server `sim-insert_HDMI/demo-50` 50-demonstration directory.

- [ ] Sync the committed source to a new immutable remote source directory; record source commit and verify no old artifact path is targeted for writes.
- [ ] Run the dynamic data audit and assert split hash, train-only delta normalization, sequence configuration, finite targets and SHA256 records.
- [ ] Run CPU/unit tests in the server environment, then encoder forward/backward, strict warm-start/reload, dynamic-gradient and small real-data overfit smoke on the available GPU.
- [ ] Run ACT two-update smoke and a true multi-step simulator smoke; verify dynamic history changes, episode reset, action updates and frozen BN.
- [ ] Launch five-epoch HDMI encoder pretraining, verify completion/checkpoint reload/SHA256, then launch the unchanged 4000-update ACT training and verify exactly 8000 micro-batches and `policy_last.ckpt`.
- [ ] Evaluate seeds `1000000..1000019`, compare exact outcomes with any trustworthy matched artifacts, and continue with `1000000..1000099` only when quick triage is materially positive.
- [ ] If the full result is below `33/100`, preserve it and execute at most the two isolated variants allowed by the spec, one principal factor at a time.
- [ ] Commit and push implementation, working pretraining/ACT milestones, and formal result to `origin/codex/dynamic-v2`; exclude datasets, checkpoints, simulator assets, credentials and absolute secret configuration.


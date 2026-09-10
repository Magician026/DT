# Task 3 Report: ACT training and deployment history integration

## Outcome

- Added causal tactile windows to `TacArenaDataset` while retaining the exact static return shape when `sequence_length=1`.
- Added strict `detail_v2_dynamic` checkpoint loading to `TactileBackbone`; canonical preprocessing, class identity, 512-D output, and frozen BatchNorm behavior are checked.
- Added per-sensor ACT-local deployment history with repeat-first padding, temporal stride selection, and episode reset isolation.
- Generalized policy training and evaluation preparation to the encoder type declared in config.
- Preserved the formal ACT contract: 4000 optimizer updates, physical batch 32, gradient accumulation 2, effective batch 64, exactly three optimizer groups, and every group at `1e-5`.
- Added the Insert HDMI dynamic policy config as an exact value-level copy of `configs/policy_v2.yml` except for encoder type, sequence length, and temporal stride.

## TDD evidence

### Initial RED

Command:

```text
/opt/anaconda3/bin/python -m pytest -q tests/test_dynamic_policy.py tests/test_eval_v2.py
```

Result: `6 failed, 8 passed`. Expected missing-feature failures were:

- `TacArenaDataset.__init__` rejected `sequence_length`.
- `TactileBackbone` rejected `detail_v2_dynamic`.
- deployment produced single-frame tactile input.
- dynamic policy config was absent.
- dynamic evaluation was rejected by the static-only validator.
- training contract helper was absent during the first collection attempt; the import was moved to module-level access so the behavioral RED suite could run fully.

### Optimizer contract RED

Command:

```text
/opt/anaconda3/bin/python -m pytest -q tests/test_dynamic_policy.py::test_formal_dynamic_config_preserves_act_and_optimizer_contract
```

Result before the minimal fix: `1 failed`; smoke validation did not reject a two-group optimizer. The contract check was then made common to smoke and formal construction.

### GREEN and regression verification

Commands and results:

```text
/opt/anaconda3/bin/python -m pytest -q tests/test_dynamic_policy.py tests/test_eval_v2.py
14 passed

/opt/anaconda3/bin/python -m pytest -q tests/test_dynamic_policy.py tests/test_policy_v2.py tests/test_eval_v2.py
23 passed

/opt/anaconda3/bin/python -m pytest -q tests/
80 passed in 8.90s
```

The dynamic policy test includes a real canonical Dynamic V2 tactile backbone forward/backward pass and verifies nonzero static attention, dynamic attention, and gate gradients plus frozen BatchNorm affine/statistics.

## Files

- `policy/ACT/utils.py`
- `policy/ACT/detr/models/backbone.py`
- `policy/ACT/act_policy.py`
- `scripts/train_policy_v2.py`
- `scripts/prepare_eval_v2.py`
- `configs/policy_dynamic_v2_hdmi.yml`
- `tests/test_dynamic_policy.py`
- `tests/test_eval_v2.py`
- `.superpowers/sdd/2026-09-10-dynamic-details-v2/task-3-report.md`

## Assumptions and residual risks

- Deployment observations continue the existing contract of supplying transformed tactile tensors; the history stores detached clones so later observation mutation cannot rewrite earlier frames.
- Static Details V2 remains the compatibility default (`sequence_length=1`, `temporal_stride=1`).
- GPU ACT smoke, real-data training, and simulator multi-step evaluation are intentionally deferred to Task 4. This task validates the canonical tactile encoder on CPU and does not claim simulator evidence.
- Task 3 commit: the commit containing this report, with subject `feat: integrate dynamic tactile history with ACT`. Its resolved hash is reported by `git rev-parse HEAD` in the parent handoff because a commit cannot embed its own final object hash.

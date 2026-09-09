# Details V2 baseline audit

Audit date: 2026-09-09 (Asia/Singapore)

## Decision

`MATCHED_BASELINE_NOT_AVAILABLE`

No existing policy checkpoint was found that satisfies the required paired
comparison for the current Details V2 run. In particular, the same-data B0
encoder is complete, but a downstream B0 policy trained with the current
Insert HDMI protocol does not exist in the inspected artifacts. No baseline
evaluation was started.

## Current Details V2 protocol

| Field | Details V2 evidence |
|---|---|
| Task / simulator | `insert_HDMI`, `demo`, ACT deployment; official external runtime with the recorded environment hashes |
| Policy checkpoint | `policy_seed42/policy_last.ckpt`, SHA256 `3047f071db09e307d9bbbf942b70ef1b49977c1c0935a7a76a676fec3c720d52` |
| Encoder | `encoder_v2_seed42/best.pt`, `encoder_type=detail_v2`, SHA256 `914d9d8315a8ccd7fd88ea853b20375a90010ac45968e88e9d2eb386abffe448` |
| Policy source | immutable release commit `3cd75efd5b80aa14bc0ba13900c9b2e6e942bea8` |
| ACT data | separate 50-episode dataset; policy split seed 1, 40 train / 10 validation; normalization uses all 50 episodes |
| Policy training | ACT, 4,000 optimizer updates, physical batch 32 × accumulation 2; LR groups `1e-5`, weight decay `1e-4`; policy seed 42 |
| Encoder pretraining | 100 clean trajectories; 21,240 train / 2,360 validation sensor frames; P0 depth + marker targets; split SHA256 `437a7bad6610b5f84888149ab6416b5f61ae1e28a5db6d2bda1e0c17aafd0e63`; 3,320 updates |
| Evaluation protocol | fixed rollout seed ranges, official task success predicate, no duplicate counting; current V2 evaluation uses the same V2 checkpoint/config for all seeds |

## Candidate audit

| Candidate artifact | Encoder / pretraining | Policy / data / steps | Task / simulator | Comparability |
|---|---|---|---|---|
| `encoder_b0_seed42/best.pt` | Original ResNet18; same V2 clean-data split and P0 target recipe; 3,320 updates; SHA256 `e7127332f08435632fdd4681b04bff60a4ebe9426eefe5f825215919a3e58063` | No downstream policy checkpoint | N/A | **Potential encoder control only**; cannot run a policy comparison |
| Git history `encoder/ablation/data_1000/resnet18/20260904-181027/best.pth` | Legacy Original ResNet18 on the older 1,000-sample / full-supervision recipe; not the current all-frame P0 recipe | No matching Insert HDMI policy checkpoint in the repository artifacts | Historical Insert HDMI encoder work | **Encoder reference only; reject** |
| `baseline_original/policy/pull_out_key/original200k_seed0` | Original ResNet18 from the separate 200K encoder recipe; encoder SHA is not the current B0 artifact | ACT, 4,000 updates, physical batch 64 × accumulation 1, policy seed 0; policy SHA256 `db731d39d162cf4ad14d29a97ad26a133ee6eecdddf454762e491421b9fff526` | `pull_out_key`, not `insert_HDMI`; separate simulator task/evaluation seeds | **Reject**: task, pretraining data/recipe, and training resource protocol differ |
| `baseline_original/policy/pull_out_key/released_control_seed0_v1` | Original/released ResNet18 control; different encoder provenance | ACT, 4,000 updates, physical batch 64 × accumulation 1; policy SHA256 `d9a4acc05ba447e364a0f568ce11c2a68828f2e1d63309afeadbbe5c08116fc7` | `pull_out_key`, not `insert_HDMI` | **Reject**: wrong task and different protocol |
| `baseline_original/policy/pull_out_key/physical64_preflight` | Original ResNet18 preflight | Only 3 updates; not a formal policy | `pull_out_key` | **Reject**: preflight and wrong task |
| `insert_HDMI` policy tree on cluster historical workspace | No policy checkpoint present in the inspected static or dynamic HDMI directories | No completed policy artifact or matching config/metrics | Historical HDMI work, protocol not established as current V2-compatible | **Unavailable** |
| Historical HDMI success figures (`18/100`, `15/100`) | Historical encoder/policy recipes are different and provenance is not the current B0 downstream run | Policy/config/simulator alignment is incomplete | HDMI, but not the current controlled protocol | **Reference only; reject as baseline** |

## Comparability checks

The current V2 and the completed B0 encoder share the clean-data manifest,
trajectory split, P0 targets, five-epoch budget and 3,320 encoder updates.
That establishes a controlled encoder pretraining comparison. It does not
establish a downstream baseline because the B0 policy stage is missing.

The completed Original policy artifacts are both `pull_out_key` runs. They
use a different task, different encoder pretraining provenance, and physical
batch 64 without accumulation, so their success rates cannot be used as a
strict B0 comparison for V2 `insert_HDMI`.

No artifact was found that simultaneously provides all of the following:

- Original ResNet18 tactile encoder;
- current Insert HDMI task and 50-demo ACT dataset;
- current ACT architecture and 4,000 optimizer updates;
- current policy split/statistics and training resource protocol;
- current official evaluation environment and success criterion.

## Action for the parent controller

Do not launch a B0 paired evaluation from the inspected historical
checkpoints. Preserve the V2 evaluation artifacts, report the baseline as
unavailable, and only create a B0 downstream policy as a separate authorized
experiment after the V2 evidence is complete.

# Dynamic Details V2 — Insert HDMI Status

Updated: 2026-09-11 (Asia/Singapore)

## Implementation and data contract

- Source release: `b89eb936f5cb25a9f5e06eced2bfdc52072669a5`
- Encoder type: `detail_v2_dynamic`
- Temporal input: four causal frames, stride 1, repeat-first episode padding
- Output: 512 dimensions
- Local/server regression suite: 83 passed
- Encoder split: 90 train / 10 validation trajectories, hash `437a7bad6610b5f84888149ab6416b5f61ae1e28a5db6d2bda1e0c17aafd0e63`
- Train-only delta-marker normalization: mean `-0.0006162313627608028`, std `0.2204084216995208`

## Formal encoder pretraining

- Status: completed
- Optimizer steps: 3320 over five epochs
- Best full-validation total loss: `0.05761060030278513`
- Checkpoint SHA256: `6d0079ddbe396041d592080c7287c2700f3956cc356b5be5c5c008cdcb7beea9`
- Spatial V2 warm-start checkpoint SHA256: `914d9d8315a8ccd7fd88ea853b20375a90010ac45968e88e9d2eb386abffe448`
- Strict warm-start accounting: 136 compatible Spatial keys, 12 new Dynamic keys
- Learned dynamic gate after pretraining: `0.14849716424942017`

## Current stage

- Formal ACT policy training completed on the immutable source release in 4,174.91 seconds.
- Completion contract verified: 4000 optimizer updates, 8000 micro-iterations, physical batch 32 × accumulation 2, and three optimizer groups at `1e-5`.
- Policy checkpoint SHA256: `ebff7456160dd312af30cf350edbdafa27da6a93fdb758148f6eecc91547e40a`
- Strict policy checkpoint reload passed; validation loss decreased from `0.170235889` at step 500 to `0.087586921` at step 4000.
- Quick triage completed on seeds 1000000–1000019: 5/20 successes (25%), with 20 valid episodes and zero infrastructure errors.
- Formal evaluation completed on seeds 1000000–1000099: 29/100 successes (29%), 100 valid episodes, no duplicate or missing seed, and zero infrastructure errors.
- Full-eval outcomes SHA256: `196775418281c6ff20f4a902fa622ec3ad2d577b472d59f13f040fe01397e34d`
- Result is +1 absolute point versus UniVTAC B0 (28/100), -2 points versus Spatial V2 (31/100), and below the minimum Dynamic V2 target of 33/100.
- The next bounded iteration is Variant A: retain the architecture and training recipe, changing only T=4 temporal stride from 1 to 2.
- Historical Spatial V2 HDMI outcomes use seeds 0–99; they are not treated as paired evidence for the required million-seed set.

## Variant A — T=4, temporal stride 2

- Source release: `585d40e5ac237b3e107f1a9c66407f6e65da45fb`; the only experiment factor changed from Dynamic V2 is temporal stride 1 to 2.
- Local regression suite: 86 passed, including strict stride-2 encoder checkpoint reload, ACT contract, and eval metadata consistency.
- Reused split hash: `437a7bad6610b5f84888149ab6416b5f61ae1e28a5db6d2bda1e0c17aafd0e63`.
- Train-only delta-marker normalization: mean `-0.0036338853887569757`, std `0.3478502683947113`.
- Formal encoder pretraining completed 3320 steps over five epochs in 607.38 seconds; best validation loss `0.059451635863821385`.
- Encoder checkpoint SHA256: `d33fc034d47af4b05541f1813c7e6fb047d090983dfb34d3faef4c07299ed115`; strict reload verified T=4, stride=2 metadata.
- Formal ACT policy training completed 4000 optimizer updates / 8000 micro-iterations in 3681.10 seconds with physical batch 32 × accumulation 2, effective batch 64, and all three learning rates at `1e-5`.
- Policy checkpoint SHA256: `02d2dda3fa2901f4ab4d44d9e68aeddf0be9eec1f0516a8423c29eb4fd6e52a5`; dataset stats SHA256: `9a92d7fe5873612d71240ecc667679ea517aadf0561e640a7d68571f02fc272b`.
- Strict ACT checkpoint reload passed with zero missing or unexpected keys; validation loss decreased from `0.172873404` at step 500 to `0.089168837` at step 4000.
- Quick triage is running on the fixed seeds 1000000–1000019 in isolated run `dynamic_v2_stride2_20260910/runs/eval_quick20_dynamic_v2_hdmi_stride2`; launched on an idle GPU 3 with launcher PID `898611` and Isaac child PID `900487`.
- Quick-eval manifest verifies source commit `585d40e5ac237b3e107f1a9c66407f6e65da45fb`, T=4 stride=2, formal `policy_last.ckpt`, 4000 optimizer updates, and the expected checkpoint/stat/encoder SHA256 values.
- Encoder, ACT-interface, and three-step simulator smoke artifacts were deleted after the formal workloads were verified live.

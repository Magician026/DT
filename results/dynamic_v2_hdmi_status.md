# Dynamic Details V2 — Insert HDMI Status

Updated: 2026-09-10 (Asia/Singapore)

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

## Sanity checks

- Encoder physical-batch-32 peak allocated memory: 4,884,893,184 bytes
- Fixed-real-batch encoder overfit: first-five mean `1.8731478929519654`, last-five mean `0.884002411365509`
- ACT physical batch 32 × accumulation 2 probe: passed; peak allocated memory 11,825,753,600 bytes
- ACT interface: `[32, 50, 8]` actions, three optimizer groups at `1e-5`, static and dynamic attention update, BatchNorm frozen
- Simulator smoke: passed three action steps on seed 1000000; three unique tactile hashes and two detected transitions; one episode reset

## Current stage

- Formal ACT policy training is running on the immutable source release.
- Required completion contract: 4000 optimizer updates, 8000 micro-iterations, physical batch 32 × accumulation 2.
- Next evaluation: quick triage on seeds 1000000–1000019, followed by 1000000–1000099 only if the quick result is materially positive.
- Historical Spatial V2 HDMI outcomes use seeds 0–99; they are not treated as paired evidence for the required million-seed set.

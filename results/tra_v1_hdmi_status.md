# Details V2-TRA v1 — Insert HDMI

Updated: 2026-09-11 (Asia/Singapore)

## Locked experiment

- Source: `5ac9bfe239e73c72ea92ec3918bd23099ff23697`; branch `codex/details-v2-tra`.
- Remote root: `/usr1/home/s126mdg41_04/UniVTAC details v2/tra_v1_20260911`; immutable source `source_5ac9bfe`.
- Anchor: Spatial pretrained encoder SHA256 `914d9d8315a8ccd7fd88ea853b20375a90010ac45968e88e9d2eb386abffe448`, associated with historical Spatial policy 31/100.
- Initial TRA checkpoint: `runs/encoder_tra_init/init.pt`, SHA256 `bbfbd856e2e0a473cd309189ebb4af3e7d59bb7a04c17aa4a0a937204980a99c`. No encoder pretraining or extra loss.
- Fixed T=4, stride=1; current global query attends separately to four shared Spatial detail representations; GRU(256,256,1), Linear/GELU/Linear residual, sigmoid(alpha)=0.1, final LayerNorm, 512D ACT interface.
- 4000 optimizer updates, physical32 × accumulation2; seed42; split seed1, same40/10 episodes; all50 normalization; AdamW and all three LR groups1e-5; same ACT/vision/state/action chunk.
- Freeze inherited Spatial trainable parameters for updates1–400, then restore normal fine-tuning. BN affine/statistics remain frozen as in Spatial policy.
- Policy data: canonical `policy/ACT/data/sim-insert_HDMI/demo-50`; statistics SHA256 `9a92d7fe5873612d71240ecc667679ea517aadf0561e640a7d68571f02fc272b`; split hash `b40d2521b5e689d03d7ff22ba0f79d7a7e81c64c40d91a720be58d27c9aae70f`.

## Evidence and current stage

- Prior Dynamic route closure committed/pushed before implementation (`f3de785`): B0 28/100, Spatial31/100, Dynamic29/100, A1/20, B5/20, original quick5/20. Old artifacts preserved.
- Local full `tests/` suite:112 passed before pipeline helper; pipeline threshold checks8 passed. Targeted encoder43, ACT/eval21 and training24 are overlapping subsets, not additional independent trials.
- Warm-start:136 inherited state keys match exactly;11 new keys. On four real training examples (episode27, both sensors, anchors0 and3), inherited current representation max error0; gate≈0 final512D cosine0.952906–0.962852, relativeL2 0.312386–0.363021. The specified extra final LayerNorm changes scale even at zero residual; no calibration or architecture change was introduced. Initial gate0.1 cosine0.952880–0.962827. Strict reload exact.
- Policy GPU smoke completed2 optimizer updates /4 micro-iterations at32×2. Smoke-only freeze1 exercises both warmup and actual unfreeze; formal remains400. Nonzero GRU/alpha gradients, alpha update, zero inherited-weight change during warmup, frozenBN, forward/backward and strict reload passed. Policy split/statistics match Spatial exactly.
- Simulator smoke passed:3 actions,3 tactile observations,2 changes,3 unique hashes,1 reset; exact TRA source/encoder/stats/T4/stride1 provenance. Formal pipeline subsequently launched on GPU2 with24230MiB free (launcher PID3057038; remote logs/pipeline.pid records it).

## Execution and triage

- Formal policy run: `runs/policy_tra_v1_hdmi`; realtime log: `logs/policy_tra_v1_hdmi.log`.
- Pipeline launcher: `source_5ac9bfe/scripts/run_tra_v2.py`; status `results/pipeline_status.json`.
- Quick: fixed seeds1000000–1000019 copied byte-for-byte from Dynamic manifest, SHA256 `9361cd7bcc4ad3c121d2e0532ed52c4758b79c54a1858396eca11c3d10341862`.
- <=5/20 stops,6/20 requires bounded curve/failure review,>=7/20 launches full100 on an idleGPU. No stride/gate/loss/hidden-size/sequence/optimizer sweep.
- Full100 uses seeds1000000–1000099 and policy_last.ckpt. No intermediate success-rate stopping.
- Spatial31/100 used historical seeds0–99, so comparison with TRA is not seed-paired.
- If full result<=31/100, stop this temporal architecture and revisit the task bottleneck;33 minimum,35 target,36 strong.

## Delegation provenance

Real tool roles were used: one `luna_executor` read-only anchor audit and two `spark_coder` workers (encoder and training). Tool-advertised fixed configurations are luna=gpt-5.6-luna/high and spark=gpt-5.6-sol/high, differing from AGENTS.md requested model/effort descriptors. Actual backend model/effort fields were not returned and are unverified; no model switch was simulated.

## Formal launch update

Formal pipeline PID3057038 and training child3057201 run on GPU2. Verified step25/4000,50 micro-iterations, loss6.349237, warmup stage, gate0.09999261, nonzero GRU/alpha gradients. Smoke runtime/checkpoints/logs/config/seed and formal interface_smoke.json were deleted only after real progress; exact cleanup paths are archived in canonical operations. Formal training, initialization, seed manifests and tests remain.

Long-run follow-through: server pipeline automatically chains training to quick/full under the fixed thresholds. Thread heartbeat `follow-tra-hdmi-to-evaluation` runs every30minutes to verify completion/failure, handle6/20 with a bounded review, and sync final evidence; it stays quiet when healthy and unchanged and is removed after completion.

Latest verified stage: formal update425/4000 (850micro), loss2.109101, gate0.10002768. The log records the exact transition after400 completed updates to joint_finetune with Spatial trainable. Post-unfreeze forward/backward proceeds with finite nonzero temporal gradients; peak allocated GPU memory12,036,379,136bytes. No hyperparameters changed. Quick evaluation is pending training completion.

First validation checkpoint: update500/4000,1000micro; train_loss1.963796854, val_loss0.179113835, stagejoint_finetune, sigmoid(alpha)0.100032598. This is a training metric, not a success-rate result.

Data equivalence verified: all50 canonical ACT HDF5 files are byte-identical to the historical Spatial input files (fullSHA256 comparison). Evidence: results/tra_v1_data_equivalence.json.

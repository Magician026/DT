# Details V2-TRA v1 — Insert HDMI

Updated: 2026-09-11 (Asia/Singapore)

## Final disposition — stopped after quick20

Verified 2026-09-11 14:44 Asia/Singapore. **TRA v1: 4/20 (20%)**. The single pipeline completed quick20 at 13:07 and exited with `stopped_after_quick`. Under the locked `<=5/20` rule, this version is closed: no full100, retraining, pretraining loss, or sweep. The full100 run directory does not exist; no process referencing this TRA root remains. All formal checkpoints, configurations, logs, outcomes and immutable source snapshots are retained.

| Model | Insert HDMI result | Evaluation seeds |
| --- | --- | --- |
| UniVTAC B0 | 28/100 | Historical baseline |
| Spatial Details V2 | 31/100 | Historical 0–99 |
| Details V2-TRA v1 | **4/20 quick**; full100 not run | 1000000–1000019 |

The quick result does not support advancement under the prespecified gate. It is not a measured full100 success rate and does not establish that temporal tactile information has no value. Spatial's historical seeds0–99 differ from TRA's seeds, so this is **not a paired comparison**. No architecture or training recipe was changed after seeing results.

- Quick has exactly20 valid seeds1000000–1000019, no missing/duplicate seeds and zero infrastructure errors. Success seeds:1000005,1000007,1000008,1000016. Outcomes SHA256 `25a8d68feaf993507e74999bfac2ad23d6ed20793c53464cbf53588ddde311ac`.
- Completion/manifest provenance matches the locked source5ac9bfe, policy_last.ckpt, datasetstats, initialencoder, T4/stride1 and seedmanifest. Training checks remain passed:4000updates/8000micro/32×2, actual unfreeze after400, nonzeroGRU/alpha gradients, alpha update and strictreload.
- DNS access recovered by this verification without changing SSH configuration or restarting any job. The earlier interruption below is historical, not a current blocker.
- Final evidence: `tra_v1_quick_completion.json`, `tra_v1_quick_outcomes.jsonl`, `tra_v1_quick_manifest.json`, `tra_v1_quick_summary.json`, `tra_v1_pipeline_status.json`, `tra_v1_quick_decision.json`, `tra_v1_final_verification.json`, and `tra_v1_hdmi_result.json` in this directory.
- Experiment processing is complete; remove `follow-tra-hdmi-to-evaluation` after synchronizing final records. The old Dynamic monitor remains paused and that route remains stopped.

The dated execution and access records below are retained as history; this final disposition supersedes their then-current stages.

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

Heartbeat 2026-09-11 12:00 Asia/Singapore: pipeline/trainer remain healthy onGPU2; observed2825/4000updates,5650micro, trainloss0.414801, latest validation at2500=0.101571944, sigmoid(alpha)=0.100143574 and nonzeroGRU/alpha gradients. No completion or quickeval yet; same recipe/stage, no intervention. No reusable operations facts changed.

## Training completed; quick evaluation running (2026-09-11 12:31)

- Formal ACT completed4000 optimizer updates /8000micro in3294.03s, physical32×2 and all3LR groups1e-5. Freeze applied400updates, then jointfine-tune; final GRU/alpha gradients nonzero.
- Final/best validation loss0.0933153480; final trainloss0.2213700339. Initial gate0.1000000015→final0.1001756340.
- Final policy_last.ckpt size386,062,104bytes, SHA256 `148b28e2c26d962f9747272cf431d64c18b80d91fab68c98b924cb80c883ff18`. SHA independently recomputed; datasetstats and initencoder SHA also match locked values. Immutable trainer strict-loads the saved checkpoint before writing completion; liveeval loaded the same checkpoint.
- Automatic quick20 started only after GPU2 was fully idle (17MiB,0%util,no compute processes). Manifest matches source, T4/stride1, exact policy and fixed seeds1000000–1000019.
- At12:31, first3valid episodes were0/3; seed1000003 was running. This is partial progress, not a quick20 result, and is not a stop criterion.
- Evidence: `results/tra_v1_policy_completion.json`, `results/tra_v1_policy_metrics.jsonl`, `results/tra_v1_policy_verification.json`. No reusable operating constraints changed; no newjob submitted by the heartbeat.

## Access interruption (2026-09-11 13:18)

The heartbeat could not connect: mlda2 resolves to gpu41.dynip.ntu.edu.sg, which returnsNXDOMAIN via localDNS,1.1.1.1,and both NTU authoritative DNS servers. No verified alternateIP was present in scoped existing connection records. Evidence: `results/tra_v1_connection_issue.json`.

No remote job or artifact was changed. Current quick/full result and process state are unknown; DNS failure does not establish experiment failure. Last verified state remains trainingcomplete4000updates and partialquick0/3 at12:31. The original single-run pipeline and30minuteheartbeat remain configured; do not duplicate/restart experiments because monitoring cannot connect. Retry access on the next scheduled check; notify again only if access status meaningfully changes or a different actionable issue appears. This is a transient connection observation, not a new stable server constraint, so canonical operations is unchanged.

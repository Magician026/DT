# Details V2: Insert Tube and Pull Out Key

User scope: task-specific V2 encoder pretraining, followed by ACT policy training with seed 0 and simulation evaluation. No new architecture or loss changes.

| Field | Insert Tube | Pull Out Key |
|---|---|---|
| Task | insert_tube | pull_out_key |
| Encoder training seed | 1226746006 | 821018648 |
| Policy training seed | 0 | 0 |
| Cameras | cam_high + cam_wrist | cam_high |
| Policy data | clean-50 | clean-50 |
| Evaluation seeds | 1000000–1000099 | 1000000–1000099 |
| Official action limit | 300 | 300 |

Encoder seeds were generated once from OS entropy and retained before training; they are not selected by results. Encoder split seed remains 42, policy split seed remains 1. Each task uses its own clean encoder dataset and train-only normalization. Tube P0 retains depth + marker MSE; Key uses depth-only MSE under attachment P0 rule B because marker reference semantics are unverified. Both use, five epochs, batch32 with normal encoder BatchNorm, lr1e-4 and weight decay1e-4. Policy retains 4000 updates, physical32 × accumulation2, LR1e-5, final checkpoint selection, and fine-tuned tactile encoder with fixed BN as before.

The original released-encoder baselines have already been completed (documented protocol: Tube63/100, Key51/100). These are reference results, not a matched encoder-pretraining ablation: released encoder pretraining provenance differs from P0, and old micro-batch handling differs at epoch boundaries. The new V2 runs use exact rollout seed manifests and stop on infrastructure errors; historical evaluation allowed excluding exceptions and incrementing seeds. Any exception must be audited before claiming strict comparability. No historical result will be overwritten.

Execution: serial GPU2; task-specific encoder smoke and policy/simulator smoke precede formal policy. The persistent runner records stages, child PIDs, source commit, logs, completion/hash checks, and fails closed. Existing audit process is reused. No training seed42 task is submitted; uncommitted seed42 draft configs are not used.

Live stage: Insert Tube encoder smoke passed (50 updates; fixed-batch loss 1.2743→0.8395), formal encoder PID1726270 on GPU2 using immutable encoder code79b0913; 33606 train /3952 validation sensor frames. Pull Out Key original audit stopped on nonstationary reference in10.hdf5 (right sensor after frame132, max234.4pixels; not a simple permutation). No marker supervision is claimed. New isolated depth-only audit PID2092406; no Key training has started. Authoritative live status will be `details_v2_tasks_20260910/status.json` and per-task run directories, not this preparation snapshot.

Stage update: Tube encoder completed5255updates, bestval0.03204907; policy smoke completed2updates. Three-step simulator smoke failed because all3 tactile hashes were identical while3 actions advanced. Retained eval_smoke; same-checkpoint30-step diagnostic in eval_smoke_30 started. Key depth-only data audit completed successfully. No formal policy or task success result yet.

Recovery: same-checkpoint30-step simulator diagnostic passed with30 actions,30 unique tactile observations and29 changes. Three-step failure was not reproduced; precise startup cause remains unconfirmed. Extended smoke coverage is an engineering preflight only; formal protocol and model unchanged. Runner reuses successful eval_smoke_30; old failed eval_smoke preserved.

2026-09-10 09:06 Asia/Singapore: Tube policy seed0 completed4000updates/8000microiterations; checkpoint SHA256786021c67c63ec246c2f4663e6428335b0953c16f180b4810284b2e4f56e7f2e independently checked. Formal100-seed eval running GPU2, launcher3904307, controller687252; observed completed through seed1000003. No final success rate yet. Key remains queued.

Tube final:58/100,58%, Wilson95%CI48.2–67.2%, exact100 unique seeds,0 infrastructure errors. Completed2026-09-10 10:49:37 Asia/Singapore. Historical released encoder63/100 differs by−5pp descriptively; this is not a controlled encoder architecture effect because pretraining provenance differs. Key encoder smoke passed and formal depth-only encoder PID2592609 is running GPU2, observedstep221; policy seed0 and100-rollout eval remain pending.

2026-09-10 13:33 Asia/Singapore: Key policy seed0 completed4000updates/8000microiterations; formal eval active under new canonical root, launcher2454869/watcher692452. Verified83 completed rollouts,30 successes so far (partial). Policy SHA256951b1f2a80230deef1efd326b7c89690bc44adec1a3d195bc34d13df8182e6a3. Comparison reporting now uses only official published task rates per user; prior user reproduction numbers are not the comparator.

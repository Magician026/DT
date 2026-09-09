# 1 Current Status

Updated: 2026-09-09T22:52:08.714927+00:00

Encoder: completed (V2 and same-data B0). Policy seed 42: completed, 4000 updates.

Eval: policy seed 42 and policy seed 1 each completed 100 / 100 verified seeds. Night stage: night_completed; status: completed.

# 2 Details V2 Result

Success: 31 / 100

Success Rate: 31.0%

95% Wilson CI: [22.8%, 40.6%]

Task: Insert HDMI. Official environment action limit: 600; unchanged for all nightly runs. Current source policy commit: 3cd75efd5b80aa14bc0ba13900c9b2e6e942bea8.

Policy seed 1: 23 / 100, success rate 23%, Wilson 95% CI [15.8%, 32.2%]. Across-policy mean: 27%; observed difference: 8 percentage points. These are two policy trainings with one fixed encoder, not 200 independent model seeds.

# 3 Matched B0 Result

MATCHED_BASELINE_NOT_AVAILABLE. Same-P0 original encoder exists, but no matched trained B0 policy was found; historical policies are reference only.

# 4 Policy Training Seeds

- Seed 42: training completed; checkpoint `v2_chain_seed42/policy_seed42/policy_last.ckpt`; Success: 31 / 100
- Seed 1: training completed; checkpoint expected `night_control/policy_seed1/policy_last.ckpt`; 100-seed result 23/100 (23.0%).

Mean success rate across the two policy seeds: 27.0%. Separate policy rates retained; not a pooled independent-training estimate.

# 5 Integrity Audit

- V2 attention parameters trained? YES — Q/K/V gradients and parameter updates verified.
- V2 checkpoint loaded correctly? YES — strict state load, critical coverage 100%, checkpoint hashes recorded.
- Policy uses V2 full embedding? YES — canonical encoder and tactile perturbation changes ACT actions.
- Output [B,512]? YES.
- Random-init fallback? NO — missing checkpoint/statistics rejected.
- Encoder frozen or fine-tuned? Fine-tuned trunk and attention; BN affine/running statistics frozen to match reference policy.

# 6 Jobs / Processes

Night sequence completed at 2026-09-10 06:52:08 Asia/Singapore. Controller 2906792, launcher 1586874 and simulator 1592219 have exited (verified after completion). No remaining jobs in this night sequence. Logs retained under `runs/night_control/`: `policy_seed1.log`, `seed1_eval_0_19.log`, `seed1_eval_20_99.log`, and each eval directory `results/`.

# 7 Failures

- Initial eval preparation: PermissionError writing a copied read-only release. Fixed in 35a8a12 by making only the isolated copy writable; external symlinks unchanged. No scientific configuration changed.
- Earlier recovery wrapper: unmatched parenthesis before simulator launch; corrected and logs retained.
- Prior launcher started a duplicate full 0–99 attempt after successful quick 0–19, before the latest no-repeat instruction. Duplicate attempt stopped; all its rows excluded. Quick results preserved; continuation runs only 20–99. SIGTERM did not terminate that simulator; SIGKILL was required for that duplicate process only. Continuation startup briefly overlapped process teardown before its first rollout; no completed quick rollout was interrupted.
- Documentation previously said 300 actions; actual HDMI source specifies 600. Documentation corrected; environment and criterion unchanged.

- Local SSH DNS resolution failed during monitoring from 04:54 to 06:33; recovered by 06:48. Remote evaluation continued, with no restart or scientific configuration change.

# 8 Git

Night controller source commit: `79b0913a033561a301c7fd097a4be52d22f380bc`. Public stage results and latest successful push are recorded in `results/git_sync.json`; if absent, final sync is pending. Do not infer push success from a local commit.

# 9 Evidence-backed Conclusion

INCONCLUSIVE — current results can establish pipeline operation and estimate this policy's success rate. A matched B0 policy comparison and independent policy seed are needed before claiming the architecture improves success.

# 10 Recommended Next Action

1. Train a strictly matched Original ResNet18 B0 policy using the existing same-data B0 encoder, then evaluate paired rollout seeds 0–99.
2. Compare matched B0 against both V2 policy seeds before deciding whether to invest in architecture changes.

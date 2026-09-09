# 1 Current Status

Updated: 2026-09-09T18:27:52.626466+00:00

Encoder: completed (V2 and same-data B0). Policy seed 42: completed, 4000 updates.

Eval: 100 / 100 verified seeds completed. Night stage: policy_seed1; status: running.

# 2 Details V2 Result

Success: 31 / 100

Success Rate: 31.0%

95% Wilson CI: [22.8%, 40.6%]

Task: Insert HDMI. Official environment action limit: 600; unchanged for all nightly runs. Current source policy commit: 3cd75efd5b80aa14bc0ba13900c9b2e6e942bea8.

# 3 Matched B0 Result

MATCHED_BASELINE_NOT_AVAILABLE. Same-P0 original encoder exists, but no matched trained B0 policy was found; historical policies are reference only.

# 4 Policy Training Seeds

- Seed 42: training completed; checkpoint `v2_chain_seed42/policy_seed42/policy_last.ckpt`; Success: 31 / 100
- Seed 1: training running (observed 2900/4000 updates at 2026-09-10 02:50 Asia/Singapore; started 02:27:52); checkpoint expected `night_control/policy_seed1/policy_last.ckpt`; 100-seed result not yet available.

# 5 Integrity Audit

- V2 attention parameters trained? YES — Q/K/V gradients and parameter updates verified.
- V2 checkpoint loaded correctly? YES — strict state load, critical coverage 100%, checkpoint hashes recorded.
- Policy uses V2 full embedding? YES — canonical encoder and tactile perturbation changes ACT actions.
- Output [B,512]? YES.
- Random-init fallback? NO — missing checkpoint/statistics rejected.
- Encoder frozen or fine-tuned? Fine-tuned trunk and attention; BN affine/running statistics frozen to match reference policy.

# 6 Jobs / Processes

Current stage: `policy_seed1`; controller PID `2906792`; child PID `737327`; GPU `2`. Log: `night_control/policy_seed1.log`. These are timestamped observations, not proof of perpetual liveness.

# 7 Failures

- Initial eval preparation: PermissionError writing a copied read-only release. Fixed in 35a8a12 by making only the isolated copy writable; external symlinks unchanged. No scientific configuration changed.
- Earlier recovery wrapper: unmatched parenthesis before simulator launch; corrected and logs retained.
- Prior launcher started a duplicate full 0–99 attempt after successful quick 0–19, before the latest no-repeat instruction. Duplicate attempt stopped; all its rows excluded. Quick results preserved; continuation runs only 20–99. SIGTERM did not terminate that simulator; SIGKILL was required for that duplicate process only. Continuation startup briefly overlapped process teardown before its first rollout; no completed quick rollout was interrupted.
- Documentation previously said 300 actions; actual HDMI source specifies 600. Documentation corrected; environment and criterion unchanged.

# 8 Git

Last confirmed GitHub push: `4041cac7d38c80d16be582c7739ca66a73bbb915`; current result snapshot sync follows.

Night controller source commit: `79b0913a033561a301c7fd097a4be52d22f380bc`. Public stage results and latest successful push are recorded in `results/git_sync.json`; if absent, final sync is pending. Do not infer push success from a local commit.

# 9 Evidence-backed Conclusion

INCONCLUSIVE — current results can establish pipeline operation and estimate this policy's success rate. A matched B0 policy comparison and independent policy seed are needed before claiming the architecture improves success.

# 10 Recommended Next Action

1. Complete and verify disjoint seed 0–99 results for each authorized policy seed.
2. Obtain a strictly matched B0 policy before drawing an architecture improvement conclusion.
3. Review paired results and policy-seed variability; keep negative outcomes without tuning this experiment.

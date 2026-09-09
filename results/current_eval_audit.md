# Active evaluation audit

Snapshot: 2026-09-09, Asia/Singapore. Root artifact directory on mlda2: `details_v2_20260909`.

- Accepted quick run: `runs/v2_chain_seed42/eval_quick`, seeds 0–19, completed 5/20, no infrastructure errors.
- Active continuation: `runs/v2_chain_seed42/eval_20_99`, seeds 20–99; launcher PID 2353201, simulator PID 2358130, GPU 2. Seeds 20 and 21 have completed real 600-action timeout rollouts at this snapshot.
- Policy: `runs/v2_chain_seed42/policy_seed42/policy_last.ckpt`, SHA256 `3047f071db09e307d9bbbf942b70ef1b49977c1c0935a7a76a676fec3c720d52`.
- Encoder: `runs/encoder_v2_seed42/best.pt`, type `detail_v2`, SHA256 `914d9d8315a8ccd7fd88ea853b20375a90010ac45968e88e9d2eb386abffe448`.
- Policy source: immutable `3cd75efd5b80aa14bc0ba13900c9b2e6e942bea8`; preparation fix `35a8a12`. Active code/config/checkpoints are not modified.
- Config: `runs/v2_chain_seed42/policy_seed42/train_config.yml`, SHA256 `42b99d3c1cb94d08e75f13b4204f87922866992439a5411f5a829bef1c4a6fb4`.
- Eval source tree SHA256: `12c8da06f3669a9589d526a7503dabb4d94c9f7ce38e8ab24cbdfef2107f37ac`.
- Task: `insert_HDMI`, `demo`, `ACT/deploy`. Command: `run_eval_compat.py scripts/eval_policy.py insert_HDMI demo ACT/deploy --headless --device cuda:0 --total_num 80 --start_seed 20` (GPU 2 exposed as cuda:0). Quick command differs only in seed range: total_num 20, start_seed 0.
- Success predicate: prism relative to target has absolute x/y < 0.005, z < 0.005, end-effector z > 0.145, and prism local z axis dot world z > 0.965. Uses environment `eval_success`; official action limit 600, with existing early-stop behavior.
- Logs: each run's `eval.log`, `results/log.log`, `results/outcomes.jsonl`, `results/completion.json`, and `manifest.json`; per-seed CSV references the detailed log.
- Excluded duplicate attempt: `eval_full`; never combined with accepted quick/tail. Original quick was already finished and was not interrupted. See Morning Report for duplicate-process cleanup disclosure.

Night controller waits for the active continuation's completion and process exit, validates disjoint seeds and equal policy/encoder/config/source/environment hashes, then aggregates. It refuses missing/duplicate/invalid outcomes. No parameter or protocol changes are based on success rate. A 0/100 result stops replica spending for diagnosis; other engineering failures stop the sequence for the authorized heartbeat to inspect.

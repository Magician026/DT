# detail_v2_eval_0_99

Success: 31 / 100

Success rate: 31.0%; Wilson 95% CI: [22.8%, 40.6%].

- policy_checkpoint_sha256: `"3047f071db09e307d9bbbf942b70ef1b49977c1c0935a7a76a676fec3c720d52"`
- encoder_sha256: `"914d9d8315a8ccd7fd88ea853b20375a90010ac45968e88e9d2eb386abffe448"`
- train_config_sha256: `"42b99d3c1cb94d08e75f13b4204f87922866992439a5411f5a829bef1c4a6fb4"`
- source_tree_sha256: `"12c8da06f3669a9589d526a7503dabb4d94c9f7ce38e8ab24cbdfef2107f37ac"`
- task: `"insert_HDMI"`
- policy_source_commit: `"3cd75efd5b80aa14bc0ba13900c9b2e6e942bea8"`
- success_count: `31`
- total_count: `100`
- success_rate: `0.31`
- wilson95: `[0.22779697212376038, 0.40626055719489407]`
- successful_seeds: `[0, 3, 5, 7, 19, 24, 25, 26, 30, 31, 32, 33, 36, 42, 43, 44, 46, 48, 49, 52, 54, 59, 65, 67, 71, 72, 80, 81, 85, 94, 99]`
- failed_seeds: `[1, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 27, 28, 29, 34, 35, 37, 38, 39, 40, 41, 45, 47, 50, 51, 53, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 68, 69, 70, 73, 74, 75, 76, 77, 78, 79, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98]`
- abnormal_seeds: `[]`
- infrastructure_error_count: `0`
- episode_action_limit: `600`
- evaluation_commands: `["insert_HDMI demo ACT/deploy --total_num 20 --start_seed 0", "insert_HDMI demo ACT/deploy --total_num 80 --start_seed 20"]`
- timestamp: `"2026-09-09T18:27:52.507156+00:00"`

This describes one trained policy. It does not establish an encoder architecture improvement.

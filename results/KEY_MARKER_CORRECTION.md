# Key marker validation correction and controlled rerun

Previous depth-only Key result36/100 is NON_COMPARABLE to the intended depth+marker recipe. It remains historical evidence, not the result of this corrected experiment.

Root cause: the audit incorrectly required `marker[:,0]` to remain constant by array slot over an entire trajectory. Actual ManiSkill flow generation applies current-point visibility and tracking masks to `stack([init,current])`, selects both planes with the same indices, and pads both by repeating the last pair. Point slots can change between frames while within-frame displacement stays valid.

Source chain (server UniVTAC source):
- `envs/sensors/tactile.py` selects `ManiSkillSimulatorCfg`, exposes `marker_motion` unchanged.
- `fem_based/sim/__init__.py` imports `VisionTactileSensorUIPC` from `tactile_sensor_sapienipc_modified.py`.
- Its `gen_marker_flow`, lines317–365, stacks initial/current before shared filtering and paired padding.
- `fem_based/mani_skill_sim.py`, marker_motion_simulation, directly returns that array.
- `envs/utils/data.py`, dict_to_hdf5 lines215–232, stores non-RGB arrays without permutation.

Real `clean/10.hdf5` right sensor frames131/132/133 contain63/61/58 distinct prefix pairs, followed by identical last-pair padding in both planes. Displacements are finite. This matches changing visibility, not invalid reference semantics. No rewriting, reordering, masking or reweighting of training labels is introduced.

Correction changes only the audit invariant: validate frame-local paired shape/finite values and record cross-frame reference-slot changes. Loader still uses exactly `marker[1]-marker[0]`, all1200 entries, train-only mean/std and equally weighted normalized MSE with depth. Same V2 architecture, five epochs, batch32, LR1e-4, encoder seed821018648; same ACT4000updates/physical32×2/policy seed0; formal100 seeds1000000–1000099. Separate `key_depth_marker_seed0` directory under the canonical server root. No depth-only checkpoints are reused.

Validation: synthetic visibility-drop/repeat-padding regression and invalid marker tests2passed; full data audit must finish before encoder launch. No outcome-dependent hyperparameter selection.

## Corrected run: encoder completed

Verified 2026-09-10 06:41 UTC: formal depth+marker encoder completed 5 epochs / 5700 updates, best validation loss 0.03970248. Encoder SHA256: `87f66f8e7ca0825c8db1cf9ee211f311e6f01093b2f901afb437dc4d24b44fe9`. Source: `4460420f6d475c9570d1d103ef7157ffd960b27a`.

Policy smoke and 30-action simulation smoke passed (30 distinct tactile observations). Formal policy seed 0 is running on GPU 2, PID 2632718, observed 2700/4000 updates; no formal evaluation result yet. Controller PID 1747332 will launch 100 rollouts after verified policy completion.

Server root: `/usr1/home/s126mdg41_04/UniVTAC details v2/key_depth_marker_seed0`; policy log: `runs/pull_out_key/policy_seed0.log`.

## Corrected run: policy completed

Verified 2026-09-10 07:12 UTC: policy seed0 completed 4000 optimizer updates / 8000 micro iterations. Policy SHA256 `c25eddbb1f88e8ff3eae02d09be40457d0263366558ab87795c6fbebe64cd8dc`; encoder hash unchanged. Formal evaluation started 06:53 UTC, launcher PID551341, GPU2, exact seeds1000000–1000099. Evaluation is in progress; final success rate pending. Evidence: `key_depth_marker_policy_completion.json`.

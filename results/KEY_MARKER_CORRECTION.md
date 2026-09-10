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

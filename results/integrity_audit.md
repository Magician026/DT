# Details V2 integrity audit

Audit date: 2026-09-09 (Asia/Singapore)

Scope: read-only inspection of the completed encoder/policy artifacts and the
immutable V2 release. The active evaluation was not interrupted, restarted or
modified.

## Evidence identity

| Item | Evidence |
|---|---|
| Encoder completion | `encoder_v2_seed42/completion.json`: `status=completed`, `encoder_type=detail_v2`, 3,320 updates |
| Encoder checkpoint | `encoder_v2_seed42/best.pt`, SHA256 `914d9d8315a8ccd7fd88ea853b20375a90010ac45968e88e9d2eb386abffe448` |
| Policy completion | `policy_seed42/completion.json`: `status=completed`, 4,000 optimizer updates / 8,000 micro-iterations |
| Policy checkpoint | `policy_seed42/policy_last.ckpt`, SHA256 `3047f071db09e307d9bbbf942b70ef1b49977c1c0935a7a76a676fec3c720d52` |
| Policy config | `policy_seed42/resolved_config.json`: `tactile_encoder_type=detail_v2`, `tactile_ckpt` matches the encoder SHA, task `sim-insert_HDMI-demo-50`, seed 42 |
| Immutable source | release commit `3cd75efd5b80aa14bc0ba13900c9b2e6e942bea8`; evaluation preparation commit `35a8a12` |
| Active job observation | During audit, the continuation evaluation process was running; no process or output directory was changed |

## Required integrity answers

| Check | Result | Evidence |
|---|---|---|
| Details V2 attention parameters trained? | **YES** | Encoder `gradient_step_1..3.json` report nonzero gradients and nonzero updates for `global_query`, `cross_attention.in_proj_weight`, `cross_attention.out_proj.weight`, and `detail_projection`; policy `interface_smoke.json` also reports nonzero gradients/updates for every V2 projection and attention group |
| Semantic/local projection trained? | **YES** | The same encoder gradient artifacts report nonzero gradients and updates for `semantic_projection.0.*` and `local_projection.0.*`; policy smoke reports both groups in the managed tactile optimizer |
| V2 checkpoint loaded correctly? | **YES** | Encoder checkpoint has exactly the public fields `encoder_type`, `structure`, `preprocess`, and `encoder_state`; the canonical loader validates metadata and calls `load_state_dict(..., strict=True)`; policy interface smoke has `critical_weight_coverage=1.0` and the completed policy artifact is tied to the recorded encoder SHA |
| Random-init fallback? | **NO** | The V2 tactile loader requires an existing checkpoint, validates `expected_encoder_type=detail_v2` and preprocessing, and raises on missing/incompatible files. Evaluation preparation also rejects a config without `detail_v2`; no fallback branch is used in the recorded run |
| Policy actually uses the full V2 embedding? | **YES** | `DetailV2Encoder.forward()` concatenates the 256-D semantic and 256-D detail vectors; policy smoke returns actions with nonzero tactile perturbation (`0.0648438931` max absolute) and the policy checkpoint contains the V2 branch keys under `model.backbones.1.backbone.*` |
| Output shape `[B,512]`? | **YES** | V2 source contract requires `latent_dims=512`; the implementation concatenates two 256-D vectors; the existing full M2 test suite passed 42 tests and the policy ACT interface smoke passed with action shape `[32,50,8]` |
| Q/K/V branch present in policy checkpoint? | **YES** | Read-only checkpoint inspection found `model.backbones.1.backbone.cross_attention.*` keys, together with `local_projection`, `semantic_projection`, `global_query`, and `detail_projection` keys; the checkpoint contained 483 total state keys and 139 tactile-backbone keys |
| Encoder frozen or fine-tuned in policy? | **Fine-tuned, with BN frozen** | Policy smoke reports `bn_statistics=frozen` and `bn_affine=frozen`; optimizer groups include the tactile parameters; nonzero tactile parameter updates are recorded. Thus ordinary trunk/projection/attention weights are fine-tuned while BN affine parameters and running statistics remain fixed |
| ACT architecture/action head changed? | **NO** | V2 config retains the existing ACT dimensions (`hidden_dim=512`, encoder 4 layers, decoder 7 layers, chunk 50, action dimension 8); the V2 change is confined to the tactile encoder path |
| Evaluation protocol changed by the audit? | **NO** | No remote files, config, checkpoint, process, seed manifest, or active evaluation directory was modified |

## Architecture contract checked

The immutable V2 implementation matches the declared structure:

1. Shared ResNet18 trunk.
2. Layer4 global average pool → semantic projection to 256-D and global query.
3. Layer2 full spatial grid → local projection and positional encoding.
4. Four-head cross-scale attention with the global query over local keys/values.
5. Detail projection to 256-D.
6. Concatenation of semantic and detail vectors to `[B,512]`.

The encoder completion metadata records the expected `detail_v2` structure and
the policy resolved config points to the same encoder checkpoint hash. The
policy checkpoint contains the branch parameters rather than only the
ResNet layer4 trunk, so the detail path was not bypassed.

## Limitations and non-claims

This audit establishes implementation and artifact integrity. It does not
establish a success rate, an improvement over B0, or a causal contribution of
the attention branch. The B0 encoder artifact is valid as a same-data encoder
control, but no matching B0 downstream policy was found. Evaluation results
must remain pending until the evaluator publishes complete per-seed results.


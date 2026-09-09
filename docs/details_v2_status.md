# Details V2 stage evidence

## M1 engineering checks (2026-09-09)

- DT pre-change HEAD: `5f73589`; remote default branch confirmed `main`.
- Separate source/run directory; prior server HEAD and worktree diff archived.
- Model: 16 CPU tests pass; original/V2 batch 1 and 4 at 256×256 finite.
- Loader and active-only P0 tests pass (tests first failed for missing modules).
- Clean split hash: `437a7bad6610b5f84888149ab6416b5f61ae1e28a5db6d2bda1e0c17aafd0e63`.
- Data: 100 complete trajectory files, 21,240/2,360 train/val sensor frames;
  separate ACT demo-50, 5,850 action frames, qpos/action dimensions 8.
- Native decoded clean0 and ACT0 tactile image exactly equal.
- Real overfit: 50 optimizer steps, normalized P0 mean first5=1.8490346670,
  last5=0.9603957891. Checkpoint strict reload equality passed.
- Three distinct real batch32 updates: finite losses/gradients, semantic/local
  projection/query/attention/trunk updated. Peak allocation ~1.45 GB.
- One initial probe invocation omitted the probe flag; it was stopped explicitly
  before formal training and retained as an aborted diagnostic, not a result.
- Encoder formal training: not yet launched at the M1 commit boundary.
- Policy/eval: integration in progress, no success rate produced.

References: encoder server source `a9b8d3998c3106137772fa7e22741dcef9e37f76`;
policy reference `dbb5fdc4fc270debedfaa92050ac9d505dde58a1`;
local external simulator source `05bcd3e`. No reference server was modified.
Historical baselines are reference, not comparable to P0.

## Encoder result

The formal encoder run completed from immutable source `6c79d02`: five full
epochs, 3,320 optimizer updates, best full-validation P0 loss
`0.025028514015977665` (epoch 3, zero-based). Encoder SHA256:
`914d9d8315a8ccd7fd88ea853b20375a90010ac45968e88e9d2eb386abffe448`.
See `results/encoder_v2_seed42.json` and `results/encoder_v2_curve.jsonl`.

ACT real batch smoke passed with this formal encoder. Earlier batch32 ×
accumulation2 smoke used four micro-batches for exactly two updates; peak
allocated memory 5.65 GB. Tactile perturbation changes actual ACT action output;
all trainable attention parameters belong to optimizer groups. The consolidated
encoder/loader/P0/policy tests passed 27/27 on the training server.

The same-data original B0 encoder has now started with the explicitly exported
V2 trunk initialization, identical split/P0/batch/LR/five-epoch budget. This is not
yet a downstream comparable result: policy training and simulation results are
still required. No manipulation success rate or improvement claim is available.

## M2 integration

Simulator smoke passed with the formal V2 encoder origin: reset once at seed 0,
three actual actions, three tactile observations with three unique hashes, two
consecutive changes. ACT policy loading reported all keys matched. This is a
smoke result, not a manipulation success rate.

B0 encoder also completed 3,320 updates; best val P0 loss 0.0262942573. Both
optimizer states independently contain exactly step 3320. The explicit common
trunk initialization SHA is
`84021dab18740505166250626b0df16527e3df6be76540aeac74412dc8367508`.

Policy/eval runner code is ready for an immutable M2 release. Formal policy is
not yet launched at this commit boundary; run status will be recorded after
checking the actual process and first training log. Simulator environment source
files remain unchanged from the working external runtime. Only evaluation seed
handling, telemetry and failure accounting are wrapped.

## Formal chain started (2026-09-09 21:58 +08:00)

- Immutable policy/eval source: `3cd75efd5b80aa14bc0ba13900c9b2e6e942bea8`.
- Full M2 test suite: `42 passed in 9.06s` on the training server.
- Chain PID 4044695; policy PID 4045080; GPU 2.
- Actual observed policy progress: 175/4000 optimizer updates, 350 micro-batches,
  physical/effective batch 32/64, LR groups all 1e-5, loss 3.0397.
- Policy status: running. Quick/full eval: waiting for verified policy completion.
- No success rate exists yet. The thread follows meaningful stage changes and
  will sync actual results. This record is a timestamped observation, not a claim
  that the processes remain alive indefinitely.
- No current blocking issue. The GPU phase lock and run lock prevent duplicate
  launches of this chain; the fixed release is read-only.

## Policy completed; eval preparation repair

Policy completed exactly 4,000 optimizer updates / 8,000 micro-batches.
Checkpoint SHA256: `3047f071db09e307d9bbbf942b70ef1b49977c1c0935a7a76a676fec3c720d52`.
The initial quick-eval preparation failed before simulator launch: copying the
immutable release preserved read-only directory permissions. Preparation now
makes only the isolated runtime copy writable, without following asset symlinks.
Eight eval tests pass, including the read-only-copy/symlink regression test.
No algorithm, checkpoint, preprocessing, task definition or seeds changed.

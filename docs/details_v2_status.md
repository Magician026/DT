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

"""Manifest-driven UniVTAC evaluation loop with per-step tactile telemetry.

The reset/error handling is derived from UniVTAC's evaluation loop and the
project's ``two_axis_eval_loop.py``. This version consumes an immutable seed
list and never substitutes a different seed after an infrastructure failure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import torch


def load_seed_list(path: str | Path) -> list[int]:
    """Load an exact JSON list of unique integer seeds, preserving its order."""
    value = json.loads(Path(path).read_text())
    if not isinstance(value, list) or not value:
        raise ValueError("seed manifest must be a non-empty JSON list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
        raise ValueError("every seed must be an integer")
    if len(value) != len(set(value)):
        raise ValueError("seed list contains duplicates")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def _hash_value(hasher: Any, value: Any) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            hasher.update(str(key).encode("utf-8"))
            _hash_value(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _hash_value(hasher, item)
        return
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(str(tuple(tensor.shape)).encode("ascii"))
        hasher.update(tensor.numpy().tobytes())
        return
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(str(tuple(value.shape)).encode("ascii"))
        hasher.update(value.tobytes())
        return
    hasher.update(repr(value).encode("utf-8"))


def tactile_sha256(observation: Mapping[str, Any]) -> str:
    tactile = observation.get("tactile")
    if not isinstance(tactile, Mapping) or not tactile:
        raise RuntimeError("observation has no tactile payload")
    hasher = hashlib.sha256()
    _hash_value(hasher, tactile)
    return hasher.hexdigest()


def _reset_same_seed(task: Any, seed: int, instructions: Any, output: Path, log: Callable) -> int:
    for attempt in range(1, 4):
        log(f"EPISODE_RESET seed={seed} attempt={attempt}")
        try:
            task.reset(seed=seed, instructions=instructions)
        except Exception as error:
            _append_jsonl(
                output / "infrastructure_failures.jsonl",
                {
                    "seed": seed,
                    "phase": "reset",
                    "attempt": attempt,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
            task.clean_cache(result="error")
        else:
            return attempt
    raise RuntimeError(f"reset failed three times for seed {seed}")


def _step_with_telemetry(
    task: Any,
    policy: Any,
    *,
    seed: int,
    step_index: int,
    previous_tactile_hash: str | None,
    output: Path,
) -> tuple[str, bool]:
    observation = task._get_observations()
    tactile_hash = tactile_sha256(observation)
    actions_before = int(task.take_action_cnt)
    policy.eval(task, observation)
    actions_after = int(task.take_action_cnt)
    if actions_after <= actions_before:
        raise RuntimeError("policy evaluation did not advance the action counter")
    changed = previous_tactile_hash is not None and tactile_hash != previous_tactile_hash
    _append_jsonl(
        output / "step_telemetry.jsonl",
        {
            "seed": seed,
            "policy_step": step_index,
            "actions_before": actions_before,
            "actions_after": actions_after,
            "tactile_sha256": tactile_hash,
            "changed_from_previous": changed,
        },
    )
    return tactile_hash, changed


def _run_smoke(
    task: Any,
    policy: Any,
    seeds: Sequence[int],
    instructions: Any,
    log: Callable,
    output: Path,
    smoke_steps: int,
    completion_context: Mapping[str, Any],
) -> dict[str, Any]:
    seed = seeds[0]
    result: dict[str, Any]
    try:
        reset_attempts = _reset_same_seed(task, seed, instructions, output, log)
        policy.reset()
        start_actions = int(task.take_action_cnt)
        hashes: list[str] = []
        changes = 0
        previous = None
        for step_index in range(smoke_steps):
            tactile_hash, changed = _step_with_telemetry(
                task,
                policy,
                seed=seed,
                step_index=step_index,
                previous_tactile_hash=previous,
                output=output,
            )
            hashes.append(tactile_hash)
            changes += int(changed)
            previous = tactile_hash
        action_count = int(task.take_action_cnt) - start_actions
        if len(hashes) != smoke_steps or action_count < smoke_steps or changes < 1:
            raise RuntimeError("smoke did not prove tactile and action updates")
        task.clean_cache(result="smoke_pass")
        result = {
            "status": "smoke_pass",
            "seed": seed,
            "requested_steps": smoke_steps,
            "action_count": action_count,
            "tactile_observation_count": len(hashes),
            "tactile_change_count": changes,
            "unique_tactile_hashes": len(set(hashes)),
            "reset_attempts": reset_attempts,
            "test_num": 0,
            "succ_num": 0,
        }
    except Exception as error:
        task.clean_cache(result="smoke_failed")
        _append_jsonl(
            output / "infrastructure_failures.jsonl",
            {
                "seed": seed,
                "phase": "smoke",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        result = {
            "status": "smoke_failed",
            "seed": seed,
            "requested_steps": smoke_steps,
            "error": repr(error),
            "test_num": 0,
            "succ_num": 0,
        }
    result = {**completion_context, **result}
    _atomic_json(output / "completion.json", result)
    return result


def run_eval_loop(
    task: Any,
    policy: Any,
    seeds: Sequence[int],
    instructions: Mapping[str, Any],
    instruction_type: str,
    log: Callable[[str], None],
    *,
    smoke_steps: int | None = None,
    completion_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate exactly the requested seeds or run a bounded simulator smoke."""
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be a non-empty integer sequence")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed sequence contains duplicates")
    if instruction_type not in instructions:
        raise KeyError(f"instruction type {instruction_type!r} is unavailable")
    if smoke_steps is not None and smoke_steps <= 0:
        raise ValueError("smoke_steps must be positive")

    output = Path(task.save_root)
    output.mkdir(parents=True, exist_ok=True)
    if completion_context is None:
        completion_context = {}
    elif not isinstance(completion_context, Mapping):
        raise TypeError("completion_context must be a mapping")
    selected_instructions = instructions[instruction_type]
    if smoke_steps is not None:
        return _run_smoke(
            task,
            policy,
            seeds,
            selected_instructions,
            log,
            output,
            smoke_steps,
            completion_context,
        )

    rows: list[dict[str, Any]] = []
    valid_episode_count = 0
    success_count = 0
    infrastructure_error_count = 0
    for seed in seeds:
        row: dict[str, Any] = {"seed": seed}
        try:
            reset_attempts = _reset_same_seed(
                task, seed, selected_instructions, output, log
            )
        except Exception as error:
            infrastructure_error_count += 1
            row.update(
                success=None,
                termination="reset_infrastructure_error",
                reset_attempts=3,
                actions=0,
                tactile_observations=0,
                tactile_changes=0,
                error=repr(error),
            )
            _append_jsonl(output / "outcomes.jsonl", row)
            rows.append(row)
            continue

        task.mode = "eval"
        task.mean_steps = task.cfg.step_lim
        policy.reset()
        hashes: list[str] = []
        changes = 0
        previous = None
        success = False
        termination = "timeout"
        try:
            while task.take_action_cnt < task.cfg.step_lim:
                tactile_hash, changed = _step_with_telemetry(
                    task,
                    policy,
                    seed=seed,
                    step_index=len(hashes),
                    previous_tactile_hash=previous,
                    output=output,
                )
                hashes.append(tactile_hash)
                changes += int(changed)
                previous = tactile_hash
                if task.eval_success:
                    success = True
                    termination = "success"
                    break
                if task.check_early_stop():
                    termination = "early_stop"
                    break
        except Exception as error:
            infrastructure_error_count += 1
            termination = "policy_or_simulator_error"
            _append_jsonl(
                output / "infrastructure_failures.jsonl",
                {
                    "seed": seed,
                    "phase": "episode",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
            task.clean_cache(result="error")
            row.update(
                success=None,
                termination=termination,
                reset_attempts=reset_attempts,
                actions=int(task.take_action_cnt),
                tactile_observations=len(hashes),
                tactile_changes=changes,
                error=repr(error),
            )
        else:
            valid_episode_count += 1
            success_count += int(success)
            task.clean_cache(result="success" if success else "failed")
            row.update(
                success=int(success),
                termination=termination,
                reset_attempts=reset_attempts,
                actions=int(task.take_action_cnt),
                tactile_observations=len(hashes),
                tactile_changes=changes,
                unique_tactile_hashes=len(set(hashes)),
            )
            rollouts = output / "rollouts.csv"
            with rollouts.open("a", newline="") as stream:
                writer = csv.writer(stream)
                if valid_episode_count == 1:
                    writer.writerow(["episode_id", "seed_or_initial_state", "success"])
                writer.writerow([valid_episode_count - 1, seed, int(success)])
        _append_jsonl(output / "outcomes.jsonl", row)
        rows.append(row)
        log(
            f"SEED_FINISHED seed={seed} termination={termination} "
            f"valid={row['success'] is not None}"
        )

    completed = valid_episode_count == len(seeds)
    result = {
        **completion_context,
        "status": "completed" if completed else "incomplete",
        "requested_seeds": list(seeds),
        "requested_seed_count": len(seeds),
        "valid_episode_count": valid_episode_count,
        "success_count": success_count,
        "infrastructure_error_count": infrastructure_error_count,
        "test_num": valid_episode_count,
        "succ_num": success_count,
    }
    if completed:
        result["success_rate"] = success_count / valid_episode_count
    _atomic_json(output / "completion.json", result)
    return result


__all__ = ["load_seed_list", "run_eval_loop", "tactile_sha256"]

#!/usr/bin/env python3
"""Prepare an isolated Details V2 evaluation runtime from official assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import torch
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.eval_loop_v2 import load_seed_list
except ImportError:  # Direct execution from the scripts directory.
    from eval_loop_v2 import load_seed_list


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parse_task_name(value: str) -> tuple[str, str, int]:
    if not isinstance(value, str) or not value.startswith("sim-"):
        raise ValueError("train_config task_name must use sim-<task>-<config>-<episodes>")
    parts = value.removeprefix("sim-").rsplit("-", 2)
    if len(parts) != 3 or not parts[0] or not parts[1]:
        raise ValueError("train_config task_name must use sim-<task>-<config>-<episodes>")
    try:
        episodes = int(parts[2])
    except ValueError as error:
        raise ValueError("task_name episode count must be an integer") from error
    if episodes <= 0:
        raise ValueError("task_name episode count must be positive")
    return parts[0], parts[1], episodes


def _validate_policy_run(policy_run: Path, *, allow_smoke: bool, smoke_steps: int | None):
    required = [
        policy_run / "completion.json",
        policy_run / "train_config.yml",
        policy_run / "policy_last.ckpt",
        policy_run / "dataset_stats.pkl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"policy run is incomplete; missing={missing}")

    completion = dict(_read_json(policy_run / "completion.json"))
    if completion.get("status") != "completed":
        raise ValueError("policy completion status must be completed")
    is_smoke = completion.get("smoke") is True
    if is_smoke and not allow_smoke:
        raise ValueError("smoke policy requires explicit --allow-smoke")
    if is_smoke and smoke_steps is None:
        raise ValueError("smoke policy requires --smoke-steps")
    if completion.get("checkpoint") != "policy_last.ckpt":
        raise ValueError("policy completion must select policy_last.ckpt")

    checkpoint = policy_run / "policy_last.ckpt"
    stats = policy_run / "dataset_stats.pkl"
    if _sha256(checkpoint) != completion.get("checkpoint_sha256"):
        raise ValueError("policy checkpoint SHA256 does not match completion.json")
    if _sha256(stats) != completion.get("dataset_stats_sha256"):
        raise ValueError("dataset stats SHA256 does not match completion.json")

    config = yaml.safe_load((policy_run / "train_config.yml").read_text())
    if not isinstance(config, Mapping):
        raise ValueError("train_config.yml must contain a mapping")
    encoder_type = config.get("tactile_encoder_type")
    if encoder_type not in ("detail_v2", "detail_v2_dynamic"):
        raise ValueError(
            "train_config must use tactile_encoder_type=detail_v2 or detail_v2_dynamic"
        )
    if config.get("tactile_type") != "feat":
        raise ValueError("train_config must use tactile_type=feat")
    tactile_checkpoint = Path(str(config.get("tactile_ckpt", "")))
    if not tactile_checkpoint.is_absolute() or not tactile_checkpoint.is_file():
        raise ValueError("train_config tactile_ckpt must be an existing absolute path")
    if _sha256(tactile_checkpoint) != completion.get("encoder_sha256"):
        raise ValueError("encoder SHA256 does not match policy completion")
    if encoder_type == "detail_v2_dynamic":
        temporal_metadata = {
            "tactile_sequence_length": config.get("tactile_sequence_length"),
            "tactile_temporal_stride": config.get("tactile_temporal_stride"),
        }
        temporal_stride = temporal_metadata["tactile_temporal_stride"]
        if (
            temporal_metadata["tactile_sequence_length"] != 4
            or type(temporal_stride) is not int
            or temporal_stride not in (1, 2)
        ):
            raise ValueError(
                "dynamic tactile temporal metadata must be length=4 and stride=1 or 2"
            )
        if any(completion.get(key) != value for key, value in temporal_metadata.items()):
            raise ValueError("dynamic tactile temporal metadata differs from completion")
        if completion.get("tactile_encoder_type") != encoder_type:
            raise ValueError("dynamic tactile encoder type differs from completion")
        checkpoint_metadata = torch.load(
            tactile_checkpoint, map_location="cpu", weights_only=True
        )
        checkpoint_structure = (
            checkpoint_metadata.get("structure", {})
            if isinstance(checkpoint_metadata, Mapping)
            else {}
        )
        if (
            checkpoint_metadata.get("encoder_type") != encoder_type
            or checkpoint_structure.get("sequence_length")
            != temporal_metadata["tactile_sequence_length"]
            or checkpoint_structure.get("temporal_stride")
            != temporal_metadata["tactile_temporal_stride"]
        ):
            raise ValueError("dynamic tactile temporal metadata differs from encoder checkpoint")

    optimizer_updates = completion.get("optimizer_updates")
    if isinstance(optimizer_updates, bool) or not isinstance(optimizer_updates, int):
        raise ValueError("policy optimizer_updates must be an integer")
    if is_smoke:
        if optimizer_updates <= 0:
            raise ValueError("smoke policy has no optimizer updates")
    elif optimizer_updates != config.get("num_steps"):
        raise ValueError("formal policy did not complete its configured optimizer updates")
    if not completion.get("source_commit"):
        raise ValueError("policy completion is missing source_commit")
    return completion, dict(config), is_smoke


def _make_copy_writable(root: Path) -> None:
    """Only change copied files; never follow external asset symlinks."""
    for directory, dirs, files in os.walk(root, followlinks=False):
        path = Path(directory)
        path.chmod(path.stat().st_mode | 0o700)
        for name in files:
            child = path / name
            if not child.is_symlink():
                child.chmod(child.stat().st_mode | 0o600)


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {".git", "__pycache__", "act_ckpt", "eval_result", "data"}
    return {
        name
        for name in names
        if name in ignored or name.startswith("._") or name.endswith((".pyc", ".png"))
    }


def _tree_python_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*.py"))
        if path.is_file()
        and not path.name.startswith("._")
        and "__pycache__" not in path.parts
    }


def _source_tree_sha256(source: Path) -> str:
    hasher = hashlib.sha256()
    for relative_root in (Path("encoder"), Path("policy/ACT")):
        root = source / relative_root
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if "__pycache__" in path.parts or path.name.startswith("._"):
                continue
            relative = relative_root / path.relative_to(root)
            hasher.update(str(relative).encode("utf-8"))
            hasher.update(bytes.fromhex(_sha256(path)))
    return hasher.hexdigest()


def _replace_once(text: str, old: str, new: str, description: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"external eval script {description} anchor count is {text.count(old)}")
    return text.replace(old, new, 1)


def _patch_eval_script(path: Path) -> None:
    original = path.read_text()
    start = original.find("def eval_policy(")
    end = original.find("\ndef get_config(", start)
    if start < 0 or end < 0:
        raise ValueError("external eval script has no recognized eval_policy function")
    replacement = '''def eval_policy(task, policy, expert_check, start_seed, max_seed, test_total_num, instructions, instruciton_type='seen'):
    assert not expert_check
    import os
    from eval_loop_v2 import load_seed_list, run_eval_loop
    seeds = load_seed_list(os.environ['V2_EVAL_SEEDS'])
    completion_context = json.loads(Path(os.environ['V2_EVAL_CONTEXT']).read_text())
    smoke_value = os.environ.get('V2_SMOKE_STEPS')
    smoke_steps = int(smoke_value) if smoke_value else None
    return run_eval_loop(
        task, policy, seeds, instructions, instruciton_type, log,
        smoke_steps=smoke_steps, completion_context=completion_context,
    )
'''
    edited = original[:start] + replacement + original[end:]

    save_lines = [line for line in edited.splitlines() if "env_cfg.save_dir =" in line]
    if len(save_lines) != 1:
        raise ValueError("external eval script save directory anchor is ambiguous")
    indentation = save_lines[0][: len(save_lines[0]) - len(save_lines[0].lstrip())]
    edited = _replace_once(
        edited,
        save_lines[0],
        indentation + "env_cfg.save_dir = Path(os.environ['V2_EVAL_OUTPUT'])",
        "save directory",
    )

    final_lines = [line for line in edited.splitlines() if "Final Result:" in line]
    if len(final_lines) != 1:
        raise ValueError("external eval script final result anchor is ambiguous")
    indentation = final_lines[0][: len(final_lines[0]) - len(final_lines[0].lstrip())]
    final_replacement = (
        indentation + "if results['status'] == 'smoke_pass':\n"
        + indentation + "    log(f\"Smoke Result: {results['status']}\")\n"
        + indentation + "elif results['status'] == 'completed':\n"
        + indentation
        + "    log(f\"Final Result: {results['success_count']}/{results['valid_episode_count']}\")\n"
        + indentation + "else:\n"
        + indentation + "    log(f\"Evaluation incomplete: {results}\")"
    )
    edited = _replace_once(edited, final_lines[0], final_replacement, "final result")
    close_anchor = "    simulation_app.close()"
    edited = _replace_once(
        edited,
        close_anchor,
        close_anchor
        + "\n    if results['status'] not in ('completed', 'smoke_pass'):\n"
        + "        raise RuntimeError(f\"Evaluation did not complete: {results}\")",
        "simulator close",
    )
    path.write_text(edited)


def prepare_runtime(
    *,
    external_runtime: str | Path,
    source: str | Path,
    policy_run: str | Path,
    out: str | Path,
    gpu: str,
    seeds_path: str | Path,
    smoke_steps: int | None = None,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    """Validate artifacts and atomically prepare an isolated eval directory."""
    external_runtime = Path(external_runtime).resolve()
    source = Path(source).resolve()
    policy_run = Path(policy_run).resolve()
    out = Path(out).resolve()
    seeds_path = Path(seeds_path).resolve()
    if out.exists():
        raise FileExistsError(f"evaluation output already exists: {out}")
    if not gpu:
        raise ValueError("gpu must be non-empty")
    if smoke_steps is not None and smoke_steps <= 0:
        raise ValueError("smoke_steps must be positive")
    for required in (
        external_runtime / "scripts" / "eval_policy.py",
        external_runtime / "run_eval_compat.py",
        source / "encoder",
        source / "policy" / "ACT",
        Path(__file__).with_name("eval_loop_v2.py"),
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    seeds = load_seed_list(seeds_path)
    completion, config, is_smoke_policy = _validate_policy_run(
        policy_run, allow_smoke=allow_smoke, smoke_steps=smoke_steps
    )
    task, task_config, episode_count = _parse_task_name(config["task_name"])
    if not (external_runtime / "envs" / f"{task}.py").is_file():
        raise FileNotFoundError(f"external runtime has no task environment for {task}")
    if not (external_runtime / "task_config" / f"{task_config}.yml").is_file():
        raise FileNotFoundError(f"external runtime has no task config for {task_config}")

    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]", "_", policy_run.name)
    train_config_name = f"detail_v2_{safe_run_name}"
    staging = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    try:
        runtime = staging / "runtime"
        staging.mkdir(parents=True)
        shutil.copytree(
            external_runtime,
            runtime,
            symlinks=True,
            ignore=_copy_ignore,
        )
        official_env_hashes = _tree_python_hashes(external_runtime / "envs")
        shutil.copytree(
            source / "encoder",
            runtime / "encoder",
            dirs_exist_ok=True,
            symlinks=True,
            ignore=_copy_ignore,
        )
        shutil.copytree(
            source / "policy" / "ACT",
            runtime / "policy" / "ACT",
            dirs_exist_ok=True,
            symlinks=True,
            ignore=_copy_ignore,
        )
        _make_copy_writable(runtime)
        shutil.copy2(Path(__file__).with_name("eval_loop_v2.py"), runtime / "eval_loop_v2.py")
        _patch_eval_script(runtime / "scripts" / "eval_policy.py")
        if _tree_python_hashes(runtime / "envs") != official_env_hashes:
            raise RuntimeError("official environment code changed while preparing runtime")

        config_file = runtime / "policy" / "ACT" / f"{train_config_name}.yml"
        shutil.copy2(policy_run / "train_config.yml", config_file)
        checkpoint_dir = (
            runtime
            / "policy"
            / "ACT"
            / "act_ckpt"
            / f"act-{task}"
            / f"{task_config}-{episode_count}"
            / train_config_name
        )
        checkpoint_dir.mkdir(parents=True)
        checkpoint_link = checkpoint_dir / "policy_last.ckpt"
        stats_link = checkpoint_dir / "dataset_stats.pkl"
        checkpoint_link.symlink_to(policy_run / "policy_last.ckpt")
        stats_link.symlink_to(policy_run / "dataset_stats.pkl")
        shutil.copy2(seeds_path, staging / "seeds.json")

        source_tree_sha256 = _source_tree_sha256(source)
        train_config_sha256 = _sha256(policy_run / "train_config.yml")
        manifest = {
            "status": "prepared",
            "mode": "smoke" if smoke_steps is not None else "formal",
            "runtime": str((out / "runtime").resolve()),
            "external_runtime": str(external_runtime),
            "source": str(source),
            "policy_run": str(policy_run),
            "source_commit": completion["source_commit"],
            "task": task,
            "task_config": task_config,
            "episode_count": episode_count,
            "gpu": gpu,
            "requested_seeds": seeds,
            "seed_manifest_sha256": _sha256(seeds_path),
            "smoke_steps": smoke_steps,
            "smoke_policy": is_smoke_policy,
            "train_config_name": train_config_name,
            "train_config_file": config_file.name,
            "checkpoint_runtime_path": str(checkpoint_link.relative_to(runtime)),
            "stats_runtime_path": str(stats_link.relative_to(runtime)),
            "checkpoint_sha256": completion["checkpoint_sha256"],
            "dataset_stats_sha256": completion["dataset_stats_sha256"],
            "encoder_sha256": completion["encoder_sha256"],
            "tactile_encoder_type": config["tactile_encoder_type"],
            "tactile_sequence_length": config.get("tactile_sequence_length", 1),
            "tactile_temporal_stride": config.get("tactile_temporal_stride", 1),
            "optimizer_updates": completion["optimizer_updates"],
            "train_config_sha256": train_config_sha256,
            "source_tree_sha256": source_tree_sha256,
            "official_env_hashes": official_env_hashes,
        }
        completion_context = {
            "policy_checkpoint_sha256": completion["checkpoint_sha256"],
            "dataset_stats_sha256": completion["dataset_stats_sha256"],
            "encoder_sha256": completion["encoder_sha256"],
            "tactile_encoder_type": config["tactile_encoder_type"],
            "tactile_sequence_length": config.get("tactile_sequence_length", 1),
            "tactile_temporal_stride": config.get("tactile_temporal_stride", 1),
            "policy_source_commit": completion["source_commit"],
            "train_config_sha256": train_config_sha256,
            "seed_manifest_sha256": _sha256(seeds_path),
            "source_tree_sha256": source_tree_sha256,
            "requested_seeds": seeds,
            "policy_run": str(policy_run),
            "task": task,
        }
        (staging / "completion_context.json").write_text(
            json.dumps(completion_context, indent=2, sort_keys=True)
        )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        staging.replace(out)
    except Exception:
        if staging.exists():
            _make_copy_writable(staging)
            shutil.rmtree(staging)
        raise
    return manifest


def _launch(out: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    runtime = Path(manifest["runtime"])
    seeds = manifest["requested_seeds"]
    total = 1 if manifest["mode"] == "smoke" else len(seeds)
    required_environment = {
        name: os.environ.get(name)
        for name in ("CONDA_SH", "CONDA_ENV", "ISAAC_SIM_ROOT")
    }
    missing_environment = [name for name, value in required_environment.items() if not value]
    if missing_environment:
        raise RuntimeError(
            f"--launch requires environment variables: {missing_environment}"
        )
    conda_sh = Path(required_environment["CONDA_SH"]).resolve()
    isaac_root = Path(required_environment["ISAAC_SIM_ROOT"]).resolve()
    isaac_python = isaac_root / "kit" / "python" / "bin" / "python3"
    setup_script = isaac_root / "setup_conda_env.sh"
    for required in (conda_sh, isaac_python, setup_script):
        if not required.exists():
            raise FileNotFoundError(required)

    eval_arguments = [
        "run_eval_compat.py",
        "scripts/eval_policy.py",
        manifest["task"],
        manifest["task_config"],
        "ACT/deploy",
        "--headless",
        "--device",
        "cuda:0",
        "--total_num",
        str(total),
        "--start_seed",
        str(seeds[0]),
    ]
    shell = """set -eo pipefail
conda_sh_path=$1
conda_env_name=$2
isaac_root_path=$3
shift 3
source "$conda_sh_path"
conda activate "$conda_env_name"
source "$isaac_root_path/setup_conda_env.sh"
exec "$isaac_root_path/kit/python/bin/python3" "$@"
"""
    command = [
        "bash",
        "-c",
        shell,
        "detail-v2-eval",
        str(conda_sh),
        str(required_environment["CONDA_ENV"]),
        str(isaac_root),
        *eval_arguments,
    ]
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(manifest["gpu"]),
        TRAIN_CONFIG=str(manifest["train_config_name"]),
        EP_NUM=str(manifest["episode_count"]),
        V2_EVAL_SEEDS=str((out / "seeds.json").resolve()),
        V2_EVAL_OUTPUT=str((out / "results").resolve()),
        V2_EVAL_CONTEXT=str((out / "completion_context.json").resolve()),
    )
    if manifest["smoke_steps"] is not None:
        environment["V2_SMOKE_STEPS"] = str(manifest["smoke_steps"])
    log_path = out / "eval.log"
    with log_path.open("ab", buffering=0) as log_stream:
        process = subprocess.run(
            command,
            cwd=runtime,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    launch = {"returncode": process.returncode, "command": command, "log": str(log_path)}
    if process.returncode != 0:
        launch["status"] = "failed"
        (out / "launch.json").write_text(json.dumps(launch, indent=2))
        raise subprocess.CalledProcessError(process.returncode, command)

    completion_path = out / "results" / "completion.json"
    if not completion_path.is_file():
        raise RuntimeError("eval process exited without results/completion.json")
    completion = dict(_read_json(completion_path))
    expected_status = "smoke_pass" if manifest["mode"] == "smoke" else "completed"
    if completion.get("status") != expected_status:
        raise RuntimeError(
            f"eval completion status is {completion.get('status')!r}, expected {expected_status!r}"
        )
    context = _read_json(out / "completion_context.json")
    mismatches = {
        key: {"expected": value, "actual": completion.get(key)}
        for key, value in context.items()
        if completion.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"eval completion contract mismatch: {mismatches}")
    if manifest["mode"] == "formal" and completion.get("valid_episode_count") != len(seeds):
        raise RuntimeError("formal eval completed without every requested seed")
    launch.update(status="completed", completion=str(completion_path))
    (out / "launch.json").write_text(json.dumps(launch, indent=2))
    return launch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-runtime", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--policy-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    manifest = prepare_runtime(
        external_runtime=args.external_runtime,
        source=args.source,
        policy_run=args.policy_run,
        out=args.out,
        gpu=args.gpu,
        seeds_path=args.seeds,
        smoke_steps=args.smoke_steps,
        allow_smoke=args.allow_smoke,
    )
    result: dict[str, Any] = manifest
    if args.launch:
        result = {"manifest": manifest, "launch": _launch(Path(args.out).resolve(), manifest)}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

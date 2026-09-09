import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts.eval_loop_v2 import load_seed_list, run_eval_loop
from scripts.prepare_eval_v2 import _launch, prepare_runtime


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeTask:
    def __init__(self, output: Path, *, reset_failures=None):
        self.save_root = output
        self.cfg = SimpleNamespace(step_lim=3)
        self.reset_failures = dict(reset_failures or {})
        self.reset_calls = []
        self.take_action_cnt = 0
        self.eval_success = False
        self.step_count = 0
        self.mode = "eval"

    def reset(self, seed, instructions):
        self.reset_calls.append(seed)
        self.take_action_cnt = 0
        self.step_count = 0
        self.eval_success = False
        if self.reset_failures.get(seed, 0):
            self.reset_failures[seed] -= 1
            raise RuntimeError(f"reset failed for {seed}")

    def _get_observations(self):
        value = self.take_action_cnt + len(self.reset_calls)
        return {
            "tactile": {
                "left_tactile": {"rgb_marker": torch.full((2, 2, 3), value)},
                "right_tactile": {"rgb_marker": torch.full((2, 2, 3), value + 1)},
            }
        }

    def check_early_stop(self):
        return False

    def clean_cache(self, result):
        pass


class FakePolicy:
    def reset(self):
        pass

    def eval(self, task, observation):
        task.take_action_cnt += 1
        task.step_count += 1
        task.eval_success = task.take_action_cnt == 1


def test_seed_manifest_preserves_exact_order_and_rejects_invalid_lists(tmp_path):
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps([9, 2, 5]))
    assert load_seed_list(path) == [9, 2, 5]

    for invalid in ([1, 1], [True, 2], [1, "2"], [], {"seeds": [1, 2]}):
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="seed"):
            load_seed_list(path)


def test_formal_eval_keeps_seed_identity_and_refuses_completed_when_one_is_invalid(
    tmp_path,
):
    task = FakeTask(tmp_path, reset_failures={4: 3})

    result = run_eval_loop(
        task,
        FakePolicy(),
        seeds=[3, 4, 8],
        instructions={"seen": ["insert"]},
        instruction_type="seen",
        log=lambda _message: None,
    )

    assert task.reset_calls == [3, 4, 4, 4, 8]
    assert result["status"] == "incomplete"
    assert result["requested_seeds"] == [3, 4, 8]
    assert result["valid_episode_count"] == 2
    assert result["success_count"] == 2
    assert result["infrastructure_error_count"] == 1
    assert "success_rate" not in result
    rows = [json.loads(line) for line in (tmp_path / "outcomes.jsonl").read_text().splitlines()]
    assert [row["seed"] for row in rows] == [3, 4, 8]
    assert rows[1]["termination"] == "reset_infrastructure_error"


def test_smoke_runs_exact_steps_and_reports_tactile_and_action_updates(tmp_path):
    task = FakeTask(tmp_path)

    result = run_eval_loop(
        task,
        FakePolicy(),
        seeds=[17],
        instructions={"seen": ["insert"]},
        instruction_type="seen",
        log=lambda _message: None,
        smoke_steps=3,
        completion_context={
            "policy_checkpoint_sha256": "policy-hash",
            "source_tree_sha256": "source-hash",
        },
    )

    assert result["status"] == "smoke_pass"
    assert result["seed"] == 17
    assert result["requested_steps"] == 3
    assert result["action_count"] == 3
    assert result["tactile_observation_count"] == 3
    assert result["tactile_change_count"] == 2
    assert result["unique_tactile_hashes"] == 3
    assert result["policy_checkpoint_sha256"] == "policy-hash"
    assert result["source_tree_sha256"] == "source-hash"
    assert "success_rate" not in result
    telemetry = [
        json.loads(line)
        for line in (tmp_path / "step_telemetry.jsonl").read_text().splitlines()
    ]
    assert [row["seed"] for row in telemetry] == [17, 17, 17]
    assert all(row["actions_after"] > row["actions_before"] for row in telemetry)


def _write_eval_script(path: Path):
    path.write_text(
        "def eval_policy(task, policy, expert_check, start_seed, max_seed, "
        "test_total_num, instructions, instruciton_type='seen'):\n"
        "    return {'test_num': 0, 'succ_num': 0}\n"
        "\ndef get_config(file, default_root, type):\n"
        "    return {}, file\n"
        "\ndef main():\n"
        "    env_cfg.save_dir = Path('eval_result') / 'old'\n"
        "    results = eval_policy(task, policy, False, 0, -1, 1, instructions)\n"
        "    log(f\"Final Result: {results['succ_num']}/{results['test_num']}\")\n"
        "    task.close()\n"
        "    policy.close()\n"
        "    simulation_app.close()\n"
    )


def _runtime_fixture(tmp_path: Path):
    external = tmp_path / "official"
    (external / "scripts").mkdir(parents=True)
    (external / "envs").mkdir()
    (external / "policy" / "ACT").mkdir(parents=True)
    (external / "task_config").mkdir()
    (external / "run_eval_compat.py").write_text("# compat\n")
    (external / "envs" / "insert_HDMI.py").write_text("# official env\n")
    (external / "envs" / "._insert_HDMI.py").write_bytes(b"mac metadata")
    (external / "task_config" / "demo.yml").write_text("observations: {}\n")
    _write_eval_script(external / "scripts" / "eval_policy.py")

    source = tmp_path / "source"
    (source / "encoder").mkdir(parents=True)
    (source / "policy" / "ACT").mkdir(parents=True)
    (source / "encoder" / "detail_v2.py").write_text("MODEL = 'v2'\n")
    (source / "policy" / "ACT" / "deploy_policy.py").write_text("POLICY = 'v2'\n")

    policy_run = tmp_path / "policy_run"
    policy_run.mkdir()
    checkpoint = policy_run / "policy_last.ckpt"
    stats = policy_run / "dataset_stats.pkl"
    tactile_checkpoint = tmp_path / "encoder.pt"
    checkpoint.write_bytes(b"policy")
    stats.write_bytes(b"stats")
    tactile_checkpoint.write_bytes(b"encoder")
    config = {
        "task_name": "sim-insert_HDMI-demo-50",
        "tactile_type": "feat",
        "tactile_encoder_type": "detail_v2",
        "tactile_ckpt": str(tactile_checkpoint.resolve()),
        "num_steps": 4000,
    }
    (policy_run / "train_config.yml").write_text(yaml.safe_dump(config))
    completion = {
        "status": "completed",
        "smoke": False,
        "checkpoint": "policy_last.ckpt",
        "checkpoint_sha256": _sha(checkpoint),
        "dataset_stats_sha256": _sha(stats),
        "source_commit": "abc123",
        "encoder_sha256": _sha(tactile_checkpoint),
        "optimizer_updates": 4000,
    }
    (policy_run / "completion.json").write_text(json.dumps(completion))
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps([0, 7, 11]))
    return external, source, policy_run, seeds


def test_prepare_creates_isolated_official_runtime_with_only_requested_overlays(tmp_path):
    external, source, policy_run, seeds = _runtime_fixture(tmp_path)
    out = tmp_path / "eval"

    prepared = prepare_runtime(
        external_runtime=external,
        source=source,
        policy_run=policy_run,
        out=out,
        gpu="2",
        seeds_path=seeds,
    )

    runtime = out / "runtime"
    assert prepared["runtime"] == str(runtime.resolve())
    assert (runtime / "envs" / "insert_HDMI.py").read_text() == "# official env\n"
    assert (runtime / "encoder" / "detail_v2.py").read_text() == "MODEL = 'v2'\n"
    assert (runtime / "policy" / "ACT" / "deploy_policy.py").read_text() == "POLICY = 'v2'\n"
    assert "run_eval_loop" in (runtime / "scripts" / "eval_policy.py").read_text()
    assert json.loads((out / "seeds.json").read_text()) == [0, 7, 11]
    assert (runtime / "policy" / "ACT" / prepared["train_config_file"]).read_bytes() == (
        policy_run / "train_config.yml"
    ).read_bytes()
    checkpoint_link = runtime / prepared["checkpoint_runtime_path"]
    stats_link = runtime / prepared["stats_runtime_path"]
    assert checkpoint_link.resolve() == policy_run / "policy_last.ckpt"
    assert stats_link.resolve() == policy_run / "dataset_stats.pkl"
    assert prepared["checkpoint_sha256"] == _sha(policy_run / "policy_last.ckpt")
    assert prepared["requested_seeds"] == [0, 7, 11]
    assert prepared["status"] == "prepared"
    completion_context = json.loads((out / "completion_context.json").read_text())
    assert completion_context["policy_checkpoint_sha256"] == _sha(
        policy_run / "policy_last.ckpt"
    )
    assert completion_context["requested_seeds"] == [0, 7, 11]
    assert completion_context["train_config_sha256"] == _sha(
        policy_run / "train_config.yml"
    )
    assert len(completion_context["source_tree_sha256"]) == 64


def test_prepare_rejects_checkpoint_drift_and_unapproved_smoke_policy(tmp_path):
    external, source, policy_run, seeds = _runtime_fixture(tmp_path)
    completion_path = policy_run / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["checkpoint_sha256"] = "0" * 64
    completion_path.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        prepare_runtime(
            external_runtime=external,
            source=source,
            policy_run=policy_run,
            out=tmp_path / "bad_hash",
            gpu="0",
            seeds_path=seeds,
        )

    completion["checkpoint_sha256"] = _sha(policy_run / "policy_last.ckpt")
    completion["smoke"] = True
    completion["optimizer_updates"] = 2
    completion_path.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="allow-smoke"):
        prepare_runtime(
            external_runtime=external,
            source=source,
            policy_run=policy_run,
            out=tmp_path / "smoke",
            gpu="0",
            seeds_path=seeds,
            smoke_steps=3,
        )


def test_prepare_accepts_smoke_only_with_explicit_flag_and_smoke_steps(tmp_path):
    external, source, policy_run, seeds = _runtime_fixture(tmp_path)
    completion_path = policy_run / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["smoke"] = True
    completion["optimizer_updates"] = 2
    completion_path.write_text(json.dumps(completion))

    prepared = prepare_runtime(
        external_runtime=external,
        source=source,
        policy_run=policy_run,
        out=tmp_path / "smoke",
        gpu="0",
        seeds_path=seeds,
        smoke_steps=3,
        allow_smoke=True,
    )

    assert prepared["mode"] == "smoke"
    assert prepared["smoke_steps"] == 3


def test_launch_waits_for_eval_completion_and_checks_injected_contract(tmp_path, monkeypatch):
    external, source, policy_run, seeds = _runtime_fixture(tmp_path)
    out = tmp_path / "eval"
    prepared = prepare_runtime(
        external_runtime=external,
        source=source,
        policy_run=policy_run,
        out=out,
        gpu="2",
        seeds_path=seeds,
    )
    (out / "runtime" / "run_eval_compat.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "context=json.loads(Path(os.environ['V2_EVAL_CONTEXT']).read_text())\n"
        "result={**context,'status':'completed','valid_episode_count':3,"
        "'success_count':1,'infrastructure_error_count':0}\n"
        "target=Path(os.environ['V2_EVAL_OUTPUT']);target.mkdir(parents=True)\n"
        "(target/'completion.json').write_text(json.dumps(result))\n"
    )
    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text("conda() { return 0; }\n")
    isaac_root = tmp_path / "isaac"
    (isaac_root / "kit" / "python" / "bin").mkdir(parents=True)
    (isaac_root / "setup_conda_env.sh").write_text(":\n")
    (isaac_root / "kit" / "python" / "bin" / "python3").symlink_to(
        Path(__import__("sys").executable)
    )
    monkeypatch.setenv("CONDA_SH", str(conda_sh))
    monkeypatch.setenv("CONDA_ENV", "UniVTAC")
    monkeypatch.setenv("ISAAC_SIM_ROOT", str(isaac_root))

    launch = _launch(out, prepared)

    assert launch["status"] == "completed"
    assert launch["returncode"] == 0
    completion = json.loads((out / "results" / "completion.json").read_text())
    assert completion["policy_checkpoint_sha256"] == prepared["checkpoint_sha256"]
    assert completion["requested_seeds"] == [0, 7, 11]


def test_readonly_copy_permissions_do_not_follow_external_symlinks(tmp_path):
    from scripts.prepare_eval_v2 import _make_copy_writable
    root = tmp_path / 'copy'
    root.mkdir()
    child = root / 'model.py'
    child.write_text('pass')
    external = tmp_path / 'external'
    external.mkdir()
    asset = external / 'asset'
    asset.write_text('preserve')
    asset.chmod(0o444)
    (root / 'assets').symlink_to(external, target_is_directory=True)
    child.chmod(0o444)
    root.chmod(0o555)
    _make_copy_writable(root)
    assert root.stat().st_mode & 0o200
    assert child.stat().st_mode & 0o200
    assert not asset.stat().st_mode & 0o200

"""Policy-training seed must come from the experiment config."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from scripts.train_policy_v2 import resolve_policy_config


def _resolve(tmp_path, config):
    config_path = tmp_path / "policy.yml"
    config_path.write_text(yaml.safe_dump(config))
    return resolve_policy_config(config_path, tmp_path / "encoder.pt", tmp_path / "run")


def test_policy_seed_defaults_to_original_seed_42(tmp_path):
    resolved = _resolve(tmp_path, {"num_steps": 4000})

    assert resolved["seed"] == 42


def test_policy_seed_from_config_is_not_overridden(tmp_path):
    resolved = _resolve(tmp_path, {"num_steps": 4000, "seed": 1})

    assert resolved["seed"] == 1


def test_seed1_config_changes_only_the_policy_seed():
    baseline = yaml.safe_load((PROJECT_ROOT / "configs" / "policy_v2.yml").read_text())
    replica = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "policy_v2_seed1.yml").read_text()
    )

    assert baseline["seed"] == 42
    assert replica["seed"] == 1
    assert {key: value for key, value in baseline.items() if key != "seed"} == {
        key: value for key, value in replica.items() if key != "seed"
    }

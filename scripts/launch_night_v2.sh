#!/usr/bin/env bash
set -euo pipefail
base="$HOME/details_v2_20260909"
source_release="$1"
initial_pid="$2"
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_ENV=UniVTAC
export ISAAC_SIM_ROOT="$HOME/isaacsim-4.5.0"
export PYTHONPATH="$source_release"
exec "$base/venv/bin/python" "$source_release/scripts/night_v2_control.py" \
 --base "$base" --initial-eval-pid "$initial_pid" --source "$source_release" \
 --external-runtime "$HOME/UniVTAC details/reproduction_official_20260905/eval_runtime_official" \
 --policy-data "$HOME/UniVTAC details v2/policy/ACT/data/sim-insert_HDMI/demo-50" --gpu 2

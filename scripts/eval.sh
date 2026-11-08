#!/bin/bash
# Closed-loop evaluation, 50 episodes per task (a later --n_episodes N in the extra args overrides it).
#   bash scripts/eval.sh <policy checkpoint: stage3, self_evolve* or baselines/*/stage3> [suite] [extra args, e.g. --record_video]
# suite: libero_target (default; tasks A-E) | libero_camera | libero_bottle
set -euo pipefail
caller=$PWD
cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-egl}
if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/eval.sh <policy checkpoint> [suite] [evaluation options]" >&2
    exit 2
fi
if ! command -v python >/dev/null; then
    echo "python not found; activate the environment first (source .venv/bin/activate)" >&2
    exit 1
fi
if [[ -z "${python_config_path:-}" ]]; then
    echo "python_config_path is not set; source env.sh after activating the environment (in every new shell and batch job)" >&2
    exit 1
fi
checkpoint=$1
# A relative checkpoint path is relative to the directory the script is called from.
if [[ "$checkpoint" != /* && "$caller" != "$PWD" ]]; then
    checkpoint=$caller/$checkpoint
fi
shift
suite=libero_target
if [[ $# -gt 0 && "$1" != --* ]]; then
    suite=$1
    shift
fi
python -m freeact.eval.run --vla_path "$checkpoint" --task "$suite" --n_episodes 50 --seed 42 "$@"

"""Distributed closed-loop evaluation on the target suites (Ray)."""

# First, before anything imports LIBERO: writes its config so that the import never prompts, and
# registers the target suites (``python -m freeact.eval.run`` imports this package before run.py).
import freeact.envs  # noqa: F401

from .coordinator import EvalTask, run_step_level_evaluation

__all__ = ["EvalTask", "run_step_level_evaluation"]

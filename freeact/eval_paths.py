"""Allocate evaluation output directories without importing the inference / environment stack."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def _label(value: str) -> str:
    # Keep each label a short path component, even for long or unusual checkpoint names.
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80] or "checkpoint"


def _checkpoint_label(vla_path: str | Path) -> str:
    """Parent and name of the checkpoint directory, e.g. ``lapa_stage3`` for ``.../baselines/lapa/stage3``."""
    path = Path(vla_path).absolute()
    return _label("_".join(p.name for p in (path.parent, path) if p.name))


def create_eval_run_dir(output_dir: str | Path, vla_path: str | Path, suite: str, seed: int) -> Path:
    """Atomically reserve a fresh directory for one evaluation, never reusing an old run.

    Suite, seed and the checkpoint's parent and name make runs recognizable; microseconds and a UUID
    distinguish concurrent jobs and repeated evaluations of the same checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_label(vla_path)
    prefix = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{_label(suite)}_{seed}_{checkpoint}"
    while True:
        run_dir = output_dir / f"{prefix}_{uuid4().hex}"
        try:
            # mkdir is the reservation: an existence check followed by exist_ok=True races.
            run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return run_dir

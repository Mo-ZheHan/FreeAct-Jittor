"""Closed-loop evaluation of a FreeAct policy on the target suites (libero_target, libero_camera, libero_bottle).

    python -m freeact.eval.run --vla_path <ckpt> --task libero_target

By default every task of the suite is evaluated: ``libero_target`` holds the paper's tasks A-E,
``libero_camera`` / ``libero_bottle`` one task each.  50 episodes per task (one per fixed initial
state), seed 42.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import ray

os.environ.setdefault("MUJOCO_GL", "egl")

import freeact.envs  # noqa: F401  (importing registers the target suites)
from freeact.envs.libero_env import get_suite
from freeact.eval import EvalTask, run_step_level_evaluation
from freeact.eval_paths import create_eval_run_dir

logger = logging.getLogger(__name__)


def summarize(episodes) -> dict:
    def stats(eps):
        return {"n_episodes": len(eps), "pc_success": float(np.mean([e.success for e in eps]) * 100)}

    by_group, by_task = defaultdict(list), defaultdict(list)
    for e in episodes:
        by_group[e.task_group].append(e)
        by_task[f"{e.task_group}/{e.task_id}"].append(e)
    return {
        "overall": stats(episodes),
        "per_group": {g: stats(v) for g, v in by_group.items()},
        "per_task": {k: stats(v) for k, v in sorted(by_task.items())},
    }


def _visible_gpus() -> int:
    import subprocess

    if (env := os.environ.get("CUDA_VISIBLE_DEVICES")) is not None:
        return len([d for d in env.split(",") if d.strip()])
    return len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())


POLICY_FILES = ("adapter_config.json", "action_decoder.pt", "action_stats.json")


def _checkpoint_problem(path: Path) -> str | None:
    """Why ``path`` is not a policy checkpoint (LoRA adapter and action head), or None if it is one."""
    missing = [f for f in POLICY_FILES if not (path / f).is_file()]
    if not missing:
        return None
    msg = f"{path} is not a policy checkpoint (missing {', '.join(missing)})"
    if candidates := sorted(str(c.parent) for d in ("*", "*/*", "*/*/*") for c in path.glob(f"{d}/action_decoder.pt")):
        return f"{msg}; checkpoints below it: {', '.join(candidates[:5])}"
    if (path / "adapter_config.json").is_file():
        return f"{msg}; stage-2 checkpoints have no action head, evaluate a stage-3 or self-evolved one"
    return msg


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vla_path", required=True, help="policy checkpoint directory (stage3, self_evolve* or baselines/*/stage3)")
    p.add_argument("--task", default="libero_target", help="comma-separated task suites")
    p.add_argument(
        "--task_ids",
        default=None,
        help="comma-separated task ids; default: all tasks of the suite",
    )
    p.add_argument("--n_episodes", type=int, default=50, help="rollouts per task (1-50, fixed initial states)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="./outputs/eval")
    p.add_argument("--num_gpus", type=int, default=-1, help="-1: all visible GPUs")
    p.add_argument("--max_envs", type=int, default=16, help="max concurrent environments")
    p.add_argument("--record_video", action="store_true")
    p.add_argument(
        "--save_rollouts",
        action="store_true",
        help="save every rollout as LIBERO-format HDF5 (input to self-evolution in the PyTorch implementation)",
    )
    args = p.parse_args()

    np.random.seed(args.seed)

    groups = [t.strip() for t in args.task.split(",") if t.strip()]
    if not groups:
        p.error("--task must contain at least one suite")
    if not Path(args.vla_path).is_dir():
        p.error(f"policy checkpoint directory not found: {args.vla_path}")
    if problem := _checkpoint_problem(Path(args.vla_path)):
        p.error(problem)
    if not 1 <= args.n_episodes <= 50:
        p.error("--n_episodes must be between 1 and 50 (fixed initial states)")
    if args.max_envs < 1:
        p.error("--max_envs must be positive")
    if args.num_gpus != -1 and args.num_gpus < 1:
        p.error("--num_gpus must be -1 or positive")
    num_gpus = args.num_gpus if args.num_gpus > 0 else _visible_gpus()

    if num_gpus < 1:
        p.error("evaluation requires at least one visible CUDA GPU")

    tasks = []
    for g in groups:
        try:
            n_tasks = len(get_suite(g).tasks)
        except KeyError:
            p.error(f"unknown suite {g!r}; choose from libero_target, libero_camera, libero_bottle")
        ids = range(n_tasks) if args.task_ids is None else [int(x) for x in args.task_ids.split(",")]
        if not ids:
            p.error("--task_ids must contain at least one task")
        if bad := [i for i in ids if not 0 <= i < n_tasks]:
            p.error(f"--task_ids {bad} out of range for {g} ({n_tasks} tasks)")
        tasks += [
            EvalTask(task_group=g, task_id=t, seed=args.seed + ep, episode_index=ep)
            for t in ids
            for ep in range(args.n_episodes)
        ]
    logger.info(f"{len(tasks)} episodes on {num_gpus} GPUs")

    run_dir = create_eval_run_dir(args.output_dir, args.vla_path, groups[0], args.seed)

    start = time.time()
    # Ray stops creating objects once the disk is 95% full by default; allow up to 99%.
    ray.init(num_gpus=num_gpus, _system_config={"local_fs_capacity_threshold": 0.99})
    try:
        episodes = run_step_level_evaluation(
            vla_path=args.vla_path,
            tasks=tasks,
            num_gpus=num_gpus,
            max_envs=args.max_envs,
            videos_dir=run_dir / "videos" if args.record_video else None,
            rollouts_dir=run_dir / "rollouts" if args.save_rollouts else None,
            progress_file=run_dir / "progress.txt",
            seed=args.seed,
        )
    finally:
        ray.shutdown()

    results = summarize(episodes)
    results.update(config=vars(args), eval_seconds=time.time() - start)
    results["episodes"] = [
        {"task_group": e.task_group, "task_id": e.task_id, "episode": e.episode_index,
         "success": bool(e.success), "n_steps": e.n_steps, "instruction": e.task_instruction}
        for e in episodes
    ]
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Overall: {results['overall']['pc_success']:.1f}% over {results['overall']['n_episodes']} episodes")
    for k, v in results["per_task"].items():
        print(f"  {k}: {v['pc_success']:.1f}% ({v['n_episodes']})")
    print(f"Results: {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()

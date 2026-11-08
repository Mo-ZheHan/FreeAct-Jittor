"""Asynchronous coordinator for step-level distributed evaluation.

EnvActors (CPU, one Ray actor per episode) step the simulators while InferenceWorkers (one per
GPU) serve observations from a shared pool, so no GPU waits on a single slow environment.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple, Optional

import h5py
import numpy as np
import ray

from .env_manager import EnvActor, EpisodeResult
from .inference_worker import InferenceWorker
from .step_task import StepTask

logger = logging.getLogger(__name__)


class PendingTask(NamedTuple):
    task_type: str  # "obs" | "infer" | "step" | "done"
    env_id: int
    worker_idx: int = -1  # only for "infer"


@dataclass
class EvalTask:
    task_group: str
    task_id: int
    seed: int
    episode_index: int = 0


@ray.remote
def _write_video_task(path: str, frames, fps: int) -> str:
    import av

    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames.shape[1:3]
        stream.pix_fmt = "yuv420p"
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _status(result: EpisodeResult) -> str:
    if result.success:
        return "success"
    return "error" if result.n_steps == 0 else "failure"


def _format_time(seconds: float) -> str:
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def _task_dirname(result: EpisodeResult) -> str:
    return f"{result.task_group}/task{result.task_id}"


def _save_video(videos_dir: Path, result: EpisodeResult) -> ray.ObjectRef:
    task_dir = videos_dir / _task_dirname(result)
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"{_status(result)}_ep_{result.episode_index}.mp4"
    return _write_video_task.remote(str(path), result.frames, result.render_fps)


def write_rollout_hdf5(path: Path, payload: dict[str, Any], episode_index: int, success: bool) -> None:
    """Write one rollout as a LIBERO-format HDF5 file (single ``demo_0``).

    ``obs[i]``, ``states[i]`` and ``actions[i]`` are all captured *before* ``env.step(actions[i])``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = payload["meta"]
    tmp = path.with_suffix(".hdf5.tmp")
    with h5py.File(tmp, "w") as f:
        grp = f.create_group("data")
        grp.attrs["env_name"] = meta["env_name"]
        grp.attrs["problem_info"] = json.dumps(meta["problem_info"])
        grp.attrs["bddl_file_content"] = meta.get("bddl_file_content", "")
        grp.attrs["num_demos"] = 1
        grp.attrs["total"] = payload["num_samples"]

        ep = grp.create_group("demo_0")
        ep.attrs["num_samples"] = payload["num_samples"]
        ep.attrs["episode_index"] = episode_index
        ep.attrs["success"] = bool(success)
        ep.attrs["init_state"] = meta.get("init_state", np.zeros(0))
        obs = ep.create_group("obs")
        ee = payload["ee_states"]
        obs.create_dataset("ee_states", data=ee)
        obs.create_dataset("ee_pos", data=ee[:, :3])
        obs.create_dataset("ee_ori", data=ee[:, 3:])
        for k in ("gripper_states", "joint_states", "agentview_rgb", "eye_in_hand_rgb"):
            obs.create_dataset(k, data=payload[k])
        for k in ("actions", "states", "rewards", "dones"):
            ep.create_dataset(k, data=payload[k])
    tmp.replace(path)


def run_step_level_evaluation(
    vla_path: str,
    tasks: list[EvalTask],
    num_gpus: int,
    max_envs: int = 16,
    videos_dir: Optional[Path] = None,
    rollouts_dir: Optional[Path] = None,
    progress_file: Optional[Path] = None,
    seed: Optional[int] = None,
) -> list[EpisodeResult]:
    """Run all ``tasks`` and return their results.

    videos_dir:   if set, save an agent-view / wrist-view video per episode.
    rollouts_dir: if set, save every rollout as ``<task_group>/task<id>_ep<k>.hdf5`` for
                  self-evolution with the PyTorch implementation (``freeact.selfevolve``).
    """
    record_video = videos_dir is not None
    collect_hdf5 = rollouts_dir is not None

    logger.info(f"Creating {num_gpus} inference workers...")
    workers = [
        InferenceWorker.remote(vla_path=vla_path,
                               seed=None if seed is None else seed + 1000 * i)
        for i in range(num_gpus)
    ]

    pending_tasks = list(tasks)
    all_results: list[EpisodeResult] = []
    env_actors: dict[int, Any] = {}
    env_to_task: dict[int, tuple[str, int]] = {}
    pending_refs: dict[ray.ObjectRef, PendingTask] = {}
    worker_busy = [False] * num_gpus
    ready_obs: dict[int, Any] = {}
    video_refs: list[ray.ObjectRef] = []
    next_env_id = next_gpu_id = 0

    pbar_file = open(progress_file, "w") if progress_file else None
    group_success: dict[str, int] = defaultdict(int)
    group_total: dict[str, int] = defaultdict(int)
    start_time = time.time()

    try:
        while pending_tasks or env_actors or pending_refs:
            # 1. Spawn environments (a few per iteration to spread the start-up cost)
            for _ in range(min(4, max_envs - len(env_actors), len(pending_tasks))):
                t = pending_tasks.pop(0)
                actor = EnvActor.remote(
                    t.task_group, t.task_id, t.seed, t.episode_index, record_video, next_gpu_id, collect_hdf5
                )
                next_gpu_id = (next_gpu_id + 1) % num_gpus
                env_actors[next_env_id] = actor
                env_to_task[next_env_id] = (t.task_group, t.task_id)
                pending_refs[actor.get_observation.remote()] = PendingTask("obs", next_env_id)
                next_env_id += 1

            # 2. Dispatch inference to idle workers
            for w_idx, worker in enumerate(workers):
                if worker_busy[w_idx] or not ready_obs:
                    continue
                eid, obs = ready_obs.popitem()
                ref = worker.infer_single.remote(StepTask(env_id=eid, observation=obs))
                pending_refs[ref] = PendingTask("infer", eid, w_idx)
                worker_busy[w_idx] = True

            if not pending_refs:
                break

            # 3. Process completed tasks
            ready, _ = ray.wait(list(pending_refs.keys()), timeout=0.001)
            for ref in ready:
                task = pending_refs.pop(ref)
                eid = task.env_id

                if task.task_type == "obs":
                    if eid in env_actors and (obs := ray.get(ref)) is not None:
                        ready_obs[eid] = obs

                elif task.task_type == "infer":
                    worker_busy[task.worker_idx] = False
                    if eid in env_actors:
                        result = ray.get(ref)
                        actor = env_actors[eid]
                        if result.action is None:  # inference error: end the episode
                            pending_refs[actor.get_result.remote()] = PendingTask("done", eid)
                        else:
                            pending_refs[actor.step.remote(result.action)] = PendingTask("step", eid)

                elif task.task_type == "step" and eid in env_actors:
                    actor, done = env_actors[eid], ray.get(ref)
                    next_ref = actor.get_result.remote() if done else actor.get_observation.remote()
                    pending_refs[next_ref] = PendingTask("done" if done else "obs", eid)

                elif task.task_type == "done" and eid in env_actors:
                    result = ray.get(ref)
                    group, tid = env_to_task.pop(eid)
                    if record_video and result.frames is not None:
                        video_refs.append(_save_video(videos_dir, result))
                    if collect_hdf5 and result.hdf5_payload is not None:
                        path = rollouts_dir / group / f"task{tid}_ep{result.episode_index}.hdf5"
                        write_rollout_hdf5(path, result.hdf5_payload, result.episode_index, result.success)
                    all_results.append(replace(result, frames=None, hdf5_payload=None))

                    group_total[group] += 1
                    group_success[group] += int(result.success)
                    if pbar_file:
                        n_done = len(all_results)
                        elapsed = time.time() - start_time
                        pace = elapsed / n_done
                        sr = group_success[group] / group_total[group] * 100
                        pbar_file.write(
                            f"[{n_done}/{len(tasks)}] {group} task {tid}: {_status(result)} | "
                            f"SR {sr:.1f}% | {_format_time(elapsed)}, {pace:.1f}s/it, "
                            f"ETA {_format_time((len(tasks) - n_done) * pace)}\n"
                        )
                        pbar_file.flush()
                    del env_actors[eid]

    except KeyboardInterrupt:
        logger.warning("Interrupted, killing all actors...")
        for actor in list(env_actors.values()) + workers:
            ray.kill(actor)
        return all_results
    finally:
        if pbar_file:
            pbar_file.close()

    if video_refs:
        try:
            ray.get(video_refs, timeout=60.0)
        except ray.exceptions.GetTimeoutError:
            logger.warning("Video save timeout, some videos may be incomplete")

    return all_results

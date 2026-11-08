"""LIBERO environment wrapper (torch-free port of ``lerobot.envs.libero.LiberoEnv``, lerobot 0.4.4).

Only the parts the evaluation uses are kept: the ``pixels`` observation of the agent-view and
wrist cameras, fixed initial states (one per episode), 10 settle steps of the no-op action after
every reset, relative control, success-based termination and auto-reset.  Reset / step
semantics are identical to lerobot's, so the PyTorch and Jittor implementations evaluate on the
same simulator states.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import numpy as np

from freeact.utils.checkpoint_io import load_torch_file

TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
CAMERA_NAME_MAPPING = {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}


def get_libero_dummy_action():
    """No-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_task_init_states(task_suite: Any, i: int) -> np.ndarray:
    from libero.libero import get_libero_path

    task = task_suite.tasks[i]
    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    return np.asarray(load_torch_file(path))


class LiberoEnv:
    def __init__(
        self,
        task_suite: Any,
        task_id: int,
        task_suite_name: str,
        camera_name: Sequence[str] = ("agentview_image", "robot0_eye_in_hand_image"),
        observation_width: int = 256,
        observation_height: int = 256,
        episode_index: int = 0,
        num_steps_wait: int = 10,
    ):
        self.task_id = task_id
        self.camera_name = list(camera_name)
        self.observation_width, self.observation_height = observation_width, observation_height
        self.num_steps_wait = num_steps_wait
        self.episode_index = episode_index
        self._init_states = get_task_init_states(task_suite, task_id)
        self.init_state_id = episode_index
        self._env = self._make_envs_task(task_suite, task_id)
        self._max_episode_steps = TASK_SUITE_MAX_STEPS.get(task_suite_name, 500)

    def _make_envs_task(self, task_suite: Any, task_id: int = 0):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_heights=self.observation_height, camera_widths=self.observation_width
        )
        env.reset()
        return env

    def _format_raw_obs(self, raw_obs: dict) -> dict:
        return {"pixels": {CAMERA_NAME_MAPPING[c]: raw_obs[c] for c in self.camera_name}}

    def reset(self, seed: Optional[int] = None):
        self._env.seed(seed)
        self._env.reset()
        raw_obs = self._env.set_init_state(self._init_states[self.init_state_id % len(self._init_states)])
        self.init_state_id += 1
        # After reset, objects may be unstable; let everything settle with the no-op action.
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
        for robot in self._env.robots:
            robot.controller.use_delta = True
        return self._format_raw_obs(raw_obs), {"is_success": False}

    def step(self, action: np.ndarray):
        if action.ndim != 1:
            raise ValueError(f"Expected a 1-D action, got shape {action.shape}")
        raw_obs, reward, done, info = self._env.step(action)
        is_success = self._env.check_success()
        terminated = done or is_success
        info.update(task=self.task, task_id=self.task_id, done=done, is_success=is_success)
        observation = self._format_raw_obs(raw_obs)
        if terminated:
            info["final_info"] = dict(task=self.task, task_id=self.task_id, done=bool(done), is_success=bool(is_success))
            self.reset()
        return observation, reward, terminated, False, info

    def close(self):
        self._env.close()


def get_suite(name: str):
    """Instantiate a LIBERO benchmark suite by name (target suites are registered by ``freeact.envs``)."""
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[name]()

"""One Ray actor per LIBERO episode, so that environment steps run in parallel."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import ray

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Result of a completed episode."""

    task_group: str
    task_id: int
    episode_seed: int
    episode_index: int
    success: bool
    sum_reward: float
    max_reward: float
    n_steps: int
    task_instruction: str = ""  # Task language instruction
    frames: Optional[np.ndarray] = field(default=None, repr=False)  # (T, H, W, C)
    render_fps: int = 10
    # LIBERO-format HDF5 payload, populated when the actor collects rollouts (collect_hdf5)
    hdf5_payload: Optional[dict[str, Any]] = field(default=None, repr=False)


@ray.remote
class EnvActor:
    """Single environment Ray Actor - enables parallel env.step() across actors.

    HDF5 contract (``collect_hdf5=True``): ``(obs[i], state[i], action[i])`` are
    all captured pre-step, i.e. ``obs[i]`` is the observation the policy saw
    when it produced ``action[i]``.
    """

    # Video views: agent camera and wrist camera.
    VIDEO_RENDER_SIZE = 512
    VIDEO_CAMERAS = (
        ("agentview", "robot0_eye_in_hand"),
    )

    def __init__(
        self,
        task_group: str,
        task_id: int,
        episode_seed: int,
        episode_index: int = 0,
        record_video: bool = False,
        gpu_id: int = 0,
        collect_hdf5: bool = False,
    ):
        self.task_group = task_group
        self.task_id = task_id
        self.episode_seed = episode_seed
        self.episode_index = episode_index
        self.record_video = record_video
        self.collect_hdf5 = collect_hdf5

        # Fix EGL rendering: set specific GPU for this actor
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)

        # Target-domain suites (e.g. libero_target) patch LIBERO before env creation
        from freeact.envs import prepare_suite_env

        prepare_suite_env(task_group)

        from freeact.envs.libero_env import LiberoEnv, get_suite

        self.env = LiberoEnv(get_suite(task_group), task_id, task_group, episode_index=episode_index)

        # Reset and get initial observation
        obs, _ = self.env.reset(seed=episode_seed)
        self.max_steps = self.env._max_episode_steps
        self.render_fps = self.env._env.env.control_freq
        self.task_instruction = getattr(self.env, "task_description", "")

        # Preprocess observation
        self.current_obs = self._preprocess(obs)

        # Episode state
        self.step_idx = 0
        self.done = False
        self.success = False
        self.sum_reward = 0.0
        self.max_reward = 0.0

        # Video frames
        self.frames: list[np.ndarray] = []
        if self.record_video:
            self._capture_frame()

        # HDF5 buffers
        self._hdf5_meta: Optional[dict[str, Any]] = None
        self._hdf5_actions: list[np.ndarray] = []
        self._hdf5_states: list[np.ndarray] = []
        self._hdf5_ee_states: list[np.ndarray] = []
        self._hdf5_gripper_states: list[np.ndarray] = []
        self._hdf5_joint_states: list[np.ndarray] = []
        self._hdf5_agentview_images: list[np.ndarray] = []
        self._hdf5_eye_in_hand_images: list[np.ndarray] = []
        if self.collect_hdf5:
            self._hdf5_meta = self._capture_hdf5_static_meta()
            self._append_hdf5_step_obs()  # obs_0, so obs[i] aligns with action[i]

    def _preprocess(self, obs):
        """Policy input: (1, C, H, W) float image in [0, 1] (flipped as in lerobot) + instruction."""
        img = obs["pixels"]["image"].astype(np.float32).transpose(2, 0, 1) / 255.0
        return {
            "observation.images.image": np.ascontiguousarray(img[None, :, ::-1, ::-1]),
            "task": [self.task_instruction],
        }

    # --- HDF5 rollout collection (LIBERO demo schema) ---

    def _inner_envs(self) -> tuple:
        """Return (LiberoEnv, ControlEnv, robosuite_env)."""
        libero_env = self.env
        control_env = libero_env._env
        return libero_env, control_env, control_env.env

    def _capture_hdf5_static_meta(self) -> dict[str, Any]:
        """One-time capture of env name, problem_info, bddl and init_state."""
        _, control_env, sim_env = self._inner_envs()
        bddl_path = sim_env.bddl_file_name
        try:
            with open(bddl_path) as f:
                bddl_content = f.read()
        except OSError:
            bddl_content = ""
        return {
            "env_name": sim_env.__class__.__name__,
            "problem_info": {
                "problem_name": control_env.problem_name,
                "domain_name": control_env.domain_name,
                "language_instruction": self.task_instruction
                or control_env.language_instruction,
            },
            "bddl_file_content": bddl_content,
            "init_state": np.array(sim_env.sim.get_state().flatten(), copy=True),
        }

    def _append_hdf5_step_obs(self) -> None:
        """Snapshot pre-step raw obs + sim state. See class docstring for phase."""
        import robosuite.utils.transform_utils as T

        _, _, sim_env = self._inner_envs()
        raw = sim_env._get_observations()
        self._hdf5_ee_states.append(
            np.hstack((raw["robot0_eef_pos"], T.quat2axisangle(raw["robot0_eef_quat"])))
        )
        self._hdf5_gripper_states.append(np.array(raw["robot0_gripper_qpos"], copy=True))
        self._hdf5_joint_states.append(np.array(raw["robot0_joint_pos"], copy=True))
        self._hdf5_agentview_images.append(
            np.ascontiguousarray(raw["agentview_image"][::-1, ::-1])
        )
        self._hdf5_eye_in_hand_images.append(
            np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1])
        )
        self._hdf5_states.append(np.array(sim_env.sim.get_state().flatten(), copy=True))

    def _build_hdf5_payload(self) -> Optional[dict[str, Any]]:
        """Assemble the per-episode HDF5 payload, or None if unwritable.

        Length mismatches drop the episode: a silently misaligned HDF5 is
        worse than a missing one.
        """
        n = len(self._hdf5_actions)
        if not self.collect_hdf5 or self._hdf5_meta is None or n == 0:
            return None
        required = {
            "agentview_rgb": self._hdf5_agentview_images,
            "eye_in_hand_rgb": self._hdf5_eye_in_hand_images,
            "ee_states": self._hdf5_ee_states,
            "gripper_states": self._hdf5_gripper_states,
            "joint_states": self._hdf5_joint_states,
            "states": self._hdf5_states,
        }
        ep_tag = f"task={self.task_group} id={self.task_id} ep={self.episode_index}"
        if mismatches := {k: len(v) for k, v in required.items() if len(v) != n}:
            logger.error(
                "Dropping HDF5 payload for %s: buffers != %d actions: %s",
                ep_tag, n, mismatches,
            )
            return None
        dones = np.zeros(n, dtype=np.uint8)
        rewards = np.zeros(n, dtype=np.uint8)
        dones[-1] = rewards[-1] = int(self.success)
        payload = {
            "meta": self._hdf5_meta,
            "num_samples": n,
            "actions": np.stack(self._hdf5_actions),
            "states": np.stack(self._hdf5_states),
            "ee_states": np.stack(self._hdf5_ee_states),
            "gripper_states": np.stack(self._hdf5_gripper_states),
            "joint_states": np.stack(self._hdf5_joint_states),
            "agentview_rgb": np.stack(self._hdf5_agentview_images),
            "eye_in_hand_rgb": np.stack(self._hdf5_eye_in_hand_images),
            "rewards": rewards,
            "dones": dones,
        }
        return payload

    def get_observation(self):
        """Return current observation if not done."""
        return None if self.done else self.current_obs

    def step(self, action: np.ndarray) -> bool:
        """Execute action and return whether episode is done."""
        action = action.copy()[np.newaxis, :]  # (7,) -> (1, 7), copy to allow modification

        # Gripper: VLA [0,1] (0=close) → LIBERO [-1,1] (-1=open, +1=close).
        action[:, -1] = 1.0 - 2.0 * np.clip(action[:, -1], 0.0, 1.0)

        # Save the continuous action env.step consumes; the PyTorch implementation's HDF5 -> LeRobot
        # converter (freeact.selfevolve.to_lerobot) inverts the gripper back to [0, 1].
        if self.collect_hdf5:
            self._hdf5_actions.append(action[0].astype(np.float32, copy=True))

        obs, reward, terminated, truncated, info = self.env.step(action[0])

        if self.record_video:
            self._capture_frame()

        self.step_idx += 1
        self.sum_reward += float(reward)
        self.max_reward = max(self.max_reward, float(reward))

        # Check success
        if "final_info" in info:
            self.success = bool(info["final_info"]["is_success"])

        # Check done
        self.done = terminated or truncated or self.step_idx >= self.max_steps

        if not self.done:
            self.current_obs = self._preprocess(obs)
            # Skip on termination: LiberoEnv auto-resets, so raw_obs is stale.
            if self.collect_hdf5:
                self._append_hdf5_step_obs()

        return self.done

    def _capture_frame(self):
        """Capture the agent and wrist views side by side."""
        sim = self.env._env.sim
        h = w = self.VIDEO_RENDER_SIZE
        rows = [
            np.concatenate(
                [sim.render(camera_name=c, height=h, width=w)[::-1, ::-1] for c in row],
                axis=1,
            )
            for row in self.VIDEO_CAMERAS
        ]
        self.frames.append(np.concatenate(rows, axis=0))

    def get_result(self) -> EpisodeResult:
        """Get episode result."""
        # Exclude last frame if done (LiberoEnv auto-resets on termination)
        frames_array = (
            np.stack(self.frames[: len(self.frames) - self.done])
            if self.frames
            else None
        )
        return EpisodeResult(
            task_group=self.task_group,
            task_id=self.task_id,
            episode_seed=self.episode_seed,
            episode_index=self.episode_index,
            success=self.success,
            sum_reward=self.sum_reward,
            max_reward=self.max_reward,
            n_steps=self.step_idx,
            task_instruction=self.task_instruction,
            frames=frames_array,
            render_fps=self.render_fps,
            hdf5_payload=self._build_hdf5_payload(),
        )

    def __ray_shutdown__(self):
        """Ray lifecycle hook for graceful actor termination."""
        try:
            self.env.close()
        except Exception:
            pass

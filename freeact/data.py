"""Frame pairs from LeRobot datasets with the camera-frame action between them (RQ1 analysis).

Every sample is a frame pair ``(o_t, o_{t+Δ})`` and the 7D camera-frame end-effector displacement between the
two frames, built as in the PyTorch release's stage-1 data pipeline, together with its episode-level
train / held-out split (:func:`stratified_split`).
"""

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from .lerobot import LeRobotFrames
from .transforms import ResizeCenterCrop

# Fixed LIBERO agent-view camera rotation (quat wxyz -> scipy xyzw).
# LIBERO_R_CAM: world -> camera (action labels), LIBERO_R_CAM_INV: camera -> world (inference).
_LIBERO_QUAT = [
    0.3048497438430786,
    0.30484986305236816,
    0.6380177736282349,
    0.6380177736282349,
]
LIBERO_R_CAM = R.from_quat(_LIBERO_QUAT).inv()
LIBERO_R_CAM_INV = R.from_quat(_LIBERO_QUAT)


def libero_to_camera_frame(action: np.ndarray) -> np.ndarray:
    """Transform a 7D LIBERO action from the world frame to the camera frame."""
    dp, dr, dg = action[:3], action[3:6], action[6]
    return np.concatenate([-LIBERO_R_CAM.apply(dp), -LIBERO_R_CAM.apply(dr), [dg]])


def libero_from_camera_frame(action: np.ndarray) -> np.ndarray:
    """Transform a 7D LIBERO action from the camera frame back to the world frame (inference)."""
    dp, dr, dg = action[:3], action[3:6], action[6]
    return np.concatenate([-LIBERO_R_CAM_INV.apply(dp), -LIBERO_R_CAM_INV.apply(dr), [dg]])


_DROID_IMAGE_KEYS = {
    "dual_cam": ["observation.images.exterior_1_left", "observation.images.exterior_2_left"],
    "dual": ["observation.images.exterior_1_left", "observation.images.exterior_2_left"],
    "single_cam": ["observation.images.exterior_0_left"],
    "single": ["observation.images.exterior_0_left"],
}
LIBERO_IMAGE_KEY = "observation.images.image"


def get_image_keys(dataset_name: str) -> List[str]:
    """Camera keys used for a dataset. Each key becomes its own sub-dataset."""
    name = dataset_name.lower()
    if "droid" in name:
        return _DROID_IMAGE_KEYS[name.rstrip("/").split("/")[-1]]
    if "libero" in name:
        return [LIBERO_IMAGE_KEY]
    raise ValueError(f"Unknown dataset '{dataset_name}' (expected a DROID or LIBERO split)")


def torch_randperm(n: int, seed: int) -> np.ndarray:
    """``torch.randperm(n, generator=torch.Generator().manual_seed(seed))`` without PyTorch.

    PyTorch's CPU generator is a standard MT19937 (``init_genrand(seed)``) and ``randperm`` is a
    Fisher-Yates shuffle with ``random() % (n - i)``; reproducing it keeps the stage-1 train /
    held-out split (the RQ1 test set) identical to the PyTorch implementation.
    """
    raw = np.random.RandomState(seed)._bit_generator.random_raw(max(n - 1, 0)).astype(np.uint64)
    r = np.arange(n, dtype=np.int64)
    for i in range(n - 1):
        z = int(raw[i] % np.uint64(n - i))
        r[i], r[z + i] = r[z + i], r[i]
    return r


class MixedLeRobotDataset:
    """
    Mixture of LeRobot datasets that yields frame pairs separated by ``delta_time`` seconds.

    Each ``dataset_spec`` is ``(repo_id, weight, cam_calib)``; ``cam_calib`` marks datasets whose
    actions can be expressed in the camera frame.  The sampling weight is not
    used for feature extraction.  The action is the 7D displacement between the two frames.
    """

    # Filter thresholds for static frames (no visible motion)
    MIN_CHANGE_RATIO = 0.001  # Min fraction of significantly changed pixels (0.1%)
    PIXEL_DIFF_TOLERANCE = 0.05  # Min per-pixel diff to count as "changed" (5%)
    MAX_RESAMPLE_ATTEMPTS = 1024  # Max retries when sample is invalid

    def __init__(
        self,
        dataset_specs: List[Tuple[str, float, bool]],
        resolution: int = 224,
        seed: int = 42,
        root: Optional[str] = None,
        filter_identical_frames: bool = True,
        delta_time: float = 1.0,
    ):
        self.root = root
        self.filter_identical_frames = filter_identical_frames
        self.delta_time = delta_time

        # Each (repo_id, weight, cam_calib) with N image keys becomes N entries (one per image key)
        self.repo_ids: List[str] = []
        self.image_keys: List[str] = []
        self.cam_calib_flags: List[bool] = []
        for repo_id, _, cam_calib in dataset_specs:
            for key in get_image_keys(repo_id):
                self.repo_ids.append(repo_id)
                self.image_keys.append(key)
                self.cam_calib_flags.append(cam_calib)

        # Resampling stream of filtered pairs; re-seeded per batch by the data loader
        self.rng = np.random.default_rng(seed)
        self.image_aug = ResizeCenterCrop(resolution)

        # Cache (root, repo_id) -> (dataset, valid_indices) to avoid duplicate loading
        self._cache: Dict[str, Tuple[LeRobotFrames, np.ndarray]] = {}

        self.datasets: List[LeRobotFrames] = []
        self.dataset_lengths: List[int] = []
        self._valid_indices: List[np.ndarray] = []
        self.cumulative_lengths: List[int] = [0]

        print(f"Initializing mixed dataset: {len(self.repo_ids)} camera streams from {len(set(self.repo_ids))} datasets...")
        for repo_id, image_key in zip(self.repo_ids, self.image_keys):
            dataset, valid_indices = self._get_cached(repo_id, self.root)

            self.datasets.append(dataset)
            self._valid_indices.append(valid_indices)
            self.dataset_lengths.append(len(valid_indices))
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(valid_indices))
            print(
                f"  - {repo_id} [{image_key.split('.')[-1]}]: {len(valid_indices):,} valid / "
                f"{len(dataset):,} total, Δt={self.delta_time:.1f}s"
            )

        self._total_length = self.cumulative_lengths[-1]
        print(f"Total valid samples: {self._total_length:,}")

    def reseed(self, seed: int) -> None:
        """Independent resampling streams per data-loader batch."""
        self.rng = np.random.default_rng(seed)

    def _get_cached(self, repo_id: str, root: Optional[str]) -> Tuple[LeRobotFrames, np.ndarray]:
        """Get or create cached (dataset, valid_indices) for repo_id."""
        cache_key = f"{root}:{repo_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        dt = self.delta_time
        delta_timestamps = {k: [0.0, dt] for k in get_image_keys(repo_id)}

        # State pairs for the displacement between the two frames
        if "droid" in repo_id.lower():
            delta_timestamps["observation.state.cartesian_position"] = [0.0, dt]
            delta_timestamps["observation.state.gripper_position"] = [0.0, dt]
        elif "libero" in repo_id.lower():
            delta_timestamps["observation.state"] = [0.0, dt]

        # the parquet columns that samples are built from (everything else stays on disk)
        need = set(k for k in delta_timestamps if not k.startswith("observation.images"))
        if "droid" in repo_id.lower():
            need |= {f"camera_extrinsics.{k.split('.')[-1]}" for k in get_image_keys(repo_id)}
        dataset = LeRobotFrames(
            repo_id=repo_id,
            root=Path(root).expanduser() / repo_id if root else Path(repo_id),
            delta_timestamps=delta_timestamps,
            image_transforms=self.image_aug,
            columns=sorted(need),
        )
        valid_indices = self._compute_valid_indices(dataset, dt)
        self._cache[cache_key] = (dataset, valid_indices)
        return dataset, valid_indices

    def _compute_valid_indices(self, dataset: LeRobotFrames, delta_time: float) -> np.ndarray:
        """
        Compute valid frame indices by excluding the last N frames of each episode,
        where N = ceil(delta_time * fps). These tail frames cannot form valid pairs.
        Returns a sorted NumPy array of valid global indices.
        """
        exclude = math.ceil(delta_time * dataset.meta.fps)
        ep_from, ep_to = dataset.episode_bounds
        # Valid range for episode is [from, to - exclude)
        valid_ends = np.maximum(ep_from, ep_to - exclude)
        ranges = [np.arange(s, e, dtype=np.int64) for s, e in zip(ep_from, valid_ends) if e > s]
        return np.concatenate(ranges) if ranges else np.array([], dtype=np.int64)

    def __len__(self) -> int:
        return self._total_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample, resampling up to MAX_RESAMPLE_ATTEMPTS times on failure."""
        for i in range(self.MAX_RESAMPLE_ATTEMPTS + 1):
            target = idx if i == 0 else int(self.rng.integers(0, self._total_length))
            if (result := self._try_get_sample(target)) is not None:
                return result
        raise RuntimeError(f"Sample {idx}: failed after {self.MAX_RESAMPLE_ATTEMPTS} retries")

    def _try_get_sample(self, idx: int) -> Optional[Dict[str, Any]]:
        """Attempt to get a sample by global index. Returns None on failure."""
        dataset_idx = int(np.searchsorted(self.cumulative_lengths, idx, side="right") - 1)
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        actual_idx = int(self._valid_indices[dataset_idx][local_idx])

        dataset = self.datasets[dataset_idx]
        dataset_name = self.repo_ids[dataset_idx]
        image_key = self.image_keys[dataset_idx]

        try:
            sample = dataset[actual_idx]
        except Exception as e:
            print(f"[Warning] Failed to load sample {idx} from {dataset_name}: {e}")
            return None

        # Skip padded samples
        padding_key = f"{image_key}_is_pad"
        if padding_key in sample and sample[padding_key].any():
            return None

        return self._process_sample(
            sample,
            dataset_name,
            image_key,
            self.cam_calib_flags[dataset_idx],
        )

    def _process_sample(
        self,
        sample: Dict[str, Any],
        dataset_name: str,
        image_key: str,
        cam_calib: bool,
    ) -> Optional[Dict[str, Any]]:
        """Frame pair + camera-frame action, or None for a pair without visible motion."""
        img0, img1 = sample[image_key]  # [T=2, C, H, W] float in [0, 1]
        if self.filter_identical_frames and not self._has_visible_motion(img0, img1):
            return None

        result = {
            "initial_pixel_values": img0,
            "target_pixel_values": img1,
            "cam_calib": cam_calib,
            "task_instruction": sample.get("task") or "",
            "dataset_name": dataset_name,
        }

        if "droid" in dataset_name.lower():
            action = self._compute_droid_action(sample)
        elif "libero" in dataset_name.lower():
            action = self._compute_libero_action(sample)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if cam_calib:
            action = self._to_camera_frame(action, dataset_name, sample, image_key)

        result["action"] = action
        return result

    def _has_visible_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Check if there is visible motion between two frames."""
        change_ratio = (np.abs(frame1 - frame2) > self.PIXEL_DIFF_TOLERANCE).mean()
        return change_ratio >= self.MIN_CHANGE_RATIO

    def _to_camera_frame(self, action: np.ndarray, dataset_name: str, sample: Dict[str, Any], image_key: str) -> np.ndarray:
        """Transform 7D action [dp, dr, gripper] from base/world to camera frame."""
        dp, dr, dg = action[:3], action[3:6], action[6]

        if "droid" in dataset_name.lower():
            ext = np.asarray(sample[f"camera_extrinsics.{image_key.split('.')[-1]}"])
            rot = R.from_euler("xyz", ext[3:6]).inv()
            return np.concatenate([rot.apply(dp), rot.apply(dr), [dg]])

        if "libero" in dataset_name.lower():
            return libero_to_camera_frame(action)

        raise ValueError(f"Unknown dataset for camera transform: {dataset_name}")

    def _compute_droid_action(self, sample: Dict[str, Any]) -> np.ndarray:
        """Compute 7D action [dp, dr, gripper] for DROID in base frame."""
        cart = np.asarray(sample["observation.state.cartesian_position"])  # [2, 6]
        grip = np.asarray(sample["observation.state.gripper_position"]).reshape(-1)  # [2]

        dp = np.diff(cart[:, :3], axis=0)[0]
        dr = R.from_euler("xyz", cart[1, 3:6]) * R.from_euler("xyz", cart[0, 3:6]).inv()
        dg = np.diff(grip)[0]

        return np.concatenate([dp, dr.as_rotvec(), [dg]])

    def _compute_libero_action(self, sample: Dict[str, Any]) -> np.ndarray:
        """Compute 7D action [dp, dr, gripper] for LIBERO in world frame."""
        s = np.asarray(sample["observation.state"])  # [2, 8]

        dp = np.diff(s[:, :3], axis=0)[0]
        dr = (R.from_rotvec(s[1, 3:6]) * R.from_rotvec(s[0, 3:6]).inv()).as_rotvec()
        dg = ((s[1, -1] - s[1, -2]) - (s[0, -1] - s[0, -2])) * 12.5  # change in finger gap (up to 0.08 m) scaled to about [-1, 1]

        return np.concatenate([dp, dr, [dg]])


def stratified_split(ds: MixedLeRobotDataset, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    """Episode-level split, identical to the PyTorch implementation (same ``torch.randperm`` stream)."""
    train_idx, val_idx, train_lens = [], [], []
    perms: Dict[str, np.ndarray] = {}
    for i, d in enumerate(ds.datasets):
        repo = ds.repo_ids[i]
        off = ds.cumulative_lengths[i]
        exclude = math.ceil(ds.delta_time * d.meta.fps)
        ep_from, ep_to = d.episode_bounds
        valid = np.maximum(0, ep_to - exclude - ep_from)
        starts = np.cumsum(np.concatenate(([0], valid[:-1])))
        if repo not in perms:
            h = int(hashlib.md5(repo.encode()).hexdigest(), 16) % (2**31)
            perms[repo] = torch_randperm(len(ep_from), seed + h)
        perm = perms[repo]
        n_val = max(1, int(len(perm) * val_ratio))

        def take(eps, out):
            c = 0
            for e in eps:
                if valid[e] > 0:
                    s = off + starts[e]
                    out.extend(range(int(s), int(s + valid[e])))
                    c += int(valid[e])
            return c

        take(perm[:n_val], val_idx)
        train_lens.append(take(perm[n_val:], train_idx))
    return train_idx, val_idx, train_lens


class Subset:
    def __init__(self, ds, idx):
        self.ds, self.idx = ds, np.asarray(idx, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.ds[int(self.idx[i])]

    def reseed(self, seed):
        self.ds.reseed(seed)


def collate_latent_action(instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch frame pairs (NumPy arrays)."""
    return {
        "videos": np.stack([np.stack([x["initial_pixel_values"], x["target_pixel_values"]]) for x in instances]),
        "action": np.stack([np.asarray(x["action"], np.float32) for x in instances]),
        "cam_calib_mask": np.array([x["cam_calib"] for x in instances], dtype=bool),
        "dataset_names": [x["dataset_name"] for x in instances],
    }

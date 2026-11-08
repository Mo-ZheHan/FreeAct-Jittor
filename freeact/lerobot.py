"""Minimal reader for LeRobot v3.0 datasets (no PyTorch, no ``datasets``).

The dataset format and the frame selection are those of LeRobot
(https://github.com/huggingface/lerobot, Apache-2.0).

Covers what FreeAct reads: per-frame numeric columns from ``data/*/*.parquet``, episode
boundaries and video offsets from ``meta/episodes``, tasks, and frames of
``videos/<key>/*/*.mp4`` decoded with PyAV.  Frame selection follows LeRobot's ``pyav`` backend
(``torchvision.io.VideoReader``): seek to the keyframe before the first requested timestamp,
decode up to the last one and take the closest decoded frame for every query.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


INCOMPLETE_HINT = "the dataset is incomplete; download the full split (see README.md)"


class FrameTimestampError(ValueError):
    pass


class LeRobotMeta:
    def __init__(self, repo_id: str, root: Path):
        import pyarrow.parquet as pq

        self.repo_id = repo_id
        self.root = Path(root)
        with open(self.root / "meta" / "info.json", encoding="utf-8") as f:
            self.info = json.load(f)
        self.fps = self.info["fps"]
        self.features = self.info["features"]
        self.video_keys = [k for k, v in self.features.items() if v["dtype"] == "video"]

        tasks = pq.read_table(self.root / "meta" / "tasks.parquet").to_pydict()
        name_col = next(c for c in tasks if c != "task_index")
        self.tasks = dict(zip(tasks["task_index"], tasks[name_col]))

        files = sorted((self.root / "meta" / "episodes").glob("*/*.parquet"))
        cols = None
        eps: Dict[str, list] = {}
        for p in files:
            t = pq.read_table(p)
            cols = cols or [c for c in t.column_names if not c.startswith("stats/")]
            d = t.select(cols).to_pydict()
            for c in cols:
                eps.setdefault(c, []).extend(d[c])
        order = np.argsort(eps["episode_index"])
        self.episodes = {c: [v[i] for i in order] for c, v in eps.items()}
        self.total_episodes = len(order)

    def episode(self, ep_idx: int) -> Dict:
        return {c: v[ep_idx] for c, v in self.episodes.items()}

    def video_file(self, ep_idx: int, key: str) -> Path:
        ep = self.episode(ep_idx)
        rel = self.info["video_path"].format(
            video_key=key, chunk_index=ep[f"videos/{key}/chunk_index"], file_index=ep[f"videos/{key}/file_index"]
        )
        return self.root / rel


def _column_to_numpy(col) -> np.ndarray:
    """A parquet column as a NumPy array; fixed-length ``list<float>`` columns become ``(rows, n)`` arrays."""
    import pyarrow as pa

    if pa.types.is_list(col.type) or pa.types.is_large_list(col.type) or pa.types.is_fixed_size_list(col.type):
        arr = col.combine_chunks()
        if pa.types.is_fixed_size_list(arr.type):
            n = arr.type.list_size
        else:
            lengths = arr.value_lengths().to_numpy(zero_copy_only=False)
            n = int(lengths[0]) if len(lengths) else 0
            if not (lengths == n).all():  # ragged: keep one array per row
                return np.array([np.asarray(x, np.float32) for x in arr.to_pylist()], dtype=object)
        values = arr.flatten().to_numpy(zero_copy_only=False)
        return values.reshape(len(arr), n)
    return col.to_numpy(zero_copy_only=False)


class LeRobotFrames:
    """Random access to frames of a LeRobot v3.0 dataset.

    ``delta_timestamps`` (``{key: [dt0, dt1, ...]}`` in seconds) selects, for every index, the
    frames / values at ``t + dt`` clamped to the episode (with ``{key}_is_pad`` flags), exactly
    as ``LeRobotDataset``.
    """

    # needed for indexing, timestamps and the task text, whatever else is requested
    CORE_COLUMNS = frozenset({"index", "episode_index", "frame_index", "timestamp", "task_index"})

    def __init__(
        self,
        repo_id: str,
        root: Path,
        delta_timestamps: Optional[Dict[str, Sequence[float]]] = None,
        image_transforms=None,
        tolerance_s: float = 1e-4,
        columns: Optional[Sequence[str]] = None,
    ):
        import pyarrow.parquet as pq

        self.meta = LeRobotMeta(repo_id, root)
        self.root = self.meta.root
        self.image_transforms = image_transforms
        self.tolerance_s = tolerance_s
        fps = self.meta.fps
        self.delta_indices = (
            {k: [round(d * fps) for d in v] for k, v in delta_timestamps.items()} if delta_timestamps else None
        )

        # Non-video columns, concatenated in global index order.  ``columns`` restricts them to the ones the
        # caller reads: DROID carries a dozen large ``list<float>`` columns (joint positions, raw actions,
        # ...) that would otherwise cost tens of GB per process.
        files = sorted((self.root / "data").glob("*/*.parquet"))
        if not files:
            raise FileNotFoundError(f"{self.root}: no data/*/*.parquet files; {INCOMPLETE_HINT}")
        keep = None if columns is None else set(columns) | self.CORE_COLUMNS
        cols: Dict[str, list] = {}
        for p in files:
            names = [c for c in pq.read_schema(p).names if c not in self.meta.video_keys and (keep is None or c in keep)]
            t = pq.read_table(p, columns=names)
            for c in names:
                cols.setdefault(c, []).append(_column_to_numpy(t.column(c)))
        self.columns: Dict[str, np.ndarray] = {c: np.concatenate(chunks) for c, chunks in cols.items()}
        order = np.argsort(self.columns["index"], kind="stable")
        if not np.array_equal(order, np.arange(len(order))):
            self.columns = {c: v[order] for c, v in self.columns.items()}
        self._ep_from = np.asarray(self.meta.episodes["dataset_from_index"])
        self._ep_to = np.asarray(self.meta.episodes["dataset_to_index"])
        self._check_complete()

    def _check_complete(self) -> None:
        """Fail early with a clear message when data or video files of a partial download are missing."""
        expected = self.meta.info.get("total_frames")
        if expected is not None and len(self) != expected:
            raise FileNotFoundError(f"{self.root}: data/ holds {len(self):,} of {expected:,} frames; {INCOMPLETE_HINT}")
        keys = self.meta.video_keys
        if self.delta_indices is not None:
            keys = [k for k in keys if k in self.delta_indices]
        eps = self.meta.episodes
        missing = sorted(
            {
                self.root
                / self.meta.info["video_path"].format(video_key=k, chunk_index=c, file_index=f)
                for k in keys
                for c, f in zip(eps[f"videos/{k}/chunk_index"], eps[f"videos/{k}/file_index"])
            }
            - {p for k in keys for p in (self.root / "videos" / k).glob("*/*.mp4")}
        )
        if missing:
            raise FileNotFoundError(f"{self.root}: {len(missing)} video files are missing (e.g. {missing[0]}); {INCOMPLETE_HINT}")

    def __len__(self) -> int:
        return len(self.columns["index"])

    @property
    def episode_bounds(self):
        return self._ep_from, self._ep_to

    def column(self, key: str) -> np.ndarray:
        return self.columns[key]

    # ------------------------------------------------------------------ video
    def _decode(self, path: Path, timestamps: List[float]) -> np.ndarray:
        """``(N, C, H, W)`` float32 in [0, 1] (LeRobot ``pyav`` backend semantics)."""
        import av

        first, last = min(timestamps), max(timestamps)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            # Single-threaded decoding: the frame-threaded AV1 decoder (dav1d) crashes in data-loader
            # processes forked from a Jittor process, and a frame pair needs only a few frames anyway.
            stream.thread_type = "NONE"
            stream.codec_context.thread_count = 1
            offset = int(round(max(first, 0.0) / stream.time_base))
            container.seek(offset, backward=True, any_frame=False, stream=stream)
            frames, pts = [], []
            for frame in container.decode(stream):
                ts = float(frame.pts * frame.time_base)
                frames.append(frame.to_rgb().to_ndarray())
                pts.append(ts)
                if ts >= last:
                    break
        pts = np.asarray(pts)
        q = np.asarray(timestamps)
        dist = np.abs(q[:, None] - pts[None])
        argmin = dist.argmin(1)
        if not (dist[np.arange(len(q)), argmin] < self.tolerance_s).all():
            raise FrameTimestampError(f"query {q} vs loaded {pts} in {path}")
        out = np.stack([frames[i] for i in argmin]).astype(np.float32) / 255.0
        return out.transpose(0, 3, 1, 2)

    # ------------------------------------------------------------------ items
    def __getitem__(self, idx: int) -> Dict:
        idx = int(idx)
        ep_idx = int(self.columns["episode_index"][idx])
        abs_idx = int(self.columns["index"][idx])
        ep_start, ep_end = int(self._ep_from[ep_idx]), int(self._ep_to[ep_idx])
        item = {c: v[idx] for c, v in self.columns.items()}

        query = {}
        if self.delta_indices is not None:
            for key, deltas in self.delta_indices.items():
                query[key] = [max(ep_start, min(ep_end - 1, abs_idx + d)) for d in deltas]
                item[f"{key}_is_pad"] = np.array([(abs_idx + d < ep_start) or (abs_idx + d >= ep_end) for d in deltas])
                if key not in self.meta.video_keys:
                    item[key] = self.columns[key][query[key]]

        ep = self.meta.episode(ep_idx)
        for vkey in self.meta.video_keys:
            if query and vkey not in query:
                continue
            idxs = query.get(vkey, [idx])
            ts = [float(ep[f"videos/{vkey}/from_timestamp"]) + float(self.columns["timestamp"][i]) for i in idxs]
            frames = self._decode(self.meta.video_file(ep_idx, vkey), ts)
            item[vkey] = frames if vkey in query else frames[0]
            if self.image_transforms is not None:
                item[vkey] = self.image_transforms(item[vkey])
        item["task"] = self.meta.tasks[int(item["task_index"])]
        return item

"""3D (M-RoPE) position ids for Qwen3-VL sequences, adapted from the Qwen3-VL finetuning code
(https://github.com/QwenLM/Qwen3-VL, Apache-2.0).

NumPy port of ``get_rope_index_3`` (images only; Qwen3-VL uses timestamps rather than absolute
time position ids for videos, which FreeAct does not feed to the model).
"""

from typing import Optional, Tuple

import numpy as np

IMAGE_TOKEN_ID = 151655
VISION_START_TOKEN_ID = 151652


def get_rope_index_3(
    spatial_merge_size: int,
    input_ids: np.ndarray,
    image_grid_thw: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``input_ids`` (B, L) -> ``position_ids`` (3, B, L) and ``mrope_position_deltas`` (B, 1)."""
    input_ids = np.asarray(input_ids)
    B, L = input_ids.shape
    if image_grid_thw is None:
        pos = np.broadcast_to(np.arange(L)[None, None], (3, B, L)).copy()
        return pos, np.zeros((B, 1), dtype=np.int64)

    image_grid_thw = np.asarray(image_grid_thw)
    position_ids = np.ones((3, B, L), dtype=np.int64)
    deltas = []
    image_index = 0
    for i in range(B):
        ids = input_ids[i]
        vision_start = np.nonzero(ids == VISION_START_TOKEN_ID)[0]
        image_nums = int((ids[vision_start + 1] == IMAGE_TOKEN_ID).sum())
        tokens = ids.tolist()
        chunks, st = [], 0
        for _ in range(image_nums):
            ed = tokens.index(IMAGE_TOKEN_ID, st)
            t, h, w = (int(v) for v in image_grid_thw[image_index])
            image_index += 1
            gt, gh, gw = t, h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st
            st_idx = chunks[-1].max() + 1 if chunks else 0
            chunks.append(np.broadcast_to(np.arange(text_len)[None], (3, text_len)) + st_idx)
            t_index = np.repeat(np.arange(gt), gh * gw)
            h_index = np.tile(np.repeat(np.arange(gh), gw), gt)
            w_index = np.tile(np.arange(gw), gt * gh)
            chunks.append(np.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + gt * gh * gw
        if st < len(tokens):
            st_idx = chunks[-1].max() + 1 if chunks else 0
            text_len = len(tokens) - st
            chunks.append(np.broadcast_to(np.arange(text_len)[None], (3, text_len)) + st_idx)
        pos = np.concatenate(chunks, axis=1)
        position_ids[:, i] = pos
        deltas.append(pos.max() + 1 - L)
    return position_ids, np.asarray(deltas, dtype=np.int64)[:, None]

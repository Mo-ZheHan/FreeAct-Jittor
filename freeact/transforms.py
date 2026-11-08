"""NumPy image transforms equivalent to the torchvision v2 ones used by the PyTorch release.

* :func:`resize_chw` = ``v2.functional.resize(..., antialias=True)`` on float tensors (bilinear,
  antialiased; separable triangle filter as in PyTorch's ``_upsample_bilinear2d_aa``);
* :class:`ResizeCenterCrop` = ``Compose([Resize(s), CenterCrop(s)])``.

All functions take and return ``(..., C, H, W)`` float32 arrays in ``[0, 1]``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image


@lru_cache(maxsize=32)
def _aa_weights(in_size: int, out_size: int) -> np.ndarray:
    """``(out, in)`` interpolation matrix of PyTorch's antialiased bilinear resize."""
    scale = in_size / out_size
    support = max(scale, 1.0)  # bilinear: support 1, widened by the downscale factor
    inv = 1.0 / max(scale, 1.0)
    W = np.zeros((out_size, in_size), dtype=np.float64)
    for i in range(out_size):
        center = scale * (i + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        x = np.arange(xmin, xmax, dtype=np.float64)
        w = np.maximum(0.0, 1.0 - np.abs((x - center + 0.5) * inv))
        total = w.sum()
        if total > 0:
            W[i, xmin:xmax] = w / total
    return W.astype(np.float32)


def resize_chw(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Antialiased bilinear resize of ``(..., C, H, W)`` float arrays."""
    h, w = img.shape[-2:]
    if (h, w) == (out_h, out_w):
        return img.astype(np.float32, copy=True)
    Wh, Ww = _aa_weights(h, out_h), _aa_weights(w, out_w)
    x = np.einsum("oh,...hw->...ow", Wh, img.astype(np.float32))
    return np.einsum("...hw,pw->...hp", x, Ww).astype(np.float32)


def array_to_pil(img: np.ndarray) -> Image.Image:
    """``(C, H, W)`` float in ``[0, 1]`` -> PIL image (values are truncated, as in the PyTorch implementation)."""
    return Image.fromarray(np.clip(img.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))


# ----------------------------------------------------------------------- resize / crop


class ResizeCenterCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[-2:]
        if H <= W:
            nh, nw = self.size, int(self.size * W / H)
        else:
            nh, nw = int(self.size * H / W), self.size
        out = resize_chw(img, nh, nw)
        top, left = int(round((nh - self.size) / 2.0)), int(round((nw - self.size) / 2.0))
        return out[..., top : top + self.size, left : left + self.size]

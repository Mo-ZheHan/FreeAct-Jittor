"""Shared helpers of the latent action tokenizers: bilinear resize with PyTorch's coefficients."""

import math

import numpy as np

POS_GRID = 37  # DINOv2 is trained at 518 px: 518 / 14


def bilinear_matrix(n_in: int, n_out: int) -> np.ndarray:
    """``(n_out, n_in)`` weights of ``F.interpolate(mode="bilinear", align_corners=False)`` along one axis."""
    scale = n_in / n_out
    src = np.maximum((np.arange(n_out) + 0.5) * scale - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    lam = src - i0
    m = np.zeros((n_out, n_in), np.float32)
    rows = np.arange(n_out)
    np.add.at(m, (rows, i0), (1.0 - lam).astype(np.float32))
    np.add.at(m, (rows, i1), lam.astype(np.float32))
    return m


def resize_bilinear(x, size: int):
    """``(B, C, H, W)`` Jittor array -> ``(B, C, size, size)``, as ``F.interpolate(..., mode="bilinear")``."""
    import jittor as jt

    H, W = x.shape[-2:]
    mh = jt.array(bilinear_matrix(H, size)).stop_grad()
    mw = jt.array(bilinear_matrix(W, size)).stop_grad()
    return jt.matmul(jt.matmul(mh, x), mw.transpose(1, 0))


def bicubic_matrix(n_in: int, n_out: int, antialias: bool = True, a: float = -0.5) -> np.ndarray:
    """``(n_out, n_in)`` weights of ``F.interpolate(size=n_out, mode="bicubic", antialias=..., align_corners=False)``.

    With ``antialias`` and ``n_out < n_in`` the cubic kernel is stretched by ``n_in / n_out`` and the
    weights of each output are renormalized (PyTorch's ``_upsample_bicubic2d_aa``, which uses
    ``a = -0.5``); without it the plain 4-tap kernel with ``a = -0.75`` is used.
    """
    scale = n_in / n_out
    m = np.zeros((n_out, n_in), np.float64)
    if not antialias or scale <= 1.0:
        a = -0.75
        near = lambda x: ((a + 2) * x - (a + 3)) * x * x + 1
        far = lambda x: ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
        for i in range(n_out):
            src = scale * (i + 0.5) - 0.5
            f = math.floor(src)
            t = src - f
            for k, c in zip(range(-1, 3), (far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t))):
                m[i, min(max(f + k, 0), n_in - 1)] += c
        return m

    support = 2.0 * scale  # the kernel's radius (2) stretched by the downscale factor
    invscale = 1.0 / scale

    def kernel(x):
        x = abs(x)
        if x < 1.0:
            return ((a + 2) * x - (a + 3)) * x * x + 1
        if x < 2.0:
            return (((x - 5) * x + 8) * x - 4) * a
        return 0.0

    for i in range(n_out):
        center = scale * (i + 0.5)
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), n_in)
        w = np.array([kernel((j + 0.5 - center) * invscale) for j in range(lo, hi)])
        m[i, lo:hi] = w / w.sum()
    return m


def interpolate_pos_embed(pos: np.ndarray, grid: int) -> np.ndarray:
    """``(1, 1 + 37 * 37, D)`` -> ``(1, 1 + grid * grid, D)`` as UniVLA's ``DinoVisionTransformer.interpolate_pos_encoding``.

    UniVLA builds DINOv2 through ``torch.hub`` with ``interpolate_antialias=True`` and
    ``interpolate_offset=0``: a bicubic resize to ``size=(grid, grid)`` with anti-aliasing.
    """
    cls, patch = pos[:, :1], pos[0, 1:].reshape(POS_GRID, POS_GRID, -1)
    m = bicubic_matrix(POS_GRID, grid)
    out = np.einsum("oi,ijd,pj->opd", m, patch.astype(np.float64), m)
    return np.concatenate([cls, out.reshape(1, grid * grid, -1).astype(pos.dtype)], axis=1)

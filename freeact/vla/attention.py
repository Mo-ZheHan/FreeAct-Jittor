"""Variable-length (packed) attention for Jittor.

Jittor has no FlashAttention kernel, so packed sequences are attended to by padding every sample
to the longest one in the batch and running a batched softmax attention with a mask.  FreeAct's
samples are short and of similar length (~160 tokens), so this costs little extra compute.

Numerics follow FlashAttention-2 (which the PyTorch release uses): ``Q K^T`` accumulated in
float32, softmax in float32, probabilities cast to the value dtype before ``P V``.
"""

from functools import lru_cache
from typing import Tuple

import jittor as jt
import numpy as np


@lru_cache(maxsize=64)
def _layout(cu_key: Tuple[int, ...], causal: bool):
    """Gather / scatter indices and the additive mask for a given segmentation."""
    cu = np.asarray(cu_key, dtype=np.int64)
    lens = np.diff(cu)
    S, Lmax = len(lens), int(lens.max())
    gather = np.zeros((S, Lmax), dtype=np.int32)
    valid = np.zeros((S, Lmax), dtype=bool)
    for s, (start, n) in enumerate(zip(cu[:-1], lens)):
        gather[s, :n] = np.arange(start, start + n)
        valid[s, :n] = True
    mask = np.where(valid[:, None, :], 0.0, -np.inf).astype(np.float32).repeat(Lmax, axis=1)  # keys
    if causal:
        mask = mask + np.triu(np.full((Lmax, Lmax), -np.inf, dtype=np.float32), k=1)[None]
    # padded query rows would be all -inf -> NaN; give them one valid key (results are discarded)
    mask[:, :, 0] = np.where(np.isinf(mask[:, :, 0]) & ~valid[:, :, None][:, :, 0], 0.0, mask[:, :, 0])
    flat_pos = (np.arange(S)[:, None] * Lmax + np.arange(Lmax)[None])[valid]  # token -> (s, i)
    order = np.argsort(gather[valid], kind="stable")
    scatter = flat_pos[order].astype(np.int32)
    return gather, mask[:, None], scatter, S, Lmax


def _repeat_kv(x: jt.Var, n_rep: int) -> jt.Var:
    if n_rep == 1:
        return x
    shape = x.shape  # (..., H_kv, L, D)
    x = x.unsqueeze(-3).expand(shape[:-2] + (n_rep,) + shape[-2:])
    return x.reshape(shape[:-3] + (shape[-3] * n_rep,) + shape[-2:])


def flash_attention(q: jt.Var, k: jt.Var, v: jt.Var, cu_seqlens, causal: bool, scale: float) -> jt.Var:
    """Attention over packed sequences.

    ``q``: (H, Lq, D), ``k`` / ``v``: (H_kv, Lk, D) with ``H % H_kv == 0``.  ``cu_seqlens`` are
    the sample boundaries of the packed axis.  ``Lk > Lq`` (cached generation, a single sequence)
    aligns the queries to the end of the keys.
    """
    n_rep = q.shape[0] // k.shape[0]
    Lq, Lk = q.shape[1], k.shape[1]
    out_dtype = v.dtype

    if Lk != Lq:  # incremental decoding: one sequence, queries are the last Lq positions
        kk, vv = _repeat_kv(k, n_rep), _repeat_kv(v, n_rep)
        s = jt.matmul(q.float32(), kk.float32().transpose(0, 2, 1)) * scale
        if causal and Lq > 1:
            m = np.triu(np.full((Lq, Lk), -np.inf, dtype=np.float32), k=Lk - Lq + 1)
            s = s + jt.array(m)
        p = jt.nn.softmax(s, dim=-1).cast(out_dtype)
        return jt.matmul(p, vv)

    cu = tuple(int(c) for c in np.asarray(cu_seqlens).reshape(-1))
    if len(cu) == 2:  # a single sequence: no padding needed
        kk, vv = _repeat_kv(k, n_rep), _repeat_kv(v, n_rep)
        s = jt.matmul(q.float32(), kk.float32().transpose(0, 2, 1)) * scale
        if causal:
            s = s + jt.array(np.triu(np.full((Lq, Lk), -np.inf, dtype=np.float32), k=1))
        return jt.matmul(jt.nn.softmax(s, dim=-1).cast(out_dtype), vv)

    gather, mask, scatter, S, Lmax = _layout(cu, causal)
    g = jt.array(gather.reshape(-1))

    def to_batch(x):  # (H, L, D) -> (S, H, Lmax, D)
        H, _, D = x.shape
        return x[:, g].reshape(H, S, Lmax, D).transpose(1, 0, 2, 3)

    qb, kb, vb = to_batch(q), _repeat_kv(to_batch(k), n_rep), _repeat_kv(to_batch(v), n_rep)
    s = jt.matmul(qb.float32(), kb.float32().transpose(0, 1, 3, 2)) * scale + jt.array(mask)
    o = jt.matmul(jt.nn.softmax(s, dim=-1).cast(out_dtype), vb)  # (S, H, Lmax, D)
    H, D = o.shape[1], o.shape[3]
    o = o.transpose(1, 0, 2, 3).reshape(H, S * Lmax, D)
    return o[:, jt.array(scatter)]

"""Transformer and vector-quantizer building blocks of the latent action model.

Adapted from UniVLA (https://github.com/OpenDriveLab/UniVLA, Apache-2.0), itself derived from Genie.
"""

import math
from typing import Tuple

import jittor as jt
import numpy as np
from jittor import nn


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = np.arange(0, max_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, model_dim, 2, dtype=np.float32) * -(math.log(10000.0) / model_dim))
        pe = np.zeros((max_len, model_dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        # A constant rather than a parameter, so it is not part of the state_dict.
        self._pos_enc = jt.array(pe).stop_grad()

    def execute(self, x: jt.Var) -> jt.Var:
        return x + self._pos_enc[: x.shape[2]].cast(x.dtype)


def attention(q: jt.Var, k: jt.Var, v: jt.Var, scale: float, is_causal: bool = False) -> jt.Var:
    """Softmax attention on ``(B, H, N, D)`` tensors."""
    attn = jt.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    if is_causal:
        L, S = q.shape[-2], k.shape[-2]
        mask = jt.array(np.triu(np.full((L, S), -np.inf, dtype=np.float32), k=1)).stop_grad()
        attn = attn + mask
    return jt.matmul(nn.softmax(attn, dim=-1), v)


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.scale = (model_dim // num_heads) ** -0.5
        self.heads = num_heads
        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Dropout(dropout))

    def execute(self, x: jt.Var, is_causal: bool = False) -> jt.Var:
        B, N, C = x.shape
        h = self.heads
        q, k, v = (f(x).reshape(B, N, h, C // h).transpose(0, 2, 1, 3) for f in (self.to_q, self.to_k, self.to_v))
        out = attention(q, k, v, self.scale, is_causal=is_causal)
        return self.to_out(out.transpose(0, 2, 1, 3).reshape(B, N, C))


def _ffn(model_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(model_dim, model_dim * 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(model_dim * 4, model_dim),
    )


class SpatioTemporalBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.temporal_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = _ffn(model_dim, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def execute(self, x: jt.Var, causal_temporal: bool = False) -> jt.Var:
        B, T, S, E = x.shape

        # Spatial attention
        x = x.reshape(B * T, S, E)
        x = x + self.spatial_attn(self.norm1(x))
        x = x.reshape(B, T, S, E)

        # Temporal attention
        x = x.transpose(0, 2, 1, 3).reshape(B * S, T, E)
        x = x + self.temporal_attn(self.norm2(x), is_causal=causal_temporal)
        x = x.reshape(B, S, T, E).transpose(0, 2, 1, 3)

        # Feedforward
        return x + self.ffn(self.norm3(x))


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        causal_temporal: bool = False,
        to_out: bool = True,
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.LayerNorm(model_dim))
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [SpatioTemporalBlock(model_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.out = nn.Linear(model_dim, out_dim) if to_out else nn.Identity()
        self.causal_temporal = causal_temporal

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.pos_enc(self.ffn(x))
        for block in self.transformer_blocks:
            x = block(x, self.causal_temporal)
        return self.out(x)


class SpatioBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = _ffn(model_dim, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def execute(self, x: jt.Var) -> jt.Var:
        B, T, S, E = x.shape
        x = x.reshape(B * T, S, E)
        x = x + self.spatial_attn(self.norm1(x))
        x = x.reshape(B, T, S, E)
        return x + self.ffn(self.norm2(x))


class SpatioTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.LayerNorm(model_dim))
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList([SpatioBlock(model_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.out = nn.Linear(model_dim, out_dim)

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.pos_enc(self.ffn(x))
        for block in self.transformer_blocks:
            x = block(x)
        return self.out(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_latents: int, latent_dim: int) -> None:
        super().__init__()
        self.codebook = nn.Embedding(num_latents, latent_dim)
        self.codebook.weight.assign(jt.rand(num_latents, latent_dim) * (2.0 / num_latents) - 1.0 / num_latents)
        self.num_latents = num_latents

    def execute(self, x: jt.Var) -> Tuple[jt.Var, jt.Var, jt.Var, jt.Var]:
        # Same formula as torch.cdist for > 25 codes ("use_mm_for_euclid_dist_if_necessary"), so
        # that near-ties are broken identically to the PyTorch implementation.
        w = self.codebook.weight
        x_ = jt.concat([-2 * x, (x * x).sum(-1, keepdims=True), jt.ones(x.shape[:-1] + (1,), x.dtype)], dim=-1)
        w_ = jt.concat([w, jt.ones((w.shape[0], 1), w.dtype), (w * w).sum(-1, keepdims=True)], dim=-1)
        dist = jt.matmul(x_, w_.transpose(1, 0)).maximum(1e-30).sqrt()
        indices, _ = jt.argmin(dist, dim=-1)
        z = self.codebook(indices)
        # Straight-through estimator
        z_q = x + (z - x).stop_grad()
        return z_q, z, x, indices

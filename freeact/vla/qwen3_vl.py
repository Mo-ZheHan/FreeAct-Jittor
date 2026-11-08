"""Qwen3-VL-4B-Instruct in Jittor (vision tower with DeepStack + text decoder with interleaved M-RoPE).

A line-by-line port of ``transformers.models.qwen3_vl.modeling_qwen3_vl`` (transformers 4.57, the
version used by the PyTorch release), restricted to what FreeAct needs: images (no videos),
packed sequences (block-diagonal causal attention over ``cu_seqlens``) and batch-1 cached
generation.  Parameter names follow the Hugging Face checkpoint, so
``Qwen/Qwen3-VL-4B-Instruct`` safetensors load without renaming (``model.`` prefix dropped).

Original code: Copyright 2025 The Qwen Team and The HuggingFace Inc. team, licensed under the
Apache License, Version 2.0 (https://github.com/huggingface/transformers).

Numerics follow transformers: weights in bfloat16, RMSNorm / softmax / RoPE in float32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import jittor as jt
import numpy as np
from jittor import nn

from .attention import flash_attention


@dataclass
class VisionConfig:
    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    out_hidden_size: int = 2560
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    in_channels: int = 3
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: Sequence[int] = (5, 11, 17)


@dataclass
class TextConfig:
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    mrope_section: Sequence[int] = (24, 20, 20)
    vocab_size: int = 151936


@dataclass
class Qwen3VLConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652

    @classmethod
    def from_hf(cls, cfg: dict) -> "Qwen3VLConfig":
        v, t = cfg["vision_config"], cfg["text_config"]
        vision = VisionConfig(**{k: v[k] for k in VisionConfig.__dataclass_fields__ if k in v})
        text_kw = {k: t[k] for k in TextConfig.__dataclass_fields__ if k in t}
        if "rope_scaling" in t:
            text_kw["mrope_section"] = tuple(t["rope_scaling"]["mrope_section"])
        text = TextConfig(**text_kw)
        return cls(vision=vision, text=text, image_token_id=cfg["image_token_id"],
                   video_token_id=cfg["video_token_id"], vision_start_token_id=cfg["vision_start_token_id"])


# --------------------------------------------------------------------------- basic layers


class Linear(nn.Module):
    """``y = x W^T + b`` with a bfloat16 (or float32) weight; LoRA can be attached later."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: str = "bfloat16"):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = jt.zeros((out_features, in_features), dtype)
        self.bias = jt.zeros((out_features,), dtype) if bias else None
        self.lora = None  # set by freeact.vla.lora

    def execute(self, x: jt.Var) -> jt.Var:
        y = jt.matmul(x, self.weight.transpose(1, 0))
        if self.bias is not None:
            y = y + self.bias
        if self.lora is not None:
            y = y + self.lora(x)
        return y


def _row_mean(h: jt.Var) -> jt.Var:
    """Mean over the last axis as a matmul with a constant vector, ``(..., D) -> (..., 1)``.

    Jittor's ``mean`` / ``sum`` reductions accumulate with atomic adds, so their float32 result depends on the
    order in which threads finish and differs from run to run (visible as run-to-run noise of the bfloat16
    forward and of the gradients).  A matmul with a vector of ``1 / n`` accumulates in a fixed order.
    """
    n = h.shape[-1]
    return jt.matmul(h, jt.full((n, 1), 1.0 / n, h.dtype))


def _row_broadcast(r: jt.Var, n: int) -> jt.Var:
    """``(..., 1) -> (..., n)`` as an outer product with ones.

    The backward of an ordinary broadcast sums the gradient over the broadcast axis, which is again an atomic-add
    reduction; the backward of this matmul is a matmul.
    """
    return jt.matmul(r, jt.ones((1, n), r.dtype))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = jt.ones((dim,), "bfloat16")
        self.eps = eps

    def execute(self, x: jt.Var) -> jt.Var:
        dtype = x.dtype
        h = x.float32()
        h = h * _row_broadcast(jt.rsqrt(_row_mean(h * h) + self.eps), h.shape[-1])
        return self.weight * h.cast(dtype)


class LayerNorm(nn.Module):
    """LayerNorm with bfloat16 parameters; statistics in float32 (as PyTorch's fused kernel)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = jt.ones((dim,), "bfloat16")
        self.bias = jt.zeros((dim,), "bfloat16")
        self.eps = eps

    def execute(self, x: jt.Var) -> jt.Var:
        dtype = x.dtype
        h = x.float32()
        n = h.shape[-1]
        h = h - _row_broadcast(_row_mean(h), n)
        h = h * _row_broadcast(jt.rsqrt(_row_mean(h * h) + self.eps), n)
        return (h * self.weight.float32() + self.bias.float32()).cast(dtype)


class Embedding(nn.Module):
    def __init__(self, num: int, dim: int):
        super().__init__()
        self.weight = jt.zeros((num, dim), "bfloat16")

    def execute(self, ids: jt.Var) -> jt.Var:
        return self.weight[ids]


def gelu_tanh(x: jt.Var) -> jt.Var:
    dtype = x.dtype
    x = x.float32()
    y = 0.5 * x * (1.0 + jt.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    return y.cast(dtype)


def gelu_erf(x: jt.Var) -> jt.Var:
    dtype = x.dtype
    x = x.float32()
    return (0.5 * x * (1.0 + jt.erf(x / math.sqrt(2.0)))).cast(dtype)


def silu(x: jt.Var) -> jt.Var:
    dtype = x.dtype
    x = x.float32()
    return (x * jt.sigmoid(x)).cast(dtype)


def rotate_half(x: jt.Var) -> jt.Var:
    d = x.shape[-1] // 2
    return jt.concat([-x[..., d:], x[..., :d]], dim=-1)


# --------------------------------------------------------------------------- vision tower


class VisionMLP(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.linear_fc1 = Linear(c.hidden_size, c.intermediate_size)
        self.linear_fc2 = Linear(c.intermediate_size, c.hidden_size)

    def execute(self, x):
        return self.linear_fc2(gelu_tanh(self.linear_fc1(x)))


class VisionPatchEmbed(nn.Module):
    """Conv3d with kernel == stride == (2, 16, 16) is a linear map of each flattened patch."""

    def __init__(self, c: VisionConfig):
        super().__init__()
        self.c = c
        k = c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size
        self.proj = Linear(k, c.hidden_size)  # weight stored as (out, in, t, p, p) in the checkpoint

    def execute(self, x):
        return self.proj(x.reshape(-1, self.proj.in_features).cast(self.proj.weight.dtype))


class VisionPatchMerger(nn.Module):
    def __init__(self, c: VisionConfig, use_postshuffle_norm: bool = False):
        super().__init__()
        self.hidden_size = c.hidden_size * c.spatial_merge_size**2
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = LayerNorm(self.hidden_size if use_postshuffle_norm else c.hidden_size, eps=1e-6)
        self.linear_fc1 = Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = Linear(self.hidden_size, c.out_hidden_size)

    def execute(self, x):
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.hidden_size))
        else:
            x = self.norm(x).reshape(-1, self.hidden_size)
        return self.linear_fc2(gelu_erf(self.linear_fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.num_heads = c.num_heads
        self.head_dim = c.hidden_size // c.num_heads
        self.qkv = Linear(c.hidden_size, c.hidden_size * 3)
        self.proj = Linear(c.hidden_size, c.hidden_size)

    def execute(self, x, cu_seqlens: np.ndarray, cos, sin):
        n = x.shape[0]
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim).transpose(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (N, H, D)
        dtype = q.dtype
        c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)
        q = (q.float32() * c + rotate_half(q.float32()) * s).cast(dtype)
        k = (k.float32() * c + rotate_half(k.float32()) * s).cast(dtype)
        out = flash_attention(
            q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2), cu_seqlens, causal=False,
            scale=self.head_dim**-0.5,
        )  # (H, N, D)
        return self.proj(out.transpose(1, 0, 2).reshape(n, -1))


class VisionBlock(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.norm1 = LayerNorm(c.hidden_size, eps=1e-6)
        self.norm2 = LayerNorm(c.hidden_size, eps=1e-6)
        self.attn = VisionAttention(c)
        self.mlp = VisionMLP(c)

    def execute(self, x, cu_seqlens, cos, sin):
        x = x + self.attn(self.norm1(x), cu_seqlens, cos, sin)
        return x + self.mlp(self.norm2(x))


class VisionModel(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.c = c
        self.patch_embed = VisionPatchEmbed(c)
        self.pos_embed = Embedding(c.num_position_embeddings, c.hidden_size)
        self.num_grid_per_side = int(c.num_position_embeddings**0.5)
        head_dim = c.hidden_size // c.num_heads
        self._inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim // 2, 2, dtype=np.float32) / (head_dim // 2)))
        self.blocks = nn.ModuleList([VisionBlock(c) for _ in range(c.depth)])
        self.merger = VisionPatchMerger(c, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = list(c.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [VisionPatchMerger(c, use_postshuffle_norm=True) for _ in self.deepstack_visual_indexes]
        )

    # Position tables are computed on the host (they only depend on the grid sizes)
    def _rot_pos_emb(self, grid_thw: np.ndarray) -> np.ndarray:
        m = self.c.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max())
        freq_table = np.outer(np.arange(max_hw, dtype=np.float32), self._inv_freq).astype(np.float32)
        pos = []
        for t, h, w in grid_thw:
            mh, mw = h // m, w // m
            row = (np.arange(mh)[:, None, None, None] * m + np.arange(m)[None, None, :, None])
            col = (np.arange(mw)[None, :, None, None] * m + np.arange(m)[None, None, None, :])
            row = np.broadcast_to(row, (mh, mw, m, m)).reshape(-1)
            col = np.broadcast_to(col, (mh, mw, m, m)).reshape(-1)
            coords = np.stack([row, col], axis=-1)
            pos.append(np.tile(coords, (t, 1)))
        pos = np.concatenate(pos)
        return freq_table[pos].reshape(len(pos), -1)

    def _pos_embed_interp(self, grid_thw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Bilinear interpolation indices / weights into the 48x48 absolute position table."""
        n = self.num_grid_per_side
        m = self.c.spatial_merge_size
        idx_all, w_all = [], []
        for t, h, w in grid_thw:
            hi = np.linspace(0, n - 1, h, dtype=np.float32)
            wi = np.linspace(0, n - 1, w, dtype=np.float32)
            hf, wf = hi.astype(np.int64), wi.astype(np.int64)
            hc, wc = np.minimum(hf + 1, n - 1), np.minimum(wf + 1, n - 1)
            dh, dw = hi - hf, wi - wf
            idx = np.stack([
                (hf[:, None] * n + wf[None]).reshape(-1),
                (hf[:, None] * n + wc[None]).reshape(-1),
                (hc[:, None] * n + wf[None]).reshape(-1),
                (hc[:, None] * n + wc[None]).reshape(-1),
            ])
            wts = np.stack([
                ((1 - dh)[:, None] * (1 - dw)[None]).reshape(-1),
                ((1 - dh)[:, None] * dw[None]).reshape(-1),
                (dh[:, None] * (1 - dw)[None]).reshape(-1),
                (dh[:, None] * dw[None]).reshape(-1),
            ]).astype(np.float32)
            # (4, h*w) -> repeat over t, then reorder into merge blocks
            perm = (
                np.arange(h * w).reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).reshape(-1)
            )
            idx_all.append(np.tile(idx[:, perm], (1, t)))
            w_all.append(np.tile(wts[:, perm], (1, t)))
        return np.concatenate(idx_all, axis=1), np.concatenate(w_all, axis=1)

    def execute(self, pixel_values: jt.Var, grid_thw: np.ndarray):
        grid_thw = np.asarray(grid_thw, dtype=np.int64)
        x = self.patch_embed(pixel_values)

        idx, wts = self._pos_embed_interp(grid_thw)
        # transformers multiplies in the weight dtype (bfloat16) and sums the 4 terms in bfloat16
        pe = self.pos_embed(jt.array(idx)) * jt.array(wts).bfloat16().unsqueeze(-1)
        x = x + (((pe[0] + pe[1]) + pe[2]) + pe[3])

        rot = self._rot_pos_emb(grid_thw)
        emb = np.concatenate([rot, rot], axis=-1)
        cos, sin = jt.array(np.cos(emb)).stop_grad(), jt.array(np.sin(emb)).stop_grad()
        cu = np.concatenate([[0], np.cumsum(np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]))]).astype(np.int32)

        deepstack = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, cu, cos, sin)
            if i in self.deepstack_visual_indexes:
                deepstack.append(self.deepstack_merger_list[self.deepstack_visual_indexes.index(i)](x))
        return self.merger(x), deepstack


# --------------------------------------------------------------------------- text decoder


class TextAttention(nn.Module):
    def __init__(self, c: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads, self.num_kv_heads, self.head_dim = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.q_proj = Linear(c.hidden_size, c.num_attention_heads * c.head_dim, bias=False)
        self.k_proj = Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False)
        self.v_proj = Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False)
        self.o_proj = Linear(c.num_attention_heads * c.head_dim, c.hidden_size, bias=False)
        self.q_norm = RMSNorm(c.head_dim, eps=c.rms_norm_eps)
        self.k_norm = RMSNorm(c.head_dim, eps=c.rms_norm_eps)

    def execute(self, x, cos, sin, cu_seqlens, cache: Optional["KVCache"] = None):
        """``x``: (L, hidden) packed tokens; returns (L, hidden)."""
        L = x.shape[0]
        q = self.q_norm(self.q_proj(x).reshape(L, self.num_heads, self.head_dim)).transpose(1, 0, 2)
        k = self.k_norm(self.k_proj(x).reshape(L, self.num_kv_heads, self.head_dim)).transpose(1, 0, 2)
        v = self.v_proj(x).reshape(L, self.num_kv_heads, self.head_dim).transpose(1, 0, 2)
        c, s = cos.unsqueeze(0), sin.unsqueeze(0)  # (1, L, D)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)
        out = flash_attention(q, k, v, cu_seqlens, causal=True, scale=self.head_dim**-0.5)  # (H, L, D)
        return self.o_proj(out.transpose(1, 0, 2).reshape(L, -1))


class TextMLP(nn.Module):
    def __init__(self, c: TextConfig):
        super().__init__()
        self.gate_proj = Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = Linear(c.intermediate_size, c.hidden_size, bias=False)

    def execute(self, x):
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))


class TextDecoderLayer(nn.Module):
    def __init__(self, c: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = TextAttention(c, layer_idx)
        self.mlp = TextMLP(c)
        self.input_layernorm = RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, eps=c.rms_norm_eps)

    def execute(self, x, cos, sin, cu_seqlens, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cu_seqlens, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class KVCache:
    """Per-layer key / value cache for batch-1 generation (``(H_kv, T, D)`` per layer)."""

    def __init__(self, num_layers: int):
        self.k: List[Optional[jt.Var]] = [None] * num_layers
        self.v: List[Optional[jt.Var]] = [None] * num_layers

    @property
    def length(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[1]

    def update(self, layer: int, k: jt.Var, v: jt.Var):
        if self.k[layer] is not None:
            k = jt.concat([self.k[layer], k], dim=1)
            v = jt.concat([self.v[layer], v], dim=1)
        self.k[layer], self.v[layer] = k, v
        return k, v


def mrope_cos_sin(position_ids: np.ndarray, c: TextConfig, dtype: str) -> Tuple[jt.Var, jt.Var]:
    """Interleaved M-RoPE tables for ``position_ids`` of shape (3, L) -> two (L, head_dim) Vars."""
    inv_freq = 1.0 / (c.rope_theta ** (np.arange(0, c.head_dim, 2, dtype=np.int64).astype(np.float32) / c.head_dim))
    inv_freq = inv_freq.astype(np.float32)
    freqs = position_ids.astype(np.float32)[:, :, None] * inv_freq[None, None, :]  # (3, L, D/2)
    f = freqs[0].copy()
    for dim, offset in ((1, 1), (2, 2)):
        length = c.mrope_section[dim] * 3
        f[..., offset:length:3] = freqs[dim, ..., offset:length:3]
    emb = np.concatenate([f, f], axis=-1)
    return jt.array(np.cos(emb)).cast(dtype).stop_grad(), jt.array(np.sin(emb)).cast(dtype).stop_grad()


class TextModel(nn.Module):
    def __init__(self, c: TextConfig):
        super().__init__()
        self.c = c
        self.embed_tokens = Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([TextDecoderLayer(c, i) for i in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size, eps=c.rms_norm_eps)

    def execute(
        self,
        inputs_embeds: jt.Var,
        position_ids: np.ndarray,
        cu_seqlens: np.ndarray,
        visual_pos: Optional[jt.Var] = None,
        deepstack: Optional[List[jt.Var]] = None,
        cache: Optional[KVCache] = None,
        output_hidden_states: bool = False,
    ):
        """Returns ``(normed final states, pre-norm last-layer states, hidden_states)``.

        As in ``Qwen3VLForConditionalGeneration(output_hidden_states=True)`` (transformers 4.57),
        ``hidden_states`` holds the embeddings and every decoder layer's output (including the
        DeepStack addition, which transformers applies in place); its last entry is *not*
        normalized.  The action head consumes these pre-norm last-layer states.
        """
        cos, sin = mrope_cos_sin(position_ids, self.c, inputs_embeds.dtype)
        x = inputs_embeds
        hidden_states = [x] if output_hidden_states else None
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cu_seqlens, cache)
            if deepstack is not None and i < len(deepstack):
                x = add_at_rows(x, visual_pos, deepstack[i].cast(x.dtype))
            if output_hidden_states:
                hidden_states.append(x)
        return self.norm(x), x, hidden_states


def add_at_rows(x: jt.Var, rows: jt.Var, values: jt.Var) -> jt.Var:
    """``x[rows] += values`` (DeepStack), differentiable w.r.t. both."""
    pad = jt.concat([jt.zeros((1, x.shape[1]), values.dtype), values], dim=0)
    return x + pad[_row_map(x.shape[0], rows)]


def _row_map(n: int, rows: jt.Var) -> jt.Var:
    """Index into ``concat([0-row, values])``: ``k + 1`` at ``rows[k]``, 0 elsewhere."""
    m = np.zeros(n, dtype=np.int32)
    m[rows.numpy()] = np.arange(1, rows.shape[0] + 1, dtype=np.int32)
    return jt.array(m)


# --------------------------------------------------------------------------- full model


class Qwen3VLForConditionalGeneration(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.visual = VisionModel(config.vision)
        self.language_model = TextModel(config.text)
        self.lm_head = Linear(config.text.hidden_size, config.text.vocab_size, bias=False)

    # ---------------------------------------------------------------- inputs
    def embed(self, input_ids: np.ndarray, pixel_values: Optional[jt.Var], image_grid_thw: Optional[np.ndarray]):
        """Token + image embeddings for a packed (1, L) sequence."""
        ids = np.asarray(input_ids).reshape(-1)
        x = self.language_model.embed_tokens(jt.array(ids.astype(np.int32)))
        visual_pos, deepstack = None, None
        if pixel_values is not None:
            image_embeds, deepstack = self.visual(pixel_values, image_grid_thw)
            pos = np.nonzero(ids == self.config.image_token_id)[0]
            if len(pos) != image_embeds.shape[0]:
                raise ValueError(f"Image features and image tokens do not match: {len(pos)} vs {image_embeds.shape[0]}")
            visual_pos = jt.array(pos.astype(np.int32))
            x = replace_rows(x, visual_pos, image_embeds.cast(x.dtype))
        return x, visual_pos, deepstack

    def execute(
        self,
        input_ids: np.ndarray,
        position_ids: np.ndarray,
        cu_seqlens: np.ndarray,
        pixel_values: Optional[jt.Var] = None,
        image_grid_thw: Optional[np.ndarray] = None,
        output_hidden_states: bool = False,
        logits_rows: Optional[jt.Var] = None,
    ):
        """Packed forward. ``position_ids``: (3, L); ``cu_seqlens``: sample boundaries.

        Returns ``(logits, last pre-norm states, hidden_states)``; ``logits_rows`` restricts the
        LM head to these rows.
        """
        x, visual_pos, deepstack = self.embed(input_ids, pixel_values, image_grid_thw)
        h, last, hidden_states = self.language_model(
            x, np.asarray(position_ids).reshape(3, -1), np.asarray(cu_seqlens, dtype=np.int32),
            visual_pos, deepstack, output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(h if logits_rows is None else h[logits_rows])
        return logits, last, hidden_states


def replace_rows(x: jt.Var, rows: jt.Var, values: jt.Var) -> jt.Var:
    """``x[rows] = values`` (``masked_scatter`` in transformers), differentiable w.r.t. both."""
    m = _row_map(x.shape[0], rows).numpy()
    src = np.where(m > 0, x.shape[0] + m - 1, np.arange(x.shape[0])).astype(np.int32)
    return jt.concat([x, values], dim=0)[jt.array(src)]

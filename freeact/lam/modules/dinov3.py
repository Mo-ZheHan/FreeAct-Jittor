"""DINOv3 ViT-B/16 (``vit_base_patch16_dinov3.lvd1689m``), frozen feature extractor of the LAM.

Re-implements the timm ``Eva`` forward (https://github.com/huggingface/pytorch-image-models, Apache-2.0)
used by the PyTorch release (``forward_features``):
patch embedding, CLS + 4 register tokens, 2D RoPE on the patch tokens (DINOv3 variant, half
layout, temperature 100), 12 pre-norm blocks with layer scale.  Parameter names follow timm, so
the weights stored under ``lam.dino_encoder.*`` in ``lam.ckpt`` load unchanged.  The DINOv3 weights
are released by Meta under the DINOv3 License (see THIRD_PARTY_NOTICES.md).
"""

import math

import jittor as jt
import numpy as np
from jittor import nn

EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
PATCH_SIZE = 16
NUM_REG_TOKENS = 4
ROPE_TEMPERATURE = 100.0


def dinov3_rope(height: int, width: int, head_dim: int = EMBED_DIM // NUM_HEADS) -> np.ndarray:
    """``(H*W, 2*head_dim)`` = [sin, cos], matching ``timm.layers.RotaryEmbeddingDinoV3``."""
    dim = head_dim // 4
    periods = ROPE_TEMPERATURE ** (2.0 * np.arange(dim, dtype=np.float32) / (head_dim // 2))
    ch = (np.arange(0.5, height, dtype=np.float32) / np.float32(height)).astype(np.float32)
    cw = (np.arange(0.5, width, dtype=np.float32) / np.float32(width)).astype(np.float32)
    coords = np.stack(np.meshgrid(ch, cw, indexing="ij"), axis=-1).reshape(-1, 2)
    coords = (2.0 * coords - 1.0).astype(np.float32)
    angles = (np.float32(2 * math.pi) * coords[:, :, None] / periods[None, None, :]).astype(np.float32)
    angles = np.tile(angles.reshape(angles.shape[0], -1), 2)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1).astype(np.float32)


def _rotate_half(x: jt.Var) -> jt.Var:
    d = x.shape[-1] // 2
    return jt.concat([-x[..., d:], x[..., :d]], dim=-1)


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, EMBED_DIM, PATCH_SIZE, stride=PATCH_SIZE)

    def execute(self, x):
        x = self.proj(x)  # (B, C, H, W)
        return x.transpose(0, 2, 3, 1)  # NHWC


class Attention(nn.Module):
    def __init__(self, num_prefix_tokens: int):
        super().__init__()
        self.qkv = nn.Linear(EMBED_DIM, EMBED_DIM * 3, bias=False)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.num_prefix_tokens = num_prefix_tokens
        self.scale = (EMBED_DIM // NUM_HEADS) ** -0.5

    def execute(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, NUM_HEADS, C // NUM_HEADS).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)
        sin, cos = rope[:, : rope.shape[1] // 2], rope[:, rope.shape[1] // 2 :]
        npt = self.num_prefix_tokens

        def _rope(t):
            p = t[:, :, npt:]
            return jt.concat([t[:, :, :npt], p * cos + _rotate_half(p) * sin], dim=2)

        q, k = _rope(q), _rope(k)
        # DINOv3's q / k reach |q|, |k| ~ 3e2, so float16 logits overflow to inf: form them in float32
        logits = jt.matmul(q.float32() * self.scale, k.float32().transpose(0, 1, 3, 2))
        attn = nn.softmax(logits, dim=-1).cast(v.dtype)
        x = jt.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(EMBED_DIM, EMBED_DIM * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(EMBED_DIM * 4, EMBED_DIM)

    def execute(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, num_prefix_tokens: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM, eps=1e-5)
        self.attn = Attention(num_prefix_tokens)
        self.norm2 = nn.LayerNorm(EMBED_DIM, eps=1e-5)
        self.mlp = Mlp()
        self.gamma_1 = jt.ones(EMBED_DIM) * 1e-5
        self.gamma_2 = jt.ones(EMBED_DIM) * 1e-5

    def execute(self, x, rope):
        x = x + self.gamma_1 * self.attn(self.norm1(x), rope)
        return x + self.gamma_2 * self.mlp(self.norm2(x))


class DinoV3ViT(nn.Module):
    """``forward_features`` of timm's DINOv3 ViT-B/16 (CLS + 4 registers + patch tokens)."""

    num_prefix_tokens = 1 + NUM_REG_TOKENS

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = jt.zeros((1, 1, EMBED_DIM))
        self.reg_token = jt.zeros((1, NUM_REG_TOKENS, EMBED_DIM))
        self.blocks = nn.ModuleList([Block(self.num_prefix_tokens) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(EMBED_DIM, eps=1e-5)
        self._rope_cache = {}

    def _rope(self, h: int, w: int) -> jt.Var:
        if (h, w) not in self._rope_cache:
            self._rope_cache[(h, w)] = jt.array(dinov3_rope(h, w)).stop_grad()
        return self._rope_cache[(h, w)]

    def forward_features(self, x: jt.Var) -> jt.Var:
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = jt.concat([self.cls_token.expand(B, 1, C), self.reg_token.expand(B, NUM_REG_TOKENS, C), x], dim=1)
        rope = self._rope(H, W).cast(x.dtype)  # q / k / v stay in the encoder's dtype
        for blk in self.blocks:
            x = blk(x, rope)
        return self.norm(x)

    def execute(self, x: jt.Var) -> jt.Var:
        return self.forward_features(x)

"""DINOv2 ViT-B/14 with four register tokens (``dinov2_vitb14_reg``), the frozen backbone of the UniVLA LAM.

Re-implements ``forward_features`` of ``facebookresearch/dinov2`` (Copyright (c) Meta Platforms, Inc. and
affiliates, Apache-2.0) for a fixed input resolution.
Parameter names follow the original (``torch.hub``, un-chunked blocks), so the weights stored in
UniVLA's ``lam-stage-2.ckpt`` under ``dino_encoder.*`` load unchanged.  The positional embedding is
trained at 518 px (37 x 37); the original resizes it bicubically on every forward pass, here it is
resized once with the same coefficients.  UniVLA builds the backbone with ``interpolate_antialias=True``
and ``interpolate_offset=0`` (not DINOv2's defaults), see :func:`freeact.tokenizers.ops.interpolate_pos_embed`.
"""

import jittor as jt
from jittor import nn

from .ops import POS_GRID, interpolate_pos_embed

EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
PATCH_SIZE = 14
NUM_REGISTERS = 4


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, EMBED_DIM, PATCH_SIZE, stride=PATCH_SIZE)

    def execute(self, x):
        x = self.proj(x)  # (B, D, h, w)
        B, D, h, w = x.shape
        return x.reshape(B, D, h * w).transpose(0, 2, 1)


class LayerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = jt.ones(EMBED_DIM)

    def execute(self, x):
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMBED_DIM, EMBED_DIM * 3)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.scale = (EMBED_DIM // NUM_HEADS) ** -0.5

    def execute(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, NUM_HEADS, C // NUM_HEADS).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = nn.softmax(jt.matmul(q * self.scale, k.transpose(0, 1, 3, 2)), dim=-1)
        return self.proj(jt.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(EMBED_DIM, EMBED_DIM * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(EMBED_DIM * 4, EMBED_DIM)

    def execute(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM, eps=1e-6)
        self.attn = Attention()
        self.ls1 = LayerScale()
        self.norm2 = nn.LayerNorm(EMBED_DIM, eps=1e-6)
        self.mlp = Mlp()
        self.ls2 = LayerScale()

    def execute(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoV2ViT(nn.Module):
    """``forward_features(x)["x_norm_patchtokens"]`` for ``(B, 3, S, S)`` inputs with ``S`` a multiple of 14."""

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = jt.zeros((1, 1, EMBED_DIM))
        self.pos_embed = jt.zeros((1, 1 + POS_GRID * POS_GRID, EMBED_DIM))
        self.register_tokens = jt.zeros((1, NUM_REGISTERS, EMBED_DIM))
        self.mask_token = jt.zeros((1, EMBED_DIM))  # unused (masked modeling), part of the checkpoint
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(EMBED_DIM, eps=1e-6)
        self._pos_cache = {}

    def _pos(self, grid: int) -> jt.Var:
        key = (grid, id(self.pos_embed))
        if key not in self._pos_cache:
            self._pos_cache = {key: jt.array(interpolate_pos_embed(self.pos_embed.numpy(), grid)).stop_grad()}
        return self._pos_cache[key]

    def prepare_tokens(self, x: jt.Var) -> jt.Var:
        """Patch embedding, CLS token, positional embedding, registers (``prepare_tokens_with_masks``)."""
        B, _, H, W = x.shape
        assert H == W and H % PATCH_SIZE == 0, x.shape
        x = self.patch_embed(x)
        x = jt.concat([self.cls_token.expand(B, 1, EMBED_DIM), x], dim=1)
        x = x + self._pos(H // PATCH_SIZE)
        return jt.concat([x[:, :1], self.register_tokens.expand(B, NUM_REGISTERS, EMBED_DIM), x[:, 1:]], dim=1)

    def forward_features(self, x: jt.Var) -> jt.Var:
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def patch_tokens(self, x: jt.Var) -> jt.Var:
        return self.forward_features(x)[:, 1 + NUM_REGISTERS:]

    def execute(self, x):
        return self.patch_tokens(x)

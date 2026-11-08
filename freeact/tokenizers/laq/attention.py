"""Building blocks of LAPA's latent action quantization model (LAQ): attention, PEG, feed-forward.

Port of ``laq/attention.py`` of LAPA (https://github.com/LatentActionPretraining/LAPA, MIT).
Parameter names are those of the PyTorch module, except that the four entries of a transformer
layer (``layers.i.0`` ... ``layers.i.3``) are the attributes ``peg``, ``attn``, ``cross``, ``ff``;
:func:`remap_key` translates between the two.

Two quirks of the original are reproduced on purpose, since the released weights were trained with
them: the keys / values of a self-attention come from the *un-normalized* input (only the queries
see the layer norm), and :class:`PEG` reshapes ``(b h w) t d`` activations to ``(b t h w) d``
without transposing (the temporal transformer).
"""

import re

import jittor as jt
from jittor import nn


def remap_key(key: str) -> str:
    """PyTorch ``layers.3.1.`` -> ``layers.3.attn.`` (``0 peg, 1 attn, 2 cross, 3 ff``)."""
    return re.sub(r"layers\.(\d+)\.([0-3])\.", lambda m: f"layers.{m.group(1)}.{('peg', 'attn', 'cross', 'ff')[int(m.group(2))]}.", key)


class LayerNorm(nn.Module):
    """Layer norm with a learned ``gamma`` and a ``beta`` that is a zero buffer in the original (kept: it is in the checkpoint)."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = jt.ones(dim)
        self.beta = jt.zeros(dim)

    def execute(self, x):
        mean = x.mean(-1, keepdims=True)
        var = ((x - mean) ** 2).mean(-1, keepdims=True)
        return (x - mean) / jt.sqrt(var + 1e-5) * self.gamma + self.beta


def l2norm(t):
    return t / jt.maximum(jt.sqrt((t * t).sum(-1, keepdims=True)), 1e-12)


class GEGLU(nn.Module):
    def execute(self, x):
        d = x.shape[-1] // 2
        return nn.gelu(x[..., d:]) * x[..., :d]


def feed_forward(dim: int, mult: int = 4) -> nn.Sequential:
    inner = int(mult * (2 / 3) * dim)
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, inner * 2, bias=False), GEGLU(), nn.Identity(),
                         nn.Linear(inner, dim, bias=False))


class PEG(nn.Module):
    """Depthwise 3x3x3 convolution (causal in time) as a position generator.

    ``shape`` is the *original* ``(b, t, h, w)`` of the video, whatever the layout of ``x``: the
    original reshapes ``x`` to that shape without transposing, so inside the temporal transformer the
    spatial and temporal axes are swapped (and ``h`` / ``w`` are applied to the frames and one patch
    axis).  Reproduced as-is: the released weights were trained with this behavior.
    """

    def __init__(self, dim: int, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.dsconv = _DepthwiseConv3d(dim)

    def execute(self, x, shape=None):
        needs_shape = x.ndim == 3
        orig_shape = x.shape
        if needs_shape:
            x = x.reshape(tuple(shape) + (-1,))
        x = self.dsconv(x, self.causal)
        return x.reshape(orig_shape)


class _DepthwiseConv3d(nn.Module):
    """``nn.Conv3d(dim, dim, 3, groups=dim)`` on channel-last ``(b, t, h, w, d)`` input, zero padded."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = jt.zeros((dim, 1, 3, 3, 3))
        self.bias = jt.zeros(dim)

    def execute(self, x, causal: bool):
        b, t, h, w, d = x.shape
        before, after = (2, 0) if causal else (1, 1)
        x = jt.concat([jt.zeros((b, before, h, w, d), x.dtype), x, jt.zeros((b, after, h, w, d), x.dtype)], dim=1)
        x = jt.concat([jt.zeros((b, t + 2, 1, w, d), x.dtype), x, jt.zeros((b, t + 2, 1, w, d), x.dtype)], dim=2)
        x = jt.concat([jt.zeros((b, t + 2, h + 2, 1, d), x.dtype), x, jt.zeros((b, t + 2, h + 2, 1, d), x.dtype)], dim=3)
        out = self.bias
        for kt in range(3):
            for kh in range(3):
                for kw in range(3):
                    out = out + x[:, kt: kt + t, kh: kh + h, kw: kw + w, :] * self.weight[:, 0, kt, kh, kw]
        return out


class Attention(nn.Module):
    def __init__(self, dim: int, dim_context: int = None, dim_head: int = 64, heads: int = 8,
                 num_null_kv: int = 0, norm_context: bool = True, scale: float = 8.0):
        super().__init__()
        self.heads, self.scale, self.num_null_kv = heads, scale, num_null_kv
        inner = dim_head * heads
        dim_context = dim_context or dim
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else None
        if num_null_kv:
            self.null_kv = jt.randn(heads, 2 * num_null_kv, dim_head)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim_context, inner * 2, bias=False)
        self.q_scale = jt.ones(dim_head)
        self.k_scale = jt.ones(dim_head)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def execute(self, x, context=None, attn_bias=None):
        B = x.shape[0]
        h = self.heads
        if context is not None:
            context = self.context_norm(context)
        kv_input = context if context is not None else x  # NB: before the layer norm, as in the original
        x = self.norm(x)
        q = self.to_q(x)
        k, v = jt.chunk(self.to_kv(kv_input), 2, dim=-1)
        split = lambda t: t.reshape(B, t.shape[1], h, -1).transpose(0, 2, 1, 3)
        q, k, v = split(q), split(k), split(v)
        if self.num_null_kv:
            nk = self.null_kv[:, 0::2].unsqueeze(0).expand(B, h, self.num_null_kv, q.shape[-1])
            nv = self.null_kv[:, 1::2].unsqueeze(0).expand(B, h, self.num_null_kv, q.shape[-1])
            k, v = jt.concat([nk, k], dim=-2), jt.concat([nv, v], dim=-2)
        q, k = l2norm(q) * self.q_scale, l2norm(k) * self.k_scale
        sim = jt.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        if attn_bias is not None:
            if self.num_null_kv:
                attn_bias = jt.concat([jt.zeros(attn_bias.shape[:-1] + (self.num_null_kv,), attn_bias.dtype), attn_bias], dim=-1)
            sim = sim + attn_bias
        out = jt.matmul(nn.softmax(sim, dim=-1), v)
        return self.to_out(out.transpose(0, 2, 1, 3).reshape(B, -1, h * out.shape[-1]))


class ContinuousPositionBias(nn.Module):
    """Continuous relative position bias (Swin v2): an MLP of log-spaced offsets, ``(heads, N, N)``."""

    def __init__(self, dim: int, heads: int, num_dims: int = 2, layers: int = 2):
        super().__init__()
        self.num_dims = num_dims
        self.net = nn.ModuleList([nn.Sequential(nn.Linear(num_dims, dim), nn.LeakyReLU(0.1))])
        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(0.1)))
        self.net.append(nn.Linear(dim, heads))

    def execute(self, *dimensions):
        import numpy as np

        grid = np.stack(np.meshgrid(*[np.arange(d) for d in dimensions], indexing="ij")).reshape(self.num_dims, -1).T
        rel = (grid[:, None, :] - grid[None, :, :]).astype(np.float32)
        rel = np.sign(rel) * np.log(np.abs(rel) + 1)
        x = jt.array(rel.astype(np.float32))
        for layer in self.net:
            x = layer(x)
        return x.transpose(2, 0, 1)  # (heads, i, j)


class Layer(nn.Module):
    def __init__(self, dim, dim_head, heads, peg_causal, has_cross_attn, dim_context, attn_num_null_kv, ff_mult):
        super().__init__()
        self.peg = PEG(dim, causal=peg_causal)
        self.attn = Attention(dim, dim_head=dim_head, heads=heads)
        if has_cross_attn:
            self.cross = Attention(dim, dim_head=dim_head, dim_context=dim_context, heads=heads, num_null_kv=attn_num_null_kv)
        self.has_cross = has_cross_attn
        self.ff = feed_forward(dim, ff_mult)

    def execute(self, x, video_shape, attn_bias, context):
        x = self.peg(x, video_shape) + x
        x = self.attn(x, attn_bias=attn_bias) + x
        if self.has_cross and context is not None:
            x = self.cross(x, context=context) + x
        return self.ff(x) + x


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, dim_head: int = 64, heads: int = 8, ff_mult: int = 4, peg_causal: bool = True,
                 has_cross_attn: bool = False, dim_context: int = None, attn_num_null_kv: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            Layer(dim, dim_head, heads, peg_causal, has_cross_attn, dim_context, attn_num_null_kv, ff_mult)
            for _ in range(depth)
        ])
        self.norm_out = LayerNorm(dim)

    def execute(self, x, video_shape=None, attn_bias=None, context=None):
        for layer in self.layers:
            x = layer(x, video_shape, attn_bias, context)
        return self.norm_out(x)

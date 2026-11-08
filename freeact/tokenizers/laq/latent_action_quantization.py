"""LAPA's latent action quantization model (LAQ), encoder + NSVQ quantizer, in Jittor.

Port of ``laq/latent_action_quantization.py`` and ``laq/nsvq.py`` of LAPA
(https://github.com/LatentActionPretraining/LAPA, MIT) restricted to what a *tokenizer* needs:
``inference(video, return_only_codebook_ids=True)``.  The pixel decoder (``dec_spatial_transformer``,
``to_pixels_first_frame``) only serves LAPA's reconstruction loss and is not part of this port; the
training-time noise substitution of NSVQ is not needed for the argmin that selects the codes.
"""

import jittor as jt
from jittor import nn

from .attention import ContinuousPositionBias, Transformer


class NSVQ(nn.Module):
    """Quantizer of LAQ: project the patch tokens, shrink them with a small CNN, take the first->last difference."""

    def __init__(self, dim: int, num_embeddings: int, embedding_dim: int, code_seq_len: int, patch_size: int, image_size: int):
        super().__init__()
        self.embedding_dim, self.grid = embedding_dim, image_size // patch_size
        self.codebooks = jt.zeros((num_embeddings, embedding_dim))
        self.project_in = nn.Linear(dim, embedding_dim)
        self.project_out = nn.Linear(embedding_dim, dim)  # decoder side, kept so that the checkpoint loads whole
        conv = lambda k, s, p: nn.Conv2d(embedding_dim, embedding_dim, k, stride=s, padding=p)
        if code_seq_len == 1:
            layers = [conv(3, 2, 1), nn.ReLU(), conv(4, 1, 0)]
        elif code_seq_len == 2:
            layers = [conv(3, 2, 1), nn.ReLU(), conv((3, 4), 1, 0)]
        elif code_seq_len == 4:
            layers = [conv(3, 2, 1), nn.ReLU(), conv(3, 1, 0)]
        elif code_seq_len == 16:
            layers = [conv(3, 2, 1), nn.ReLU(), conv(3, 2, 1)]
        elif code_seq_len in (64, 256):
            layers = [conv(3, 2, 1)]
        else:
            raise ValueError(f"code_seq_len {code_seq_len} is not supported by LAQ")
        self.cnn_encoder = nn.Sequential(*layers)

    def encode(self, x):
        B = x.shape[0]
        x = self.project_in(x).transpose(0, 2, 1).reshape(B, self.embedding_dim, self.grid, self.grid)
        x = self.cnn_encoder(x).reshape(B, self.embedding_dim, -1)
        return x.transpose(0, 2, 1).reshape(-1, self.embedding_dim)

    def latent(self, first, last):
        """The vectors the nearest-code search runs on: ``(B * code_seq_len, embedding_dim)``."""
        return self.encode(last) - self.encode(first)

    def inference(self, first, last):
        diff = self.latent(first, last)
        w = self.codebooks
        dist = (diff * diff).sum(1, keepdims=True) - 2 * jt.matmul(diff, w.transpose(1, 0)) + (w * w).sum(1).unsqueeze(0)
        indices, _ = jt.argmin(dist, dim=1)
        return indices.reshape(first.shape[0], -1)


class LatentActionQuantization(nn.Module):
    def __init__(self, *, dim: int, quant_dim: int, codebook_size: int, image_size: int, patch_size: int,
                 spatial_depth: int, temporal_depth: int, dim_head: int = 64, heads: int = 8, channels: int = 3,
                 code_seq_len: int = 1):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size, self.patch_size, self.channels = image_size, patch_size, channels
        self.grid = image_size // patch_size
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)
        # [0] stands for the einops Rearrange of the original (done in ``patch_tokens``)
        self.to_patch_emb_first_frame = nn.Sequential(
            nn.Identity(), nn.LayerNorm(channels * patch_size * patch_size),
            nn.Linear(channels * patch_size * patch_size, dim), nn.LayerNorm(dim))
        kw = dict(dim=dim, dim_head=dim_head, heads=heads, peg_causal=True)
        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **kw)
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **kw)
        self.vq = NSVQ(dim, codebook_size, quant_dim, code_seq_len, patch_size, image_size)

    def patch_tokens(self, frames):
        """``(B, C, T, H, W)`` -> ``(B, T, h, w, dim)``."""
        B, C, T, H, W = frames.shape
        p, g = self.patch_size, self.grid
        x = frames.transpose(0, 2, 1, 3, 4).reshape(B * T, C, g, p, g, p).transpose(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, T, g, g, C * p * p)  # (c p1 p2) order, as the Rearrange of the original
        return self.to_patch_emb_first_frame(x)

    def encode(self, tokens):
        b, t, h, w, d = tokens.shape
        video_shape = (b, t, h, w)
        bias = self.spatial_rel_pos_bias(h, w)
        x = tokens.reshape(b * t, h * w, d)
        x = self.enc_spatial_transformer(x, video_shape=video_shape, attn_bias=bias)
        x = x.reshape(b, t, h, w, d).transpose(0, 2, 3, 1, 4).reshape(b * h * w, t, d)
        x = self.enc_temporal_transformer(x, video_shape=video_shape)
        x = x.reshape(b, h, w, t, d).transpose(0, 3, 1, 2, 4)
        return x[:, :1], x[:, 1:]

    def _first_last(self, video):
        assert tuple(video.shape[-2:]) == (self.image_size, self.image_size), video.shape
        first, last = self.encode(self.patch_tokens(video))
        b = first.shape[0]
        return first.reshape(b, -1, first.shape[-1]), last.reshape(b, -1, last.shape[-1])

    def inference(self, video):
        """``(B, C, 2, H, W)`` frame pairs in ``[0, 1]`` -> ``(B, code_seq_len)`` codebook indices."""
        return self.vq.inference(*self._first_last(video))

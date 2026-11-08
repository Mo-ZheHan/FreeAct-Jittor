"""LAPA's latent action tokenizer (LAQ, ``laq_openx.pt``) as a frozen model, in Jittor.

Port of the PyTorch release's ``freeact/tokenizers/lapa.py``.  LAPA's LAQ differs from UniVLA's LAM
in ways that are invisible from the file name, so each is fixed here:

===================  =========================  ==========================
                     UniVLA                     LAPA LAQ
===================  =========================  ==========================
video layout         ``(B, T, C, H, W)``        ``(B, C, T, H, W)``
input resolution     224 (DINOv2 ViT-B/14)      **256** (asserted, /32)
pixel preprocessing  ImageNet mean/std inside   raw ``[0, 1]``, no norm
codebook             16 codes x 4 positions     **8** codes x 4 positions
===================  =========================  ==========================

Inputs at another resolution (e.g. the policy's 224 px frames) are resized to 256 here.  LAPA's
own labeling script also feeds the model through a plain ``Resize((256, 256))``.
"""

import re

import jittor as jt
import numpy as np

from freeact.utils import freeze
from freeact.utils.checkpoint_io import load_torch_file

from .laq import LatentActionQuantization
from .laq.attention import remap_key
from .loader import copy_state
from .ops import resize_bilinear

CODE_SEQ_LEN = 4  # codes per frame pair of ``laq_openx.pt`` (published by LAPA; not recoverable from tensor shapes)

# Decoder-only sub-modules of LAQ: not built here; their checkpoint tensors are skipped by name.
DECODER_KEYS = ("dec_spatial_transformer.", "to_pixels_first_frame.")


class LapaLatentActionModel:
    """Wraps LAQ so it matches the interface of the other tokenizers."""

    backend = "lapa"

    def __init__(self, laq: LatentActionQuantization, image_size: int, num_latents: int, num_codes: int):
        self.laq = laq
        self.input_resolution = image_size
        self.num_latents = num_latents
        self.num_codes = num_codes

    def _video(self, frame1: jt.Var, frame2: jt.Var) -> jt.Var:
        if frame1.shape[-1] != self.input_resolution:
            frame1 = resize_bilinear(frame1, self.input_resolution)
            frame2 = resize_bilinear(frame2, self.input_resolution)
        return jt.stack([frame1, frame2], dim=2)  # LAQ wants (B, C, T, H, W)

    def encode_pairs(self, frame1: jt.Var, frame2: jt.Var) -> np.ndarray:
        """``(B, C, H, W)`` pairs of ``[0, 1]`` RGB -> ``(B, num_codes)`` codebook indices."""
        return self.laq.inference(self._video(frame1, frame2)).numpy()

    @property
    def codebook(self) -> jt.Var:
        return self.laq.vq.codebooks


def _infer_code_seq_len(state, grid: int) -> int:
    """Replay LAQ's ``cnn_encoder`` on a ``grid`` x ``grid`` map to recover ``code_seq_len``."""
    convs = sorted({int(m.group(1)) for k in state if (m := re.match(r"vq\.cnn_encoder\.(\d+)\.weight$", k))})
    h = grid
    for i in convs:
        w = state[f"vq.cnn_encoder.{i}.weight"]
        k = w.shape[-1]
        stride, pad = (2, 1) if i == convs[0] else (1, 0)  # as in every published LAQ config
        h = (h + 2 * pad - k) // stride + 1
    return h * h


def build_from_checkpoint(path: str, dim_head: int = 64, image_size: int = 256, code_seq_len: int = CODE_SEQ_LEN) -> LapaLatentActionModel:
    """Load ``laq_openx.pt`` into a frozen :class:`LapaLatentActionModel`.

    ``image_size`` and ``code_seq_len`` default to what LAPA published for ``laq_openx.pt`` and are
    authoritative: neither is recoverable from tensor shapes, so a wrong guess loads cleanly and
    silently changes every token.  The shapes are still cross-checked and a disagreement raises.
    """
    raw = load_torch_file(path)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    num_latents, quant_dim = state["vq.codebooks"].shape
    dim = state["to_patch_emb_first_frame.2.weight"].shape[0]
    patch_size = int(round((state["to_patch_emb_first_frame.2.weight"].shape[1] / 3) ** 0.5))
    if image_size % patch_size:
        raise ValueError(f"image_size={image_size} is not divisible by patch_size={patch_size}")
    grid = image_size // patch_size
    if (replayed := _infer_code_seq_len(state, grid)) != code_seq_len:
        raise ValueError(f"code_seq_len={code_seq_len} with image_size={image_size} (grid {grid}x{grid}) disagrees "
                         f"with the cnn_encoder in {path}, which replays to {replayed}.")

    def n_layers(prefix: str) -> int:
        idx = {int(m.group(1)) for k in state if (m := re.match(rf"{prefix}\.layers\.(\d+)\.", k))}
        return max(idx) + 1 if idx else 8

    laq = LatentActionQuantization(
        dim=dim, quant_dim=quant_dim, codebook_size=num_latents, image_size=image_size, patch_size=patch_size,
        spatial_depth=n_layers("enc_spatial_transformer"), temporal_depth=n_layers("enc_temporal_transformer"),
        dim_head=dim_head, heads=dim // dim_head, code_seq_len=code_seq_len,
    )
    # The original loads with ``strict=False``.  Here only decoder-only tensors and the empty ``null_kv`` of the
    # encoders' self-attention may be absent (shape ``(heads, 0, dim_head)``: the original creates it for every
    # attention layer, only the decoder's cross-attention has null keys), so a genuinely missing encoder tensor
    # still raises (a silently random encoder changes every token).
    ignore = [k for k, v in state.items() if k.startswith(DECODER_KEYS) or v.size == 0]
    res = copy_state(laq, state, rename=remap_key, ignore=ignore)
    if res["missing"] or res["unexpected"]:
        raise RuntimeError(f"LAPA LAQ checkpoint mismatch\n  missing: {res['missing'][:20]}\n  unexpected: {res['unexpected'][:20]}")

    freeze(laq)
    model = LapaLatentActionModel(laq, image_size, num_latents, code_seq_len)
    return model

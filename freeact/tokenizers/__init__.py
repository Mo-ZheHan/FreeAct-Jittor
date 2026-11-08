"""Frozen latent action models used as tokenizers in the latent action analysis (RQ1).

Three backends share one interface (``encode_pairs``, ``codebook``, ``num_latents``, ``num_codes``):

===========  ==========================================  ============  ==========
backend      checkpoint                                  vocabulary    input
===========  ==========================================  ============  ==========
``freeact``  ``Mo-ZheHan/FreeAct``: ``lam/lam.ckpt``     64 x 4 codes  224 px
``univla``   ``qwbu/univla-latent-action-model``         16 x 4 codes  224 px
``lapa``     ``latent-action-pretraining/LAPA-7B-openx`` 8 x 4 codes   256 px
===========  ==========================================  ============  ==========

All consume ``[0, 1]`` RGB frame pairs and handle their own resizing.  The backend is taken from
``LAM_BACKEND`` when set, and otherwise detected from the checkpoint's parameter names.
"""

import os

import jittor as jt
import numpy as np

from freeact.lam.checkpoint import load_lam
from freeact.utils.checkpoint_io import load_torch_skeleton

BACKENDS = ("freeact", "univla", "lapa")


class FreeActTokenizer:
    """The stage-1 :class:`DINOLatentActionModel` behind the common tokenizer interface."""

    backend = "freeact"

    def __init__(self, lam):
        self.lam = lam
        self.num_latents = lam.vq.num_latents
        self.num_codes = lam.num_codes

    def encode_pairs(self, frame1: jt.Var, frame2: jt.Var) -> np.ndarray:
        """``(B, C, H, W)`` pairs of ``[0, 1]`` RGB -> ``(B, num_codes)`` codebook indices."""
        return self.lam.vq_encode(jt.stack([frame1, frame2], dim=1))["indices"].numpy()

    @property
    def codebook(self) -> jt.Var:
        return self.lam.vq.codebook.weight


def detect_backend(path: str) -> str:
    """Identify the latent action model family from a checkpoint's key names (tensors are not read)."""
    if backend := os.environ.get("LAM_BACKEND", "").lower():
        if backend not in BACKENDS:
            raise ValueError(f"LAM_BACKEND={backend!r} must be one of {BACKENDS}")
        return backend
    raw = load_torch_skeleton(path)
    keys = set((raw.get("state_dict", raw) if isinstance(raw, dict) else raw) or {})
    if any(k.endswith("vq_action.codebook.weight") for k in keys):
        return "univla"
    if any(k.endswith("vq.codebooks") for k in keys):
        return "lapa"
    if any(k.endswith("action_decoder.net.0.weight") for k in keys):
        return "freeact"
    raise ValueError(f"Cannot tell which latent action model {path} is; set LAM_BACKEND explicitly.")


def load_tokenizer(path: str):
    """Load a frozen latent action tokenizer, the backend auto-detected from ``path``."""
    backend = detect_backend(path)
    if backend == "freeact":
        model = FreeActTokenizer(load_lam(path))
    elif backend == "univla":
        from .univla import build_from_checkpoint

        model = build_from_checkpoint(path)
    else:
        from .lapa import build_from_checkpoint

        model = build_from_checkpoint(path)
    model.backend = backend
    return model

"""UniVLA's task-centric latent action model as a frozen tokenizer, in Jittor.

Port of the PyTorch release's ``freeact/tokenizers/univla.py``, which is itself a vendored and
trimmed copy of ``OpenDriveLab/UniVLA`` (``latent_action_model/genie/modules/lam.py``, Apache-2.0).  Parameter
names follow upstream, so ``lam-stage-2.ckpt`` loads with the ``lam.`` wrapper prefix stripped.

Only the encoder side of the model is built: the pixel decoder (``decoder``), ``patch_up`` and
``action_up*`` serve UniVLA's reconstruction loss, not the tokenizer interface, and their checkpoint
keys are skipped by name; any other missing or unexpected key is an error.

The number of attention heads comes from the checkpoint's Lightning ``hyper_parameters``
(``lam_num_heads``, 12): it cannot be recovered from tensor shapes, and a wrong value would load
cleanly but produce other tokens.  ``lam_patch_size`` must be 14, the DINOv2 ViT-B/14 patch size.
"""

from typing import Dict

import jittor as jt
import numpy as np
from jittor import nn

from freeact.lam.modules.blocks import SpatioTemporalTransformer, VectorQuantizer
from freeact.utils import freeze
from freeact.utils.checkpoint_io import load_torch_file

from .dinov2 import EMBED_DIM as DINO_DIM
from .dinov2 import DinoV2ViT
from .loader import copy_state
from .ops import resize_bilinear

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_RESOLUTION = 224

# Reconstruction-only sub-modules of UniVLA's LAM: not part of a tokenizer.
_DECODER_KEYS = ("decoder.", "patch_up.", "action_up.", "action_up_uncontrol.")


class UniVLALatentActionModel(nn.Module):
    """Encoder and task-centric codebook of UniVLA's latent action model.

    Two codebooks act on the same DINOv2 features: ``vq`` holds the frozen stage-1 ("uncontrollable")
    codes, ``vq_action`` the task-centric codes the policy consumes.
    """

    def __init__(self, model_dim: int = 768, latent_dim: int = 128, num_latents: int = 16, num_codes: int = 4,
                 patch_size: int = 14, enc_blocks: int = 12, num_heads: int = 12, uncontrol_num_latents: int = 16):
        super().__init__()
        self.latent_dim, self.num_codes, self.patch_size = latent_dim, num_codes, patch_size
        self.input_resolution = INPUT_RESOLUTION
        self._mean = jt.array(np.array(IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1)).stop_grad()
        self._std = jt.array(np.array(IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)).stop_grad()
        self.dino_encoder = DinoV2ViT()

        self.action_latent = jt.rand((1, 1, num_codes, DINO_DIM)) * 2 - 1
        self.encoder = SpatioTemporalTransformer(
            in_dim=DINO_DIM, model_dim=model_dim, out_dim=latent_dim, num_blocks=enc_blocks,
            num_heads=num_heads, dropout=0.0, causal_temporal=True, to_out=False,
        )
        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.to_codebook_uncontrol = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(num_latents=uncontrol_num_latents, latent_dim=latent_dim)
        self.vq_action = VectorQuantizer(num_latents=num_latents, latent_dim=latent_dim)
        self.action_latent_controllable = jt.rand((1, 1, num_codes, DINO_DIM)) * 2 - 1

    @property
    def num_latents(self) -> int:
        return self.vq_action.num_latents

    @property
    def codebook(self) -> jt.Var:
        return self.vq_action.codebook.weight

    def vq_encode(self, videos: jt.Var) -> Dict:
        """``(B, T, C, H, W)`` clip of ``[0, 1]`` RGB -> dict with the task-centric ``indices`` (``(B*(T-1), K)``)."""
        B, T = videos.shape[:2]
        frames = videos.reshape((B * T,) + tuple(videos.shape[2:]))
        frames = (frames - self._mean) / self._std
        feats = self.dino_encoder.patch_tokens(frames).reshape(B, T, -1, DINO_DIM)

        # [controllable action tokens, stage-1 action tokens, DINO patch tokens]
        pad = self.action_latent.expand(B, T, self.num_codes, DINO_DIM)
        pad_c = self.action_latent_controllable.expand(B, T, self.num_codes, DINO_DIM)
        z = self.encoder(jt.concat([pad_c, pad, feats], dim=2))

        K = self.num_codes
        z_uncontrol = self.to_codebook_uncontrol(z[:, 1:, K: 2 * K]).reshape(B * (T - 1), K, self.latent_dim)
        z_q_uncontrol, _, _, indices_uncontrol = self.vq(z_uncontrol)
        z_action = self.to_codebook(z[:, 1:, :K]).reshape(B * (T - 1), K, self.latent_dim)
        z_q, z, emb, indices = self.vq_action(z_action)
        return {
            "z_q": z_q.reshape(B, T - 1, K, self.latent_dim),
            "z_q_uncontrol": z_q_uncontrol.reshape(B, T - 1, K, self.latent_dim),
            "emb": emb, "indices": indices, "indices_uncontrol": indices_uncontrol,
        }

    def _clip(self, frame1: jt.Var, frame2: jt.Var) -> jt.Var:
        if frame1.shape[-1] != self.input_resolution:
            frame1 = resize_bilinear(frame1, self.input_resolution)
            frame2 = resize_bilinear(frame2, self.input_resolution)
        return jt.stack([frame1, frame2], dim=1)

    def encode_pairs(self, frame1: jt.Var, frame2: jt.Var) -> np.ndarray:
        """``(B, C, H, W)`` pairs of ``[0, 1]`` RGB -> ``(B, num_codes)`` codebook indices."""
        return self.vq_encode(self._clip(frame1, frame2))["indices"].numpy()


def _strip_prefix(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k[4:] if k.startswith("lam.") else k: v for k, v in state.items()}


def _count_blocks(state: Dict[str, np.ndarray], prefix: str) -> int:
    import re

    pat = re.compile(rf"^{re.escape(prefix)}\.transformer_blocks\.(\d+)\.")
    idx = {int(m.group(1)) for k in state if (m := pat.match(k))}
    return max(idx) + 1 if idx else 12


def build_from_checkpoint(path: str) -> UniVLALatentActionModel:
    """Load UniVLA's ``lam-stage-2.ckpt`` into a frozen :class:`UniVLALatentActionModel`."""
    ckpt = load_torch_file(path)
    hp = ckpt.get("hyper_parameters") or {}
    state = _strip_prefix(ckpt.get("state_dict", ckpt))
    heads, patch_size = hp.get("lam_num_heads"), hp.get("lam_patch_size")
    if heads is None or patch_size is None:
        raise ValueError(
            f"{path} stores no lam_num_heads / lam_patch_size in hyper_parameters (got {heads}, {patch_size}). "
            "Neither can be recovered from tensor shapes, and guessing silently changes the tokens."
        )
    if patch_size != 14:
        raise ValueError(f"{path}: lam_patch_size={patch_size}, but the DINOv2 ViT-B/14 backbone needs 14")

    num_latents, latent_dim = state["vq_action.codebook.weight"].shape
    model_dim = state["to_codebook.weight"].shape[1]
    num_codes = state["action_latent_controllable"].shape[2]
    n_dino = sum(1 for k in state if k.startswith("dino_encoder."))
    lam = UniVLALatentActionModel(
        model_dim=model_dim, latent_dim=latent_dim, num_latents=num_latents, num_codes=num_codes,
        patch_size=patch_size, enc_blocks=_count_blocks(state, "encoder"), num_heads=heads,
        uncontrol_num_latents=state["vq.codebook.weight"].shape[0],
    )
    n_expected = sum(1 for k in lam.state_dict() if k.startswith("dino_encoder."))
    if n_dino != n_expected:
        raise RuntimeError(f"{path} holds {n_dino} of {n_expected} dino_encoder tensors; the DINOv2 backbone "
                           "must be stored in the checkpoint")

    ignore = [k for k in state if k.startswith(_DECODER_KEYS)]
    res = copy_state(lam, state, ignore=ignore)
    if res["missing"] or res["unexpected"]:
        raise RuntimeError(f"UniVLA LAM mismatch\n  missing: {res['missing'][:20]}\n"
                           f"  unexpected: {res['unexpected'][:20]}")
    lam.eval()
    return freeze(lam)

"""Loading a stage-1 latent action model checkpoint into Jittor."""

from typing import Dict

import jittor as jt
import numpy as np

from freeact.utils import freeze
from freeact.utils.checkpoint_io import load_torch_file

from .modules import DINOLatentActionModel


def build_lam(hp: Dict) -> DINOLatentActionModel:
    return DINOLatentActionModel(
        model_dim=hp["lam_model_dim"],
        latent_dim=hp["lam_latent_dim"],
        num_latents=hp["lam_num_latents"],
        num_codes=hp["lam_num_codes"],
        enc_blocks=hp["lam_enc_blocks"],
        dec_blocks=hp["lam_dec_blocks"],
        num_heads=hp["lam_num_heads"],
    )


def load_state_into(module: jt.nn.Module, state: Dict[str, np.ndarray], prefix: str = "", strict: bool = True):
    """Copy ``{name: ndarray}`` into ``module``'s parameters (names as in ``module.state_dict()``)."""
    params = module.state_dict()
    used = set()
    for name, var in params.items():
        key = prefix + name
        if key not in state:
            continue
        value = state[key]
        if tuple(value.shape) != tuple(var.shape):
            raise ValueError(f"Shape mismatch for {key}: checkpoint {value.shape} vs model {var.shape}")
        var.assign(jt.array(value.astype(np.float32) if value.dtype != np.float32 else value))
        used.add(key)
    missing = [n for n in params if prefix + n not in used]
    unexpected = [k for k in state if k.startswith(prefix) and k not in used]
    if strict and (missing or unexpected):
        raise RuntimeError(f"State mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    return missing, unexpected


def load_lam(path: str) -> DINOLatentActionModel:
    """Load a frozen :class:`DINOLatentActionModel` from ``lam.ckpt``.

    The released ``lam.ckpt`` (``{"hyper_parameters", "state_dict"}``) is the same file as used by
    the PyTorch implementation; the DINOv3 weights are part of it.
    """
    ckpt = load_torch_file(path)
    state = {
        k: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("lam.")
    }
    lam = build_lam(ckpt["hyper_parameters"])
    load_state_into(lam, state, prefix="lam.")
    lam.eval()
    freeze(lam)
    return lam

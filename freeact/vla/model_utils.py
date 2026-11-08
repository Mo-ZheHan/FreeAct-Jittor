"""Loading Qwen3-VL (base weights, processor, text tokenizer with the latent action tokens) without PyTorch / transformers."""

from __future__ import annotations

import json
import logging
import os
from typing import Tuple

import jittor as jt
import numpy as np

from freeact.utils.checkpoint_io import load_safetensors

from .processor import Qwen3VLProcessor
from .qwen3_vl import Linear, Qwen3VLConfig, Qwen3VLForConditionalGeneration

DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def resolve_base_model(path: str) -> str:
    """If ``path`` is a LoRA adapter directory, return its base model id."""
    adapter_cfg = os.path.join(path, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            if base := json.load(f).get("base_model_name_or_path"):
                return base
    return path


def resolve_local_dir(model_id: str) -> str:
    """Local directory of ``model_id`` (a path, or a Hugging Face Hub id resolved via the cache)."""
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])


def _load_into(model: Qwen3VLForConditionalGeneration, state: dict, used: set) -> None:
    params = model.state_dict()
    for name, arr in state.items():
        key = name[len("model.") :] if name.startswith("model.") else name
        if key not in params:
            continue
        var = params[key]
        if key == "visual.patch_embed.proj.weight":  # Conv3d (out, in, t, p, p) -> linear (out, in*t*p*p)
            arr = arr.reshape(arr.shape[0], -1)
        if tuple(arr.shape) != tuple(var.shape):
            raise ValueError(f"{name}: checkpoint {arr.shape} vs model {var.shape}")
        var.assign(jt.array(arr).cast(var.dtype))
        var.sync()  # materialize now: keeps the float32 staging copies from piling up on the GPU
        used.add(key)


def load_base_model(model_id: str = DEFAULT_BASE_MODEL) -> Qwen3VLForConditionalGeneration:
    local = resolve_local_dir(model_id)
    with open(os.path.join(local, "config.json"), encoding="utf-8") as f:
        config = Qwen3VLConfig.from_hf(json.load(f))
    model = Qwen3VLForConditionalGeneration(config)
    used: set = set()
    for fname in sorted(f for f in os.listdir(local) if f.endswith(".safetensors")):
        _load_into(model, load_safetensors(os.path.join(local, fname)), used)
        jt.sync_all()
        jt.gc()
    params = model.state_dict()
    missing = [k for k in params if k not in used and ".lora." not in k and k != "lm_head.weight"]
    if missing:
        raise RuntimeError(f"Missing Qwen3-VL weights: {missing[:10]}")
    if "lm_head.weight" not in used:  # tie_word_embeddings
        model.lm_head.weight.assign(model.language_model.embed_tokens.weight.clone())
    logging.info("Loaded Qwen3-VL base weights from %s", local)
    return model


def resize_embeddings(model: Qwen3VLForConditionalGeneration, new_size: int, seed: int = 0) -> bool:
    """Grow ``embed_tokens`` / ``lm_head`` to ``new_size`` rows.

    New rows are initialized as in transformers (``mean_resizing``: sampled from a Gaussian with
    the mean and covariance of the old embeddings).  The released checkpoints keep the latent
    action tokens within the reserved rows, so this is normally a no-op.
    """
    old = model.language_model.embed_tokens.weight
    if new_size <= old.shape[0]:
        return False
    for mod in (model.language_model.embed_tokens, model.lm_head):
        w = mod.weight.float32().numpy()
        mean, cov = w.mean(0), np.cov(w.T) * 1e-5
        new = np.random.default_rng(seed).multivariate_normal(mean, cov, new_size - w.shape[0]).astype(np.float32)
        mod.weight.assign(jt.array(np.concatenate([w, new])).bfloat16())
    if isinstance(model.lm_head, Linear):
        model.lm_head.out_features = new_size
    return True


def load_model_and_processor(model_name: str) -> Tuple[Qwen3VLForConditionalGeneration, Qwen3VLProcessor]:
    """Load the Qwen3-VL base model and processor.

    When ``model_name`` is a FreeAct checkpoint (LoRA adapter), the base model is loaded and the
    checkpoint's tokenizer (with the latent action tokens) replaces the stock one.
    """
    base_model = resolve_base_model(model_name)
    logging.info("Loading Qwen3-VL: %s", base_model)
    model = load_base_model(base_model)
    tok_dir = model_name if os.path.isfile(os.path.join(model_name, "tokenizer.json")) else resolve_local_dir(base_model)
    processor = Qwen3VLProcessor.from_pretrained(resolve_local_dir(base_model), tokenizer_dir=tok_dir)
    return model, processor

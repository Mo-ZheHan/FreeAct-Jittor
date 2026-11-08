"""LoRA adapters (PEFT-compatible) for the Jittor Qwen3-VL.

Follows the LoRA layers of PEFT (https://github.com/huggingface/peft, Apache-2.0).

The released checkpoints are PEFT LoRA adapters (``adapter_config.json`` +
``adapter_model.safetensors``) with ``target_modules="all-linear"`` (r=64, alpha=128) and full
copies of ``embed_tokens`` / ``lm_head`` (``modules_to_save``).  This module attaches identical
adapters to :class:`freeact.vla.qwen3_vl.Linear` layers, loads them under PEFT's parameter names
and merges them into the base weights for inference.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import jittor as jt
from jittor import nn

from freeact.utils.checkpoint_io import load_safetensors

from .qwen3_vl import Linear, Qwen3VLForConditionalGeneration

PEFT_PREFIX = "base_model.model."
# PEFT's "all-linear" on Qwen3-VL: every nn.Linear except the output layer (lm_head)
ALL_LINEAR_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "qkv", "attn.proj", "linear_fc1", "linear_fc2",
)


class LoRA(nn.Module):
    """``B(A(x)) * alpha / r``; weights are cast to the activation dtype (bfloat16)."""

    def __init__(self, in_features: int, out_features: int, r: int, alpha: float):
        super().__init__()
        self.r, self.scaling = r, alpha / r
        # overwritten by load_adapter
        self.lora_A = jt.zeros((r, in_features), "float32")
        self.lora_B = jt.zeros((out_features, r), "float32")

    def execute(self, x: jt.Var) -> jt.Var:
        h = jt.matmul(x, self.lora_A.cast(x.dtype).transpose(1, 0))
        return jt.matmul(h, self.lora_B.cast(x.dtype).transpose(1, 0)) * self.scaling


def _linear_modules(model: nn.Module) -> List[Tuple[str, Linear]]:
    out = []
    for name, m in model.named_modules():
        if isinstance(m, Linear) and name != "lm_head" and name.endswith(ALL_LINEAR_SUFFIXES):
            # the vision patch embedding is a Conv3d in transformers and is not targeted by PEFT
            if name.endswith("patch_embed.proj"):
                continue
            out.append((name, m))
    return out


def _peft_name(module_name: str) -> str:
    """Our module path -> PEFT's (``model.`` wraps visual / language_model in transformers)."""
    if module_name.startswith(("visual.", "language_model.")):
        return f"{PEFT_PREFIX}model.{module_name}"
    return f"{PEFT_PREFIX}{module_name}"


def add_lora(model: Qwen3VLForConditionalGeneration, r: int = 64, alpha: float = 128) -> List[str]:
    names = []
    for name, m in _linear_modules(model):
        m.lora = LoRA(m.in_features, m.out_features, r, alpha)
        names.append(name)
    return names


def load_adapter(model: Qwen3VLForConditionalGeneration, adapter_dir: str, merge: bool = False) -> Dict:
    """Attach and load a PEFT LoRA adapter; optionally merge it into the base weights."""
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    state = load_safetensors(os.path.join(adapter_dir, "adapter_model.safetensors"))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    add_lora(model, r, alpha)

    used = set()
    for name, m in _linear_modules(model):
        a_key = f"{_peft_name(name)}.lora_A.weight"
        b_key = f"{_peft_name(name)}.lora_B.weight"
        if a_key not in state:
            raise KeyError(f"LoRA weight {a_key} missing from {adapter_dir}")
        m.lora.lora_A.assign(jt.array(state[a_key]))
        m.lora.lora_B.assign(jt.array(state[b_key]))
        m.lora.lora_A.sync(), m.lora.lora_B.sync()
        used.update((a_key, b_key))

    # modules_to_save: full copies of the (resized) embedding and output layer
    for key, target in (
        (f"{PEFT_PREFIX}model.language_model.embed_tokens.weight", model.language_model.embed_tokens),
        (f"{PEFT_PREFIX}lm_head.weight", model.lm_head),
    ):
        if key in state:
            w = state[key]
            if tuple(w.shape) != tuple(target.weight.shape):
                raise ValueError(f"{key}: adapter {w.shape} vs model {target.weight.shape} (resize the vocabulary first)")
            target.weight.assign(jt.array(w).bfloat16())
            target.weight.sync()
            used.add(key)
    if unused := sorted(set(state) - used):
        raise RuntimeError(f"Unused adapter weights: {unused[:5]} ...")
    if merge:
        merge_lora(model)
    return cfg


def merge_lora(model: nn.Module) -> None:
    """``W <- W + (B A * scaling).to(W.dtype)``, as PEFT's ``merge(safe_merge=True)``."""
    for _, m in _linear_modules(model):
        if m.lora is None:
            continue
        delta = jt.matmul(m.lora.lora_B, m.lora.lora_A) * m.lora.scaling
        m.weight.assign(m.weight + delta.cast(m.weight.dtype))
        m.weight.sync()
        m.lora = None
    jt.sync_all()
    jt.gc()

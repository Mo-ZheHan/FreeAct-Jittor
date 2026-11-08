"""Qwen3-VL processor without transformers: chat template, tokenizer and image patchification.

Mirrors ``Qwen3VLProcessor`` + ``Qwen2VLImageProcessorFast`` (transformers 4.57) for the inputs
FreeAct uses: one RGB image per prompt, ``min_pixels=784``, ``max_pixels=50176`` (224 x 224).
The tokenizer is the Hugging Face ``tokenizer.json`` of the checkpoint (``tokenizers`` library),
so token ids are identical to the PyTorch implementation.

Original code: Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team, licensed
under the Apache License, Version 2.0 (https://github.com/huggingface/transformers).
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

IMAGE_PAD = "<|image_pad|>"


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    """Same as ``transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize``."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen2VLImageProcessor:
    def __init__(self, cfg: Dict):
        self.patch_size = cfg.get("patch_size", 16)
        self.temporal_patch_size = cfg.get("temporal_patch_size", 2)
        self.merge_size = cfg.get("merge_size", 2)
        self.image_mean = np.asarray(cfg.get("image_mean", [0.5] * 3), np.float32)
        self.image_std = np.asarray(cfg.get("image_std", [0.5] * 3), np.float32)
        size = cfg.get("size", {})
        self.min_pixels = cfg.get("min_pixels") or size.get("shortest_edge", 56 * 56)
        self.max_pixels = cfg.get("max_pixels") or size.get("longest_edge", 28 * 28 * 1280)

    def __call__(self, image) -> Tuple[np.ndarray, np.ndarray]:
        """PIL image or (H, W, 3) uint8 -> ``(pixel_values (N, 1536) float32, grid_thw (1, 3))``."""
        img = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
        h, w = img.shape[:2]
        factor = self.patch_size * self.merge_size
        rh, rw = smart_resize(h, w, factor, self.min_pixels, self.max_pixels)
        if (rh, rw) != (h, w):
            # bicubic resize (transformers uses torchvision with antialias on uint8 tensors)
            img = np.asarray(Image.fromarray(img).resize((rw, rh), Image.BICUBIC))
        # fused rescale + normalize, as in the fast processor: (x - mean*255) / (std*255)
        x = (img.astype(np.float32) - self.image_mean * 255.0) / (self.image_std * 255.0)
        x = x.transpose(2, 0, 1)[None]  # (T=1, C, H, W)
        x = np.concatenate([x] * self.temporal_patch_size, axis=0)  # repeat the frame temporally
        t, c = 1, x.shape[1]
        p, m = self.patch_size, self.merge_size
        gh, gw = rh // p, rw // p
        x = x.reshape(t, self.temporal_patch_size, c, gh // m, m, p, gw // m, m, p)
        # (t, gh/m, gw/m, m, m, c, tp, p, p)
        x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        pixel_values = x.reshape(t * gh * gw, c * self.temporal_patch_size * p * p)
        return np.ascontiguousarray(pixel_values, np.float32), np.array([[t, gh, gw]], dtype=np.int64)


class Tokenizer:
    """Thin wrapper over a Hugging Face ``tokenizer.json`` with the attributes FreeAct uses."""

    def __init__(self, tokenizer_dir: str):
        from tokenizers import Tokenizer as HFTokenizer

        self._tok = HFTokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        self.pad_token = self.config.get("pad_token")
        self.eos_token = self.config.get("eos_token")
        self.pad_token_id = self.convert_tokens_to_ids(self.pad_token) if self.pad_token else None
        self.eos_token_id = self.convert_tokens_to_ids(self.eos_token) if self.eos_token else None
        self.tokenizer_dir = tokenizer_dir

    def __len__(self) -> int:
        return self._tok.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._tok.token_to_id(tokens)
        return [self._tok.token_to_id(t) for t in tokens]


class Qwen3VLProcessor:
    def __init__(self, image_processor: Qwen2VLImageProcessor, tokenizer: Tokenizer, chat_template: str,
                 preprocessor_config: Dict):
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.preprocessor_config = preprocessor_config
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
        self._template = env.from_string(chat_template)

    @classmethod
    def from_pretrained(cls, model_dir: str, tokenizer_dir: Optional[str] = None) -> "Qwen3VLProcessor":
        with open(os.path.join(model_dir, "preprocessor_config.json"), encoding="utf-8") as f:
            pcfg = json.load(f)
        template = None
        for d in (tokenizer_dir, model_dir):
            if d and os.path.isfile(os.path.join(d, "chat_template.jinja")):
                with open(os.path.join(d, "chat_template.jinja"), encoding="utf-8") as f:
                    template = f.read()
                break
            if d and os.path.isfile(os.path.join(d, "chat_template.json")):
                with open(os.path.join(d, "chat_template.json"), encoding="utf-8") as f:
                    template = json.load(f)["chat_template"]
                break
        return cls(Qwen2VLImageProcessor(pcfg), Tokenizer(tokenizer_dir or model_dir), template, pcfg)

    def apply_chat_template(self, messages: List[Dict], add_generation_prompt: bool = False) -> Dict[str, np.ndarray]:
        """Render, tokenize and expand ``<|image_pad|>`` -> ``input_ids (1, L)``, ``pixel_values``, ``image_grid_thw``."""
        text = self._template.render(messages=messages, add_generation_prompt=add_generation_prompt)
        images = [c["image"] for m in messages if isinstance(m["content"], list) for c in m["content"] if c.get("type") == "image"]
        pixel_values, grids = [], []
        for img in images:
            pv, g = self.image_processor(img)
            pixel_values.append(pv)
            grids.append(g)
        merge = self.image_processor.merge_size**2
        parts = text.split(IMAGE_PAD)
        assert len(parts) == len(images) + 1, "number of <|image_pad|> does not match the images"
        expanded = parts[0]
        for g, rest in zip(grids, parts[1:]):
            expanded += "<|placeholder|>" * int(np.prod(g) // merge) + rest
        expanded = expanded.replace("<|placeholder|>", IMAGE_PAD)
        ids = np.asarray(self.tokenizer.encode(expanded), dtype=np.int64)[None]
        out = {"input_ids": ids}
        if images:
            out["pixel_values"] = np.concatenate(pixel_values)
            out["image_grid_thw"] = np.concatenate(grids)
        return out

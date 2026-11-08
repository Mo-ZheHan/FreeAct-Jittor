"""Closed-loop inference with a FreeAct stage-3 policy."""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

import jittor as jt
import numpy as np
from PIL import Image

from freeact.data import libero_from_camera_frame
from freeact.transforms import array_to_pil, resize_chw
from freeact.lam.checkpoint import load_state_into
from freeact.utils.checkpoint_io import load_torch_file

from .action_decoder import VLAActionDecoder
from .data.utils import IMAGE_TOKEN_INDEX, build_vla_message, denormalize_action
from .lora import load_adapter
from .model_utils import load_model_and_processor, resize_embeddings
from .qwen3_vl import KVCache
from .data.rope2d import get_rope_index_3

__all__ = ["VLAPolicy", "load_vla_model", "center_crop_resize"]


def center_crop_resize(img: np.ndarray, size: int, scale: float = 0.9) -> Image.Image:
    """Deterministic counterpart of the training-time ``RandomResizedCrop(scale=(s, s), ratio=1)``.

    ``img``: (C, H, W) float in [0, 1].  Bilinear resize with antialiasing, as torchvision's
    ``v2.functional.resize(antialias=True)`` on float tensors.
    """
    _, h, w = img.shape
    crop_side = min(int(round(math.sqrt(scale * h * w))), h, w)
    top, left = (h - crop_side) // 2, (w - crop_side) // 2
    cropped = img[:, top : top + crop_side, left : left + crop_side]
    return array_to_pil(resize_chw(cropped, size, size))


def sample_token(logits: np.ndarray, rng: np.random.Generator, temperature: float, top_k: int, top_p: float,
                 min_tokens_to_keep: int = 1) -> int:
    """Temperature -> top-k -> top-p -> multinomial, as ``transformers`` (float32 scores).

    ``temperature=0`` selects the most likely token (greedy decoding).
    """
    if temperature == 0:
        return int(np.argmax(logits))
    scores = logits.astype(np.float32) / np.float32(temperature)
    if top_k:
        kth = np.partition(scores, -top_k)[-top_k]
        scores = np.where(scores < kth, -np.inf, scores)
    if top_p < 1.0:
        order = np.argsort(scores, kind="stable")  # ascending
        s = scores[order]
        p = np.exp(s - s.max())
        p /= p.sum()
        remove = np.cumsum(p) <= (1 - top_p)
        remove[-min_tokens_to_keep:] = False
        scores[order[remove]] = -np.inf
    p = np.exp(scores - scores.max())
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


class VLAPolicy:
    """Qwen3-VL with the continuous action head (MAP pooling).

    The VLA samples ``num_codes`` latent action tokens; the hidden states of those tokens and of
    the image tokens are decoded into a normalized camera-frame action chunk, which is
    denormalized and (for LIBERO) rotated back into the robot base frame.
    """

    def __init__(
        self,
        model,
        processor,
        action_decoder: VLAActionDecoder,
        action_stats: Dict[str, np.ndarray],
        center_crop: bool = True,
        crop_scale: float = 0.9,
        resolution: int = 224,
        camera_frame: bool = True,
        temperature: float = 0.75,
        top_p: float = 0.9,
        top_k: int = 20,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.processor = processor
        self.action_decoder = action_decoder
        self.action_stats = action_stats
        self.center_crop = center_crop
        self.crop_scale = crop_scale
        self.resolution = resolution
        self.camera_frame = camera_frame
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.rng = np.random.default_rng(seed)
        self.merge_size = processor.image_processor.merge_size

    def _preprocess(self, img: np.ndarray) -> Image.Image:
        if self.center_crop:
            return center_crop_resize(img, self.resolution, self.crop_scale)
        return array_to_pil(img)

    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """``obs["observation.images.image"]``: (B, C, H, W) in [0, 1]; ``obs["task"]``: list[str]."""
        images = obs["observation.images.image"]
        tasks = obs.get("task", [""] * images.shape[0])
        actions = []
        for i, img in enumerate(images):
            task = tasks[i] if isinstance(tasks, (list, tuple)) else tasks
            inputs = self.processor.apply_chat_template(
                build_vla_message(self._preprocess(img), task), add_generation_prompt=True
            )
            action = self._predict(inputs)
            if self.camera_frame:
                action = libero_from_camera_frame(action)
            actions.append(action)
        return np.stack(actions).astype(np.float32)

    def generate_latents(self, inputs: Dict):
        """Prefill + ``num_codes`` sampled tokens.

        Returns the pre-norm last-layer states of the prompt and of the ``num_codes`` generated
        tokens (``generate(output_hidden_states=True)`` in transformers), and the token ids.
        """
        num_codes = self.action_decoder.num_codes
        m = self.model
        ids = inputs["input_ids"]
        pos, delta = get_rope_index_3(self.merge_size, ids, inputs["image_grid_thw"])
        pos = pos[:, 0]  # (3, L)
        L = ids.shape[1]
        cache = KVCache(len(m.language_model.layers))
        with jt.no_grad():
            x, visual_pos, deepstack = m.embed(ids, jt.array(inputs["pixel_values"]), inputs["image_grid_thw"])
            h, prefill, _ = m.language_model(x, pos, np.array([0, L], np.int32), visual_pos, deepstack, cache=cache)
            logits = m.lm_head(h[-1:]).float32().numpy()[0]
            latent_states, generated = [], []
            # after the prefill, each sampled token yields one hidden state: num_codes in total
            for step in range(num_codes):
                tok = sample_token(logits, self.rng, self.temperature, self.top_k, self.top_p)
                generated.append(tok)
                p = np.full((3, 1), L + step + int(delta.reshape(-1)[0]), dtype=np.int64)
                e = m.language_model.embed_tokens(jt.array(np.array([tok], np.int32)))
                h, last, _ = m.language_model(e, p, np.array([0, 1], np.int32), cache=cache)
                latent_states.append(last)
                if step < num_codes - 1:
                    logits = m.lm_head(h).float32().numpy()[0]
            latent = jt.concat(latent_states, dim=0)
        return prefill, latent, generated

    def _predict(self, inputs: Dict) -> np.ndarray:
        prefill, latent, _ = self.generate_latents(inputs)
        mask = inputs["input_ids"][0] == IMAGE_TOKEN_INDEX
        with jt.no_grad():
            visual = prefill[jt.array(np.nonzero(mask)[0].astype(np.int32))]
            pred = self.action_decoder(latent.float32().unsqueeze(0), visual.float32().unsqueeze(0))[0].numpy()
        action = pred[0]
        return denormalize_action(action, self.action_stats["q01"], self.action_stats["q99"])


def load_action_decoder(vla_path: Path) -> VLAActionDecoder:
    with open(vla_path / "action_decoder_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    decoder = VLAActionDecoder(**cfg)
    # Stored in bfloat16; widened to float32 on loading.
    load_state_into(decoder, load_torch_file(str(vla_path / "action_decoder.pt")))
    logging.info("Action head: window=%d, latent action tokens=%d", cfg["window_size"], cfg["num_codes"])
    decoder.eval()
    return decoder


def load_vla_model(
    vla_path: str,
    max_pixels: int = 50176,
    min_pixels: int = 784,
    **policy_kwargs,
) -> VLAPolicy:
    """Load a stage-3 checkpoint (LoRA adapter + ``action_decoder.pt`` + ``action_stats.json``)."""
    path = Path(vla_path)
    model, processor = load_model_and_processor(str(path))
    processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    resize_embeddings(model, len(processor.tokenizer))
    load_adapter(model, str(path), merge=True)
    model.eval()

    with open(path / "action_stats.json", "r", encoding="utf-8") as f:
        stats = {k: np.asarray(v, dtype=np.float32) for k, v in json.load(f).items()}

    return VLAPolicy(model, processor, load_action_decoder(path), stats, **policy_kwargs)

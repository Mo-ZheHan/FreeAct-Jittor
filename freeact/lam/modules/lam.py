"""Physically grounded latent action model (FreeAct stage 1).

A frozen DINOv3 ViT encodes a frame pair; a spatio-temporal transformer distills ``num_codes``
latent action tokens that are vector-quantized against a ``num_latents``-entry codebook.  A
spatial decoder reconstructs the DINO features of the future frame from the current frame and the
quantized latents, and an auxiliary MLP Action Decoder regresses the camera-frame end-effector
action from the quantized latents.
"""

from typing import Dict

import jittor as jt
import numpy as np
from jittor import nn

from freeact.utils import freeze

from .blocks import SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer
from .dinov3 import EMBED_DIM as DINO_DIM
from .dinov3 import DinoV3ViT

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
DINO_NUM_PREFIX_TOKENS = DinoV3ViT.num_prefix_tokens  # CLS + 4 register tokens


class ActionDecoder(nn.Module):
    """MLP that maps the quantized latent actions ``(B, num_codes, latent_dim)`` to a 7D action."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, action_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def execute(self, z: jt.Var) -> jt.Var:
        return self.net(z.reshape(z.shape[0], -1))


class DINOLatentActionModel(nn.Module):
    """Latent action VQ-VAE operating in the DINOv3 feature space."""

    def __init__(
        self,
        model_dim: int,
        latent_dim: int,
        num_latents: int,
        num_codes: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_codes = num_codes

        self._dino_mean = jt.array(np.array(IMAGENET_DEFAULT_MEAN, np.float32).reshape(1, 3, 1, 1)).stop_grad()
        self._dino_std = jt.array(np.array(IMAGENET_DEFAULT_STD, np.float32).reshape(1, 3, 1, 1)).stop_grad()
        self.dino_encoder = DinoV3ViT()
        freeze(self.dino_encoder)

        # Learnable query tokens that become the latent actions
        self.action_latent = jt.rand(1, 1, num_codes, DINO_DIM) * 2 - 1
        self.encoder = SpatioTemporalTransformer(
            in_dim=DINO_DIM,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )
        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(num_latents=num_latents, latent_dim=latent_dim)

        # Reconstruction decoder
        self.patch_up = nn.Linear(DINO_DIM, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=DINO_DIM,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Auxiliary Action Decoder (physical grounding)
        action_input_dim = num_codes * latent_dim
        self.action_decoder = ActionDecoder(input_dim=action_input_dim, hidden_dim=action_input_dim // 2, action_dim=7)

    def dino_features(self, frames: jt.Var) -> jt.Var:
        """``(N, C, H, W)`` frames in ``[0, 1]`` -> ``(N, 196, 768)`` frozen DINOv3 patch tokens."""
        with jt.no_grad():
            out = self.dino_encoder.forward_features((frames - self._dino_mean) / self._dino_std)
        return out[:, DINO_NUM_PREFIX_TOKENS:].float32().stop_grad()

    def vq_encode(self, videos: jt.Var) -> Dict:
        """Encode ``(B, 2, C, H, W)`` frame pairs in ``[0, 1]`` into quantized latent actions."""
        B, T = videos.shape[:2]
        dino_features = self.dino_features(videos.reshape((B * T,) + tuple(videos.shape[2:])))
        dino_features = dino_features.reshape(B, T, dino_features.shape[1], dino_features.shape[2])

        action_pad = self.action_latent.expand(B, T, self.num_codes, DINO_DIM)
        z = self.encoder(jt.concat([action_pad, dino_features], dim=2))

        # Latent actions are read out at the future frame
        z = self.to_codebook(z[:, 1:, : self.num_codes])  # (B, T-1, num_codes, latent_dim)
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z_code, z_e, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {"patches": dino_features, "z_q": z_q, "z_code": z_code, "z_e": z_e, "indices": indices}

    def execute(self, batch: Dict) -> Dict:
        outputs = self.vq_encode(batch["videos"])
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])

        video_recon = self.decoder(jt.concat([action_patches, video_patches], dim=2))
        video_recon = video_recon[:, :, self.num_codes : self.num_codes + video_patches.shape[2]]

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, -1:],
                "pred_action": self.action_decoder(outputs["z_q"].squeeze(1)),
            }
        )
        return outputs

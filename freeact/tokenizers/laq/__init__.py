"""Jittor port of LAPA's Latent Action Quantization model (inference only).

Ported from https://github.com/LatentActionPretraining/LAPA (``laq/laq_model``, MIT), whose LAQ
builds on phenaki-pytorch (https://github.com/lucidrains/phenaki-pytorch, MIT) and NSVQ
(https://github.com/MHVali/Noise-Substitution-in-Vector-Quantization, MIT); LICENSE in this
directory lists all copyright notices.
"""

from .latent_action_quantization import LatentActionQuantization

__all__ = ["LatentActionQuantization"]

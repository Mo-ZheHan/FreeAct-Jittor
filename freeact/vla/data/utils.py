"""Chat message and action denormalization helpers of the VLA policy."""

import numpy as np
from PIL import Image

IMAGE_TOKEN_INDEX = 151655


def build_vla_message(image: Image.Image, task: str) -> list:
    """Build the user turn of the VLA chat template."""
    prompt = (
        f"What action should the robot take to {task.lower()}?"
        if task
        else "What action should the robot take?"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def denormalize_action(norm: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Denormalize action from [-1, 1] using q01/q99 quantiles."""
    return (norm + 1.0) / 2.0 * (q99 - q01) + q01

"""Step-level task definitions."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class StepTask:
    """A single step task (one observation requiring inference).

    Attributes:
        env_id: Unique identifier for the environment instance.
        observation: Preprocessed observation dict ready for policy.
    """

    env_id: int
    observation: dict[str, Any]


@dataclass
class StepResult:
    """Result of a step inference.

    Attributes:
        env_id: Environment ID this result belongs to.
        action: Action to apply to the environment, or None if inference failed.
    """

    env_id: int
    action: Optional[np.ndarray]

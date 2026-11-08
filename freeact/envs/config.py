"""Perturbation configuration of the target environment."""

from dataclasses import dataclass, fields

_VALID_VALUES = {
    "lighting": {"bright"},
    "texture": {"checkerboard"},
    "robot": {"white_arm"},
}


@dataclass
class PerturbationConfig:
    lighting: str | None = None
    texture: str | None = None
    robot: str | None = None
    camera_azimuth: float | None = None  # agent-view camera rotation about the table, in degrees
    object_swap: dict[str, str] | None = None  # BDDL object type -> registered object type
    language_swap: dict[str, str] | None = None  # applied to the task instruction

    def __post_init__(self):
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None and f.name in _VALID_VALUES and val not in _VALID_VALUES[f.name]:
                raise ValueError(f"Invalid {f.name}={val!r}, must be one of {_VALID_VALUES[f.name]}")

    @property
    def has_scene_override(self) -> bool:
        return self.lighting is not None or self.texture is not None

    @property
    def is_empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))

    def apply_language_swap(self, text: str) -> str:
        for old, new in (self.language_swap or {}).items():
            text = text.replace(old, new)
        return text

    def summary(self) -> str:
        return ", ".join(f"{f.name}={v}" for f in fields(self) if (v := getattr(self, f.name)) is not None)

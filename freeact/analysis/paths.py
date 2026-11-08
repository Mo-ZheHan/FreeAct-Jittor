"""Where the RQ1 features live (kept apart from ``extract`` so that the plotting / probing scripts do not import Jittor)."""

from pathlib import Path

DEFAULT_OUT = Path("./outputs/analysis")


def feature_path(out_dir: Path, name: str) -> Path:
    return out_dir / f"features_{name}.npz"


def load_features(out_dir: Path, name: str):
    """The features extracted for ``name``, or exit with a hint when ``extract`` has not been run."""
    import numpy as np

    path = feature_path(out_dir, name)
    if not path.is_file():
        raise SystemExit(f"{path} not found; run python -m freeact.analysis.extract --lam <tokenizer checkpoint> --name {name} first")
    return np.load(path)

"""UMAP of held-out latents colored by end-effector displacement (Fig. 3b and appendix figures)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm

from .paths import DEFAULT_OUT, load_features

UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, spread=1.0)


def embed(X: np.ndarray, seed: int) -> np.ndarray:
    import umap

    return umap.UMAP(n_components=2, random_state=seed, low_memory=False, **UMAP_PARAMS).fit_transform(X)


def _component(actions: np.ndarray, which: str) -> np.ndarray:
    return np.linalg.norm(actions[:, :3], axis=1) if which == "mag" else actions[:, "xyz".index(which)]


def _square_lims(coords):
    lo, hi = coords.min(0) - 0.5, coords.max(0) + 0.5
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    return (c[0] - r, c[0] + r), (c[1] - r, c[1] + r)


def plot_comparison(data: dict, out_dir: Path) -> None:
    """One row of panels (one per model) per displacement component, with a shared colorbar."""
    names = list(data)
    S, G, CW, CG, ML, MR, MT, MB = 4.0, 0.3, 0.25, 0.15, 0.4, 0.6, 0.2, 0.4
    W, H = ML + len(names) * S + (len(names) - 1) * G + CG + CW + MR, MT + S + MB

    for which in ("x", "y", "z", "mag"):
        is_mag = which == "mag"
        vals_all = np.concatenate([_component(d["actions"], which) for d in data.values()])
        vmax = np.percentile(vals_all if is_mag else np.abs(vals_all), 99)
        norm = plt.Normalize(0, vmax) if is_mag else CenteredNorm(0, vmax)

        fig = plt.figure(figsize=(W, H))
        for i, d in enumerate(data.values()):
            ax = fig.add_axes([(ML + i * (S + G)) / W, MB / H, S / W, S / H])
            ax.set_aspect("equal")
            vals = _component(d["actions"], which)
            ok = vals <= vmax if is_mag else np.abs(vals) <= vmax
            sc = ax.scatter(d["coords"][ok, 0], d["coords"][ok, 1], c=vals[ok], cmap="viridis" if is_mag else "RdBu_r",
                            norm=norm, alpha=0.5, s=8, edgecolors="none")
            xlim, ylim = _square_lims(d["coords"])
            ax.set(xlim=xlim, ylim=ylim)
            ax.grid(True, alpha=0.3, ls="--")
        cax = fig.add_axes([(ML + len(names) * S + (len(names) - 1) * G + CG) / W, MB / H, CW / W, S / H])
        fig.colorbar(sc, cax=cax).set_label("m")
        path = out_dir / f"umap_{which}_{'_'.join(names)}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"-> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names", nargs="+", required=True, help="panels from left to right")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = {}
    for name in args.names:
        f = load_features(args.out_dir, name)
        X, y = f["X_te"], f["y_te"]
        if len(X) > args.max_samples:
            sel = np.sort(np.random.default_rng(args.seed).choice(len(X), args.max_samples, replace=False))
            X, y = X[sel], y[sel]
        print(f"UMAP {name}: {X.shape}")
        data[name] = {"coords": embed(X, args.seed), "actions": y}
    plot_comparison(data, args.out_dir)


if __name__ == "__main__":
    main()

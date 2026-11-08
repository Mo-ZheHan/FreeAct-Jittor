"""Normalized latent-distance ratio (NLDR, Fig. 3a of the paper).

Held-out sample pairs are grouped by action distance (``--intervals``).  For each group, NLDR is
the mean latent L2 distance of its pairs divided by the mean over all pairs, so 1 is chance level.
The action distance sums the mean SmoothL1 distance of translation, rotation and gripper, so each
component gets equal weight.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .paths import DEFAULT_OUT, load_features

DEFAULT_INTERVALS = [0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]


def _pairwise_smooth_l1(x: np.ndarray) -> np.ndarray:
    d = np.abs(x[:, None, :] - x[None, :, :])
    return np.where(d < 1, 0.5 * d**2, d - 0.5).mean(axis=-1)


def action_distances(actions: np.ndarray) -> np.ndarray:
    return (
        _pairwise_smooth_l1(actions[:, 0:3])
        + _pairwise_smooth_l1(actions[:, 3:6])
        + _pairwise_smooth_l1(actions[:, 6:7])
    )


def latent_distances(emb: np.ndarray) -> np.ndarray:
    sq = (emb**2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * emb @ emb.T, 0))


def nldr(actions: np.ndarray, emb: np.ndarray, intervals) -> dict:
    triu = np.triu_indices(len(actions), k=1)
    a, l = action_distances(actions)[triu], latent_distances(emb)[triu]
    random_mean = float(l.mean())
    out = []
    for lo, hi in zip(intervals[:-1], intervals[1:]):
        m = (a >= lo) & (a < hi)
        out.append({
            "interval": [lo, hi],
            "ratio": float(l[m].mean() / random_mean) if m.any() else None,
            "std": float(l[m].std() / random_mean) if m.any() else None,
            "count": int(m.sum()),
        })
    return {"random_mean": random_mean, "intervals": out}


def plot(results: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(1.0, color="gray", ls="--", lw=1.5, alpha=0.7, label="random")
    for name, res in results.items():
        # each interval is drawn at its geometric center (the first one, starting at 0, at hi / sqrt 2)
        x = np.array([np.sqrt(lo * hi) if lo > 0 else hi / np.sqrt(2) for lo, hi in (d["interval"] for d in res["intervals"])])
        r = np.array([d["ratio"] if d["ratio"] is not None else np.nan for d in res["intervals"]])
        ok = ~np.isnan(r)
        ax.plot(x[ok], r[ok], "o-", markersize=5, linewidth=1.5, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Action distance")
    ax.set_ylabel("NLDR")
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--intervals", type=float, nargs="+", default=DEFAULT_INTERVALS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    results, sel = {}, None
    for name in args.names:
        f = load_features(args.out_dir, name)
        if sel is None:  # same held-out pairs for every model
            n = len(f["y_te"])
            sel = np.sort(np.random.default_rng(args.seed).choice(n, min(args.num_samples, n), replace=False))
        results[name] = nldr(f["y_te"][sel], f["X_te"][sel], args.intervals)
        first = next(d for d in results[name]["intervals"] if d["ratio"] is not None)
        print(f"{name}: NLDR at smallest populated interval {first['interval']} = {first['ratio']:.3f}")

    with open(args.out_dir / "nldr.json", "w") as fh:
        json.dump(results, fh, indent=2)
    plot(results, args.out_dir / "nldr.png")
    print(f"-> {args.out_dir / 'nldr.json'}, {args.out_dir / 'nldr.png'}")


if __name__ == "__main__":
    main()

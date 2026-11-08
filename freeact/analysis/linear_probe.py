"""Linear decodability of camera-frame actions from frozen latents (Table 1 of the paper).

Ridge regression (RidgeCV over 20 log-spaced alphas, standardized X and y) from the flattened
post-quantization embeddings to the 7D action; R^2 is computed in the standardized space.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .paths import DEFAULT_OUT, load_features


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scaler: StandardScaler) -> dict:
    r2 = 1 - np.sum((y_true - y_pred) ** 2, 0) / np.maximum(np.sum((y_true - y_true.mean(0)) ** 2, 0), 1e-8)
    y_t, y_p = scaler.inverse_transform(y_true), scaler.inverse_transform(y_pred)
    nmse = np.mean((y_t - y_p) ** 2, 0) / np.maximum(scaler.var_, 1e-8)
    return {
        "r2": float(r2.mean()),
        "r2_pos": float(r2[:3].mean()),
        "r2_rot": float(r2[3:6].mean()),
        "r2_grip": float(r2[6]),
        "nmse": float(nmse.mean()),
        "r2_dim": r2.tolist(),
        "nmse_dim": nmse.tolist(),
    }


def probe(X_tr, y_tr, X_te, y_te) -> dict:
    xs, ys = StandardScaler(), StandardScaler()
    X_tr_s, X_te_s = xs.fit_transform(X_tr), xs.transform(X_te)
    y_tr_s, y_te_s = ys.fit_transform(y_tr), ys.transform(y_te)
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 20), cv=5, fit_intercept=False).fit(X_tr_s, y_tr_s)
    return {
        "alpha": float(ridge.alpha_),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "train": compute_metrics(y_tr_s, ridge.predict(X_tr_s), ys),
        "test": compute_metrics(y_te_s, ridge.predict(X_te_s), ys),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    results = {}
    for name in args.names:
        f = load_features(args.out_dir, name)
        results[name] = probe(f["X_tr"], f["y_tr"], f["X_te"], f["y_te"])

    rows = [("Total R2", "r2"), ("Translation R2", "r2_pos"), ("Rotation R2", "r2_rot"),
            ("Gripper R2", "r2_grip"), ("Norm. MSE", "nmse")]
    header = f"{'':<16}" + "".join(f"{n + ' train':>16}{n + ' test':>16}" for n in args.names)
    print(header + "\n" + "-" * len(header))
    for label, key in rows:
        print(f"{label:<16}" + "".join(
            f"{results[n]['train'][key]:>16.3f}{results[n]['test'][key]:>16.3f}" for n in args.names))

    out = args.out_dir / "linear_probe.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

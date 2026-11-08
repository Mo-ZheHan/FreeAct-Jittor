"""RQ1: structure of a latent action space on the held-out DROID split.

    python -m freeact.analysis.extract --lam <ckpt> --name freeact --data_root ./data
    python -m freeact.analysis.linear_probe --names freeact univla lapa
    python -m freeact.analysis.nldr --names freeact univla lapa
    python -m freeact.analysis.umap_plot --names lapa univla freeact

``extract`` samples frame pairs from the train / held-out splits of the stage-1 data pipeline
(identical episode split, ``val_split=0.002``, seed 42), tokenizes them with a frozen LAM and
stores the post-quantization embeddings (codebook lookup, flattened over the ``K`` codes) together
with the camera-frame 7D action.  The other scripts only read those features.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from freeact.analysis.paths import DEFAULT_OUT, feature_path
from freeact.data import MixedLeRobotDataset, Subset, collate_latent_action, stratified_split
from freeact.tokenizers import detect_backend, load_tokenizer
from freeact.utils.loader import DataLoader

# Calibrated DROID splits: the RQ1 evaluation set
DROID_CALIBRATED = [("droid/dual_cam", 1.0, True), ("droid/single_cam", 1.0, True)]


def _extract(tok, ds, indices, batch_size, num_workers):
    import jittor as jt

    loader = DataLoader(Subset(ds, indices), batch_size, list(range(len(indices))), collate_latent_action,
                        num_workers=num_workers)
    codebook = tok.codebook.float32().numpy()
    X, y, codes = [], [], []
    with jt.no_grad():
        for batch in tqdm(loader, desc="Extracting", leave=False):
            v = batch["videos"].astype(np.float32)
            idx = np.asarray(tok.encode_pairs(jt.array(v[:, 0]), jt.array(v[:, 1])))
            codes.append(idx)
            X.append(codebook[idx].reshape(idx.shape[0], -1))
            y.append(batch["action"])
    loader.close()
    return np.concatenate(X), np.concatenate(y), np.concatenate(codes)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lam", required=True, help="LAM checkpoint (FreeAct, UniVLA lam-stage-2.ckpt or LAPA laq_openx.pt)")
    p.add_argument("--name", required=True, help="tag used for the output file")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--num_train", type=int, default=50000)
    p.add_argument("--num_test", type=int, default=20000)
    p.add_argument("--delta_time", type=float, default=1.0)
    p.add_argument("--val_split", type=float, default=0.002, help="must match stage-1 training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=min(32, os.cpu_count() or 8))
    args = p.parse_args()
    if not Path(args.lam).is_file():
        p.error(f"--lam {args.lam} is not a file; pass a tokenizer checkpoint, e.g. checkpoints/FreeAct/lam/lam.ckpt")
    for split, _, _ in DROID_CALIBRATED:
        if not (Path(args.data_root) / split / "meta" / "info.json").is_file():
            p.error(
                f"{Path(args.data_root) / split} not found; download both calibrated DROID splits with "
                f"hf download Mo-ZheHan/droid --repo-type dataset --include '*_cam/*' --local-dir {Path(args.data_root) / 'droid'}"
            )

    import jittor as jt

    jt.flags.use_cuda = 1
    np.random.seed(args.seed)

    # LAPA reads frames at its native 256 px; the FreeAct and UniVLA tokenizers at 224 px
    resolution = 256 if detect_backend(args.lam) == "lapa" else 224
    try:
        full = MixedLeRobotDataset(DROID_CALIBRATED, resolution=resolution, seed=args.seed, root=args.data_root,
                                   delta_time=args.delta_time)
    except FileNotFoundError as e:  # partial download; --num_train / --num_test do not reduce what is needed
        p.error(f"{e}; extraction samples from every episode, so both splits must be complete (about 100 GB)")
    tr_ix, va_ix, _ = stratified_split(full, args.val_split, args.seed)
    train_ds, val_ds = Subset(full, tr_ix), Subset(full, va_ix)
    print(f"train {len(train_ds):,} / held-out {len(val_ds):,} frame pairs")

    rng = np.random.default_rng(args.seed)
    tr_idx = np.sort(rng.choice(len(train_ds), min(args.num_train, len(train_ds)), replace=False))
    te_idx = np.sort(rng.choice(len(val_ds), min(args.num_test, len(val_ds)), replace=False))

    tok = load_tokenizer(args.lam)
    X_tr, y_tr, c_tr = _extract(tok, train_ds, tr_idx, args.batch_size, args.num_workers)
    X_te, y_te, c_te = _extract(tok, val_ds, te_idx, args.batch_size, args.num_workers)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = feature_path(args.out_dir, args.name)
    np.savez(out, X_tr=X_tr, y_tr=y_tr, codes_tr=c_tr, X_te=X_te, y_te=y_te, codes_te=c_te)
    print(f"features {X_tr.shape} / {X_te.shape} -> {out}")


if __name__ == "__main__":
    main()

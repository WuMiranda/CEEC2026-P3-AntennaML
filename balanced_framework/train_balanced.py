from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.io import load_features_csv, load_labels_csv

from .metrics import confusion_matrix, metrics_dict_from_cm, macro_f1
from .models import MLP
from .plotting import save_confusion_matrix_png
from .preprocess import FeatureConfig, Standardizer, build_features
from .splits import SplitConfig, hierarchical_split_indices_within_label, stratified_split_indices


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_balanced_weights(counts: np.ndarray, beta: float) -> np.ndarray:
    counts = counts.astype(np.float64)
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.maximum(eff, 1e-12)
    w = w / np.maximum(w.mean(), 1e-12)
    return w.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="0514")
    p.add_argument("--x_csv", type=str, default="x_train.csv")
    p.add_argument("--y_csv", type=str, default="y_train.csv")
    p.add_argument("--out_dir", type=str, default="outputs/balanced_ce")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    p.add_argument("--hierarchical_col", type=str, default="itState")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--beta", type=float, default=0.999)
    p.add_argument("--tau", type=float, default=1.0)

    p.add_argument("--derived_features", action="store_true", default=True)
    p.add_argument("--no_derived_features", action="store_false", dest="derived_features")

    return p.parse_args()


def _to_numpy_int64(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    split_cfg = SplitConfig(
        train=float(args.split[0]),
        val=float(args.split[1]),
        test=float(args.split[2]),
        seed=int(args.seed),
        hierarchical_col=str(args.hierarchical_col) if args.hierarchical_col else None,
    )

    data_dir = Path(args.data_dir)
    x_path = data_dir / args.x_csv
    y_path = data_dir / args.y_csv

    features_df = load_features_csv(x_path)
    y = load_labels_csv(y_path)

    feat_cfg = FeatureConfig(derived=bool(args.derived_features))
    x, feat_meta = build_features(features_df, feat_cfg)

    num_classes = int(np.max(y)) + 1
    if split_cfg.hierarchical_col:
        h = _to_numpy_int64(features_df[split_cfg.hierarchical_col].to_numpy())
        tr_idx, va_idx, te_idx = hierarchical_split_indices_within_label(
            y=y,
            hierarchy=h,
            train=split_cfg.train,
            val=split_cfg.val,
            test=split_cfg.test,
            seed=split_cfg.seed,
        )
        split_mode = f"hierarchical_within_label:{split_cfg.hierarchical_col}"
    else:
        tr_idx, va_idx, te_idx = stratified_split_indices(
            y=y,
            train=split_cfg.train,
            val=split_cfg.val,
            test=split_cfg.test,
            seed=split_cfg.seed,
        )
        split_mode = "stratified_label"

    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_va, y_va = x[va_idx], y[va_idx]
    x_te, y_te = x[te_idx], y[te_idx]

    scaler = Standardizer().fit(x_tr)
    x_tr = scaler.transform(x_tr)
    x_va = scaler.transform(x_va)
    x_te = scaler.transform(x_te)

    counts = np.bincount(y_tr.astype(np.int64), minlength=num_classes).astype(np.float64)
    priors = counts / np.maximum(counts.sum(), 1.0)
    cb_w = class_balanced_weights(counts.astype(np.float32), beta=float(args.beta))

    sample_w = cb_w[y_tr.astype(np.int64)]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=int(sample_w.shape[0]),
        replacement=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tr_t = torch.tensor(x_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(x_tr_t, y_tr_t),
        batch_size=int(args.batch_size),
        sampler=sampler,
        drop_last=False,
    )

    x_va_t = torch.tensor(x_va, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.long, device=device)
    x_te_t = torch.tensor(x_te, dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.long, device=device)

    model = MLP(in_dim=int(x_tr.shape[1]), num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    class_weight_t = torch.tensor(cb_w, dtype=torch.float32, device=device)
    log_prior_t = torch.log(torch.tensor(priors, dtype=torch.float32, device=device) + 1e-12)
    tau = float(args.tau)
    adjust = (-tau) * log_prior_t

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = -1.0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits + adjust, yb, weight=class_weight_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_pred = (model(x_va_t) + adjust).argmax(dim=1).cpu().numpy()
        cm_va = confusion_matrix(y_va, va_pred, num_classes=num_classes)
        val_score = macro_f1(cm_va)

        if val_score > best_val:
            best_val = val_score
            best_state = {
                "model": model.state_dict(),
                "num_classes": num_classes,
                "in_dim": int(x_tr.shape[1]),
            }

        print(f"epoch={epoch} val_macro_f1={val_score:.4f}")

    if best_state is None:
        raise RuntimeError("Training failed: no checkpoint")

    torch.save(best_state, out_dir / "model.pt")

    np.savez(
        out_dir / "preprocess.npz",
        **scaler.state_dict(),
        priors=priors.astype(np.float32),
        tau=np.array([tau], dtype=np.float32),
        it_num=np.array([feat_meta["it_num"]], dtype=np.int64),
        derived=np.array([1 if feat_meta["derived"] else 0], dtype=np.int64),
        feature_dim=np.array([feat_meta["feature_dim"]], dtype=np.int64),
    )

    model.load_state_dict(best_state["model"])
    model.eval()
    with torch.no_grad():
        va_pred = (model(x_va_t) + adjust).argmax(dim=1).cpu().numpy()
        te_pred = (model(x_te_t) + adjust).argmax(dim=1).cpu().numpy()

    cm_va = confusion_matrix(y_va, va_pred, num_classes=num_classes)
    cm_te = confusion_matrix(y_te, te_pred, num_classes=num_classes)

    save_confusion_matrix_png(cm_va, out_dir / "confusion_val.png", title="Confusion Matrix (Val)")
    save_confusion_matrix_png(cm_te, out_dir / "confusion_test.png", title="Confusion Matrix (Test)")

    metrics = {
        "val": metrics_dict_from_cm(cm_va),
        "test": metrics_dict_from_cm(cm_te),
        "split_mode": split_mode,
        "class_counts_train": {str(i): int(v) for i, v in enumerate(counts.tolist())},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    cfg = {
        "data_dir": str(data_dir),
        "x_csv": args.x_csv,
        "y_csv": args.y_csv,
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "split": [float(args.split[0]), float(args.split[1]), float(args.split[2])],
        "split_mode": split_mode,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "beta": float(args.beta),
        "tau": float(args.tau),
        "derived_features": bool(args.derived_features),
        "feature_dim": int(feat_meta["feature_dim"]),
        "device_auto": str(device),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


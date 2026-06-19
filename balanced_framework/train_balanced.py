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

from .geometry import build_residual_features, build_residual_summary_features, circle_params_to_table, fit_label_circles
from .freq_tree import apply_frequency_tree, fit_frequency_tree, tree_to_dict
from .metrics import confusion_matrix, metrics_dict_from_cm, macro_f1
from .models import FrequencyGatedMLP, MLP
from .plotting import save_confusion_matrix_png
from .preprocess import (
    ABLATION_FEATURES,
    FeatureConfig,
    Standardizer,
    build_features,
    compute_log_magnitude_ratio,
    fit_freq_bin_edges,
    fit_quantile_clip_bounds,
    freq_to_bin_index,
    freq_to_bin_onehot,
)
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
    counts = np.maximum(counts.astype(np.float64), 1.0)
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
    p.add_argument("--hidden_dims", type=int, nargs="+", default=(128, 64, 32))

    p.add_argument("--beta", type=float, default=0.999)
    p.add_argument("--tau", type=float, default=1.0)

    p.add_argument("--derived_features", action="store_true", default=True)
    p.add_argument("--no_derived_features", action="store_false", dest="derived_features")

    p.add_argument("--feature_set", type=str, default="full", choices=("full", "phys_min"))
    p.add_argument("--it_encoding", type=str, default="bits", choices=("bits", "onehot"))
    p.add_argument("--no_itstate_features", action="store_false", dest="use_itstate_features", default=True)
    p.add_argument("--freq_bins", type=int, default=10)
    p.add_argument("--no_freq_bins", action="store_true", default=False)
    p.add_argument("--freq_bin_mode", type=str, default="onehot", choices=("onehot", "index", "none"))

    p.add_argument("--use_closed_state_interactions", action="store_true", default=False)
    p.add_argument("--use_impedance_features", action="store_true", default=False)
    p.add_argument("--rf_features", choices=("none", "vswr", "rl", "vswr_rl"), default="none")
    p.add_argument("--vswr_clip", type=float, default=100.0)
    p.add_argument("--z0", type=float, default=50.0)
    p.add_argument("--gamma_scale", type=float, default=1.0)
    p.add_argument("--exclude_features", nargs="*", choices=ABLATION_FEATURES, default=())
    p.add_argument("--split_indices_dir", type=str, default=None)

    p.add_argument("--use_log_magnitude_ratio", action="store_true", default=False)
    p.add_argument("--log_ratio_eps", type=float, default=1e-6)
    p.add_argument("--log_ratio_lower_quantile", type=float, default=0.001)
    p.add_argument("--log_ratio_upper_quantile", type=float, default=0.999)

    p.add_argument("--use_geometry_features", action="store_true", default=False)
    p.add_argument("--geometry_mode", type=str, default="none", choices=("none", "full", "summary"))

    p.add_argument("--freq_tree_depth", type=int, default=0)
    p.add_argument("--freq_tree_min_leaf", type=int, default=200)
    p.add_argument("--freq_tree_candidates", type=int, default=128)

    return p.parse_args()


def _to_numpy_int64(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)


def _validate_split_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    n_samples: int,
) -> None:
    parts = [np.asarray(v, dtype=np.int64).reshape(-1) for v in (train_idx, val_idx, test_idx)]
    combined = np.concatenate(parts)
    if combined.size != n_samples:
        raise ValueError(f"Fixed split has {combined.size} indices, expected {n_samples}")
    if combined.size and (int(combined.min()) < 0 or int(combined.max()) >= n_samples):
        raise ValueError("Fixed split contains an out-of-range index")
    if np.unique(combined).size != n_samples:
        raise ValueError("Fixed split indices overlap or do not cover every sample exactly once")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    hidden_dims = tuple(int(v) for v in args.hidden_dims)

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_df = load_features_csv(x_path)
    y = load_labels_csv(y_path)

    feat_cfg = FeatureConfig(
        derived=bool(args.derived_features),
        it_encoding=str(args.it_encoding),
        feature_set=str(args.feature_set),
        use_itstate_features=bool(args.use_itstate_features),
        use_closed_state_interactions=bool(args.use_closed_state_interactions),
        use_impedance_features=bool(args.use_impedance_features),
        rf_features=str(args.rf_features),
        vswr_clip=float(args.vswr_clip),
        exclude_features=tuple(str(v) for v in args.exclude_features),
        z0=float(args.z0),
        gamma_scale=float(args.gamma_scale),
    )
    x, feat_meta, scale_mask = build_features(features_df, feat_cfg)

    num_classes = int(np.max(y)) + 1

    split_indices_dir = Path(args.split_indices_dir) if args.split_indices_dir else None
    split_paths = None if split_indices_dir is None else {
        "train": split_indices_dir / "train_idx.npy",
        "val": split_indices_dir / "val_idx.npy",
        "test": split_indices_dir / "test_idx.npy",
    }
    if split_paths is not None and all(path.exists() for path in split_paths.values()):
        tr_idx = np.load(split_paths["train"]).astype(np.int64)
        va_idx = np.load(split_paths["val"]).astype(np.int64)
        te_idx = np.load(split_paths["test"]).astype(np.int64)
        _validate_split_indices(tr_idx, va_idx, te_idx, n_samples=y.shape[0])
        split_mode = f"fixed_indices:{split_indices_dir}"
    else:
        if split_paths is not None and any(path.exists() for path in split_paths.values()):
            raise FileNotFoundError(f"Incomplete fixed split files in {split_indices_dir}")
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
        _validate_split_indices(tr_idx, va_idx, te_idx, n_samples=y.shape[0])
        if split_paths is not None:
            split_indices_dir.mkdir(parents=True, exist_ok=True)
            np.save(split_paths["train"], tr_idx.astype(np.int64))
            np.save(split_paths["val"], va_idx.astype(np.int64))
            np.save(split_paths["test"], te_idx.astype(np.int64))

    np.save(out_dir / "train_idx.npy", tr_idx.astype(np.int64))
    np.save(out_dir / "val_idx.npy", va_idx.astype(np.int64))
    np.save(out_dir / "test_idx.npy", te_idx.astype(np.int64))

    log_ratio_lower: float | None = None
    log_ratio_upper: float | None = None
    if bool(args.use_log_magnitude_ratio):
        log_ratio_raw = compute_log_magnitude_ratio(
            features_df,
            gamma_scale=float(args.gamma_scale),
            eps=float(args.log_ratio_eps),
        )
        log_ratio_lower, log_ratio_upper = fit_quantile_clip_bounds(
            log_ratio_raw[tr_idx],
            lower_quantile=float(args.log_ratio_lower_quantile),
            upper_quantile=float(args.log_ratio_upper_quantile),
        )
        log_ratio_clipped = np.clip(log_ratio_raw, log_ratio_lower, log_ratio_upper).astype(np.float32)
        x = np.concatenate([x, log_ratio_clipped[:, None]], axis=1)
        scale_mask = np.concatenate([scale_mask, np.ones((1,), dtype=bool)], axis=0)

    freq_all = features_df["closeFreqMHz"].to_numpy(np.float32)
    freq_bin_mode = str(args.freq_bin_mode).lower().strip()
    if bool(args.no_freq_bins):
        freq_bin_mode = "none"

    if freq_bin_mode == "none" or int(args.freq_bins) <= 0:
        freq_edges = np.array([], dtype=np.float32)
        freq_bins_feat = np.zeros((x.shape[0], 0), dtype=np.float32)
        freq_bins_dim = 0
        freq_bins_kind = "none"
    else:
        freq_edges = fit_freq_bin_edges(freq_all[tr_idx], bins=int(args.freq_bins))
        if freq_bin_mode == "onehot":
            freq_bins_feat = freq_to_bin_onehot(freq_all, freq_edges)
            freq_bins_dim = int(freq_bins_feat.shape[1])
            freq_bins_kind = "onehot"
        elif freq_bin_mode == "index":
            freq_bins_feat = freq_to_bin_index(freq_all, freq_edges)
            freq_bins_dim = int(freq_bins_feat.shape[1])
            freq_bins_kind = "index"
        else:
            raise ValueError(f"Unknown freq_bin_mode: {freq_bin_mode}")

    if freq_bins_feat.shape[1] > 0:
        x = np.concatenate([x, freq_bins_feat.astype(np.float32)], axis=1)
        if freq_bins_kind == "onehot":
            scale_mask = np.concatenate([scale_mask, np.zeros((freq_bins_feat.shape[1],), dtype=bool)], axis=0)
        else:
            scale_mask = np.concatenate([scale_mask, np.zeros((freq_bins_feat.shape[1],), dtype=bool)], axis=0)

    geometry_mode = str(args.geometry_mode).lower().strip()
    if bool(args.use_geometry_features):
        geometry_mode = "full"

    if geometry_mode != "none":
        g1_re = features_df["gammaIn1Re"].to_numpy(np.float32)
        g1_im = features_df["gammaIn1Im"].to_numpy(np.float32)
        g2_re = features_df["gammaIn2Re"].to_numpy(np.float32)
        g2_im = features_df["gammaIn2Im"].to_numpy(np.float32)

        params = fit_label_circles(
            g1_re=g1_re[tr_idx],
            g1_im=g1_im[tr_idx],
            g2_re=g2_re[tr_idx],
            g2_im=g2_im[tr_idx],
            y=y[tr_idx],
        )
        if geometry_mode == "full":
            geom = build_residual_features(g1_re=g1_re, g1_im=g1_im, g2_re=g2_re, g2_im=g2_im, params=params)
            x = np.concatenate([x, geom.astype(np.float32)], axis=1)
            scale_mask = np.concatenate([scale_mask, np.ones((geom.shape[1],), dtype=bool)], axis=0)
        elif geometry_mode == "summary":
            geom = build_residual_summary_features(g1_re=g1_re, g1_im=g1_im, g2_re=g2_re, g2_im=g2_im, params=params)
            x = np.concatenate([x, geom.astype(np.float32)], axis=1)
            scale_mask = np.concatenate([scale_mask, np.ones((geom.shape[1],), dtype=bool)], axis=0)
        else:
            raise ValueError(f"Unknown geometry_mode: {geometry_mode}")

        header = "label\tn_train\tg1_cx\tg1_cy\tg1_r\tg2_cx\tg2_cy\tg2_r\n"
        table = circle_params_to_table(params)
        lines = [header] + ["\t".join([str(int(r[0])), str(int(r[1]))] + [f"{v:.10f}" for v in r[2:]]) + "\n" for r in table]
        (out_dir / "circle_params.tsv").write_text("".join(lines), encoding="utf-8")

    feat_meta = dict(feat_meta)
    feat_meta["feature_dim"] = int(x.shape[1])
    feat_meta["freq_bins"] = int(freq_bins_dim)
    feat_meta["freq_bin_mode"] = str(freq_bins_kind)
    feat_meta["geometry_mode"] = str(geometry_mode)
    feat_meta["use_log_magnitude_ratio"] = bool(args.use_log_magnitude_ratio)
    feat_meta["gamma_scale"] = float(args.gamma_scale)
    feat_meta["split_indices_dir"] = None if split_indices_dir is None else str(split_indices_dir)

    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_va, y_va = x[va_idx], y[va_idx]
    x_te, y_te = x[te_idx], y[te_idx]

    freq_tree_depth = int(args.freq_tree_depth)
    if freq_tree_depth < 0:
        raise ValueError("--freq_tree_depth must be >= 0")

    if freq_tree_depth > 0:
        tree = fit_frequency_tree(
            freq_train=freq_all[tr_idx],
            y_train=y_tr,
            num_classes=num_classes,
            max_depth=freq_tree_depth,
            min_leaf=int(args.freq_tree_min_leaf),
            max_candidates=int(args.freq_tree_candidates),
        )
        (out_dir / "freq_tree.json").write_text(json.dumps(tree_to_dict(tree), ensure_ascii=False, indent=2), encoding="utf-8")
        leaf_all = apply_frequency_tree(freq_all, tree)
        leaf_tr = leaf_all[tr_idx].astype(np.int64)
        leaf_va = leaf_all[va_idx].astype(np.int64)
        leaf_te = leaf_all[te_idx].astype(np.int64)
        num_leaves = int(np.max(leaf_tr)) + 1 if leaf_tr.size > 0 else int(tree.num_leaves)
    else:
        tree = None
        leaf_tr = None
        leaf_va = None
        leaf_te = None
        num_leaves = 0

    scaler = Standardizer().fit(x_tr, scale_mask=scale_mask)
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
        generator=torch.Generator().manual_seed(int(args.seed)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tr_t = torch.tensor(x_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    if freq_tree_depth > 0:
        leaf_tr_t = torch.tensor(leaf_tr, dtype=torch.long)
        train_ds = TensorDataset(x_tr_t, y_tr_t, leaf_tr_t)
    else:
        train_ds = TensorDataset(x_tr_t, y_tr_t)

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), sampler=sampler, drop_last=False)

    x_va_t = torch.tensor(x_va, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.long, device=device)
    x_te_t = torch.tensor(x_te, dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.long, device=device)

    if freq_tree_depth > 0:
        leaf_va_t = torch.tensor(leaf_va, dtype=torch.long, device=device)
        leaf_te_t = torch.tensor(leaf_te, dtype=torch.long, device=device)
        model = FrequencyGatedMLP(
            in_dim=int(x_tr.shape[1]),
            num_classes=num_classes,
            num_leaves=num_leaves,
            hidden_dims=hidden_dims,
        ).to(device)
    else:
        leaf_va_t = None
        leaf_te_t = None
        model = MLP(in_dim=int(x_tr.shape[1]), num_classes=num_classes, hidden_dims=hidden_dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    tau = float(args.tau)

    if freq_tree_depth > 0:
        leaf_counts = np.zeros((num_leaves, num_classes), dtype=np.float64)
        for lid in range(num_leaves):
            m = leaf_tr == lid
            if not np.any(m):
                leaf_counts[lid] = counts
            else:
                leaf_counts[lid] = np.bincount(y_tr[m].astype(np.int64), minlength=num_classes).astype(np.float64)

        alpha = 1.0
        leaf_priors = (leaf_counts + alpha) / np.maximum(leaf_counts.sum(axis=1, keepdims=True) + alpha * num_classes, 1.0)
        leaf_cb_w = np.stack([class_balanced_weights(leaf_counts[lid].astype(np.float32), beta=float(args.beta)) for lid in range(num_leaves)], axis=0)

        leaf_adjust = (-tau) * np.log(leaf_priors.astype(np.float32) + 1e-12)
        leaf_adjust_t = torch.tensor(leaf_adjust, dtype=torch.float32, device=device)
        leaf_class_weight_t = torch.tensor(leaf_cb_w, dtype=torch.float32, device=device)
        adjust = None
        class_weight_t = None
    else:
        class_weight_t = torch.tensor(cb_w, dtype=torch.float32, device=device)
        log_prior_t = torch.log(torch.tensor(priors, dtype=torch.float32, device=device) + 1e-12)
        adjust = (-tau) * log_prior_t
        leaf_adjust_t = None
        leaf_class_weight_t = None

    best_val = -1.0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for batch in train_loader:
            if freq_tree_depth > 0:
                xb, yb, lb = batch
                lb = lb.to(device)
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb, lb)
                adj_b = leaf_adjust_t[lb]
                logp = F.log_softmax(logits + adj_b, dim=1)
                nll = -logp[torch.arange(yb.shape[0], device=device), yb]
                sw = leaf_class_weight_t[lb, yb]
                loss = (sw * nll).mean()
            else:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits + adjust, yb, weight=class_weight_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            if freq_tree_depth > 0:
                va_logits = model(x_va_t, leaf_va_t) + leaf_adjust_t[leaf_va_t]
                va_pred = va_logits.argmax(dim=1).cpu().numpy()
            else:
                va_pred = (model(x_va_t) + adjust).argmax(dim=1).cpu().numpy()
        cm_va = confusion_matrix(y_va, va_pred, num_classes=num_classes)
        val_score = macro_f1(cm_va)

        if val_score > best_val:
            best_val = val_score
            best_state = {
                "model": model.state_dict(),
                "num_classes": num_classes,
                "in_dim": int(x_tr.shape[1]),
                "hidden_dims": hidden_dims,
                "freq_tree_depth": freq_tree_depth,
                "num_leaves": int(num_leaves),
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
        it_encoding=np.array([0 if feat_meta["it_encoding"] == "bits" else 1], dtype=np.int64),
        derived=np.array([1 if feat_meta["derived"] else 0], dtype=np.int64),
        feature_dim=np.array([feat_meta["feature_dim"]], dtype=np.int64),
        freq_bin_edges=freq_edges.astype(np.float32),
        freq_bins=np.array([feat_meta["freq_bins"]], dtype=np.int64),
        freq_bin_mode=np.array([0 if feat_meta["freq_bin_mode"] == "none" else (1 if feat_meta["freq_bin_mode"] == "index" else 2)], dtype=np.int64),
        geometry_mode=np.array([0 if feat_meta["geometry_mode"] == "none" else (1 if feat_meta["geometry_mode"] == "summary" else 2)], dtype=np.int64),
        use_itstate_features=np.array([1 if feat_meta.get("use_itstate_features", True) else 0], dtype=np.int64),
        use_closed_state_interactions=np.array([1 if feat_meta.get("use_closed_state_interactions", False) else 0], dtype=np.int64),
        use_impedance_features=np.array([1 if feat_meta.get("use_impedance_features", False) else 0], dtype=np.int64),
        rf_features=np.array([feat_meta.get("rf_features", "none")]),
        vswr_clip=np.array([feat_meta.get("vswr_clip", 100.0)], dtype=np.float32),
        exclude_features=np.array(feat_meta.get("exclude_features", [])),
        z0=np.array([feat_meta.get("z0", 50.0)], dtype=np.float32),
        gamma_scale=np.array([float(args.gamma_scale)], dtype=np.float32),
        use_log_magnitude_ratio=np.array([1 if bool(args.use_log_magnitude_ratio) else 0], dtype=np.int64),
        log_ratio_eps=np.array([float(args.log_ratio_eps)], dtype=np.float32),
        log_ratio_lower_quantile=np.array([float(args.log_ratio_lower_quantile)], dtype=np.float32),
        log_ratio_upper_quantile=np.array([float(args.log_ratio_upper_quantile)], dtype=np.float32),
        log_ratio_lower=np.array([] if log_ratio_lower is None else [log_ratio_lower], dtype=np.float32),
        log_ratio_upper=np.array([] if log_ratio_upper is None else [log_ratio_upper], dtype=np.float32),
        freq_tree_depth=np.array([freq_tree_depth], dtype=np.int64),
        freq_tree_threshold=np.array([] if tree is None else tree.threshold, dtype=np.float32),
        freq_tree_left=np.array([] if tree is None else tree.left, dtype=np.int64),
        freq_tree_right=np.array([] if tree is None else tree.right, dtype=np.int64),
        freq_tree_is_leaf=np.array([] if tree is None else tree.is_leaf, dtype=np.int64),
        freq_tree_leaf_id=np.array([] if tree is None else tree.leaf_id, dtype=np.int64),
        freq_tree_num_leaves=np.array([0 if tree is None else int(tree.num_leaves)], dtype=np.int64),
        leaf_priors=np.array([] if freq_tree_depth == 0 else leaf_priors.astype(np.float32)),
    )

    model.load_state_dict(best_state["model"])
    model.eval()
    with torch.no_grad():
        if freq_tree_depth > 0:
            va_logits = model(x_va_t, leaf_va_t) + leaf_adjust_t[leaf_va_t]
            te_logits = model(x_te_t, leaf_te_t) + leaf_adjust_t[leaf_te_t]
            va_pred = va_logits.argmax(dim=1).cpu().numpy()
            te_pred = te_logits.argmax(dim=1).cpu().numpy()
        else:
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
        "hidden_dims": [int(v) for v in hidden_dims],
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "beta": float(args.beta),
        "tau": float(args.tau),
        "derived_features": bool(args.derived_features),
        "feature_set": str(args.feature_set),
        "it_encoding": str(args.it_encoding),
        "use_itstate_features": bool(args.use_itstate_features),
        "freq_bins": 0 if freq_bin_mode == "none" else int(args.freq_bins),
        "freq_bin_mode": str(freq_bin_mode),
        "use_closed_state_interactions": bool(args.use_closed_state_interactions),
        "use_impedance_features": bool(args.use_impedance_features),
        "rf_features": str(args.rf_features),
        "vswr_clip": float(args.vswr_clip),
        "exclude_features": [str(v) for v in args.exclude_features],
        "use_log_magnitude_ratio": bool(args.use_log_magnitude_ratio),
        "log_ratio_eps": float(args.log_ratio_eps),
        "log_ratio_lower_quantile": float(args.log_ratio_lower_quantile),
        "log_ratio_upper_quantile": float(args.log_ratio_upper_quantile),
        "log_ratio_lower": log_ratio_lower,
        "log_ratio_upper": log_ratio_upper,
        "z0": float(args.z0),
        "gamma_scale": float(args.gamma_scale),
        "geometry_mode": str(geometry_mode),
        "freq_tree_depth": freq_tree_depth,
        "freq_tree_min_leaf": int(args.freq_tree_min_leaf),
        "freq_tree_candidates": int(args.freq_tree_candidates),
        "feature_dim": int(feat_meta["feature_dim"]),
        "split_indices_dir": None if split_indices_dir is None else str(split_indices_dir),
        "device_auto": str(device),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


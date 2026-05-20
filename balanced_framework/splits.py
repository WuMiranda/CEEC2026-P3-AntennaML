from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.6
    val: float = 0.2
    test: float = 0.2
    seed: int = 42
    hierarchical_col: str | None = "itState"


def _normalize_split(train: float, val: float, test: float) -> tuple[float, float, float]:
    s = train + val + test
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"split must sum to 1.0, got {s}")
    if min(train, val, test) < 0:
        raise ValueError("split ratios must be non-negative")
    return train, val, test


def stratified_split_indices(
    y: np.ndarray,
    train: float,
    val: float,
    test: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train, val, test = _normalize_split(train, val, test)
    rng = np.random.default_rng(seed)

    classes = np.unique(y)
    tr, va, te = [], [], []
    for c in classes.tolist():
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = idx.shape[0]
        n_tr = int(round(n * train))
        n_va = int(round(n * val))
        n_te = n - n_tr - n_va
        if n_te < 0:
            n_te = 0
        tr.append(idx[:n_tr])
        va.append(idx[n_tr : n_tr + n_va])
        te.append(idx[n_tr + n_va : n_tr + n_va + n_te])

    tr = np.concatenate(tr) if tr else np.array([], dtype=np.int64)
    va = np.concatenate(va) if va else np.array([], dtype=np.int64)
    te = np.concatenate(te) if te else np.array([], dtype=np.int64)
    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te


def hierarchical_split_indices_within_label(
    y: np.ndarray,
    hierarchy: np.ndarray,
    train: float,
    val: float,
    test: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train, val, test = _normalize_split(train, val, test)
    rng = np.random.default_rng(seed)

    classes = np.unique(y)
    tr, va, te = [], [], []
    for c in classes.tolist():
        idx_c = np.where(y == c)[0]
        h_vals = np.unique(hierarchy[idx_c])
        for hv in h_vals.tolist():
            idx = idx_c[hierarchy[idx_c] == hv]
            idx = idx.copy()
            rng.shuffle(idx)
            n = idx.shape[0]
            if n < 3:
                tr.append(idx)
                continue
            n_tr = int(round(n * train))
            n_va = int(round(n * val))
            n_tr = max(n_tr, 1)
            n_va = max(n_va, 0)
            if n_tr + n_va >= n:
                n_tr = max(n - 1, 1)
                n_va = n - n_tr
            n_te = n - n_tr - n_va
            tr.append(idx[:n_tr])
            va.append(idx[n_tr : n_tr + n_va])
            te.append(idx[n_tr + n_va : n_tr + n_va + n_te])

    tr = np.concatenate(tr) if tr else np.array([], dtype=np.int64)
    va = np.concatenate(va) if va else np.array([], dtype=np.int64)
    te = np.concatenate(te) if te else np.array([], dtype=np.int64)
    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te


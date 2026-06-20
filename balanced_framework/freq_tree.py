from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrequencyTree:
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    is_leaf: np.ndarray
    leaf_id: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.threshold.shape[0])

    @property
    def num_leaves(self) -> int:
        if self.leaf_id.size == 0:
            return 0
        return int(self.leaf_id.max()) + 1


def _gini(y: np.ndarray, num_classes: int) -> float:
    if y.size == 0:
        return 0.0
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    p = counts / np.maximum(counts.sum(), 1.0)
    return float(1.0 - np.sum(p * p))


def _candidate_thresholds(x: np.ndarray, max_candidates: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    xs = np.sort(x)
    if xs.size <= 2:
        return np.array([], dtype=np.float32)
    if max_candidates <= 0:
        return np.array([], dtype=np.float32)
    if xs.size <= max_candidates + 2:
        mids = (xs[:-1] + xs[1:]) / 2.0
        return np.unique(mids.astype(np.float32))

    qs = np.linspace(0.02, 0.98, max_candidates, dtype=np.float32)
    cand = np.quantile(xs, qs, method="linear").astype(np.float32)
    cand = np.unique(cand)
    return cand


def fit_frequency_tree(
    freq_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    max_depth: int,
    min_leaf: int,
    max_candidates: int = 128,
) -> FrequencyTree:
    freq_train = np.asarray(freq_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    if freq_train.shape[0] != y_train.shape[0]:
        raise ValueError("freq_train/y_train length mismatch")
    if max_depth <= 0:
        threshold = np.array([0.0], dtype=np.float32)
        left = np.array([-1], dtype=np.int64)
        right = np.array([-1], dtype=np.int64)
        is_leaf = np.array([1], dtype=np.int64)
        leaf_id = np.array([0], dtype=np.int64)
        return FrequencyTree(threshold=threshold, left=left, right=right, is_leaf=is_leaf, leaf_id=leaf_id)

    thresholds: list[float] = []
    lefts: list[int] = []
    rights: list[int] = []
    is_leafs: list[int] = []
    leaf_ids: list[int] = []

    def new_node() -> int:
        idx = len(thresholds)
        thresholds.append(0.0)
        lefts.append(-1)
        rights.append(-1)
        is_leafs.append(0)
        leaf_ids.append(-1)
        return idx

    def build(indices: np.ndarray, depth: int) -> int:
        idx = new_node()
        y_sub = y_train[indices]
        if depth <= 0 or indices.size < 2 * min_leaf or np.unique(y_sub).size <= 1:
            is_leafs[idx] = 1
            return idx

        x_sub = freq_train[indices]
        cand = _candidate_thresholds(x_sub, max_candidates=max_candidates)
        if cand.size == 0:
            is_leafs[idx] = 1
            return idx

        base_imp = _gini(y_sub, num_classes=num_classes)
        best_thr = None
        best_score = base_imp
        best_left = None
        best_right = None

        for thr in cand.tolist():
            m_left = x_sub <= float(thr)
            n_left = int(m_left.sum())
            n_right = int(indices.size - n_left)
            if n_left < min_leaf or n_right < min_leaf:
                continue
            y_left = y_sub[m_left]
            y_right = y_sub[~m_left]
            score = (n_left * _gini(y_left, num_classes) + n_right * _gini(y_right, num_classes)) / float(indices.size)
            if score < best_score:
                best_score = score
                best_thr = float(thr)
                best_left = indices[m_left]
                best_right = indices[~m_left]

        if best_thr is None:
            is_leafs[idx] = 1
            return idx

        thresholds[idx] = best_thr
        lefts[idx] = build(best_left, depth - 1)
        rights[idx] = build(best_right, depth - 1)
        return idx

    root = build(np.arange(freq_train.shape[0], dtype=np.int64), depth=max_depth)
    if root != 0:
        raise RuntimeError("Internal error: root index must be 0")

    leaf_counter = 0
    for i in range(len(thresholds)):
        if is_leafs[i] == 1:
            leaf_ids[i] = leaf_counter
            leaf_counter += 1

    return FrequencyTree(
        threshold=np.asarray(thresholds, dtype=np.float32),
        left=np.asarray(lefts, dtype=np.int64),
        right=np.asarray(rights, dtype=np.int64),
        is_leaf=np.asarray(is_leafs, dtype=np.int64),
        leaf_id=np.asarray(leaf_ids, dtype=np.int64),
    )


def apply_frequency_tree(freq: np.ndarray, tree: FrequencyTree) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float32)
    out = np.zeros((freq.shape[0],), dtype=np.int64)
    for i in range(freq.shape[0]):
        node = 0
        while tree.is_leaf[node] == 0:
            thr = float(tree.threshold[node])
            node = int(tree.left[node]) if float(freq[i]) <= thr else int(tree.right[node])
        out[i] = int(tree.leaf_id[node])
    return out


def tree_to_dict(tree: FrequencyTree) -> dict:
    return {
        "threshold": tree.threshold.tolist(),
        "left": tree.left.tolist(),
        "right": tree.right.tolist(),
        "is_leaf": tree.is_leaf.tolist(),
        "leaf_id": tree.leaf_id.tolist(),
        "num_leaves": int(tree.num_leaves),
    }


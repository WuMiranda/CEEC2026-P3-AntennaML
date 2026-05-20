from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm


def accuracy_from_cm(cm: np.ndarray) -> float:
    denom = cm.sum()
    if denom == 0:
        return 0.0
    return float(np.diag(cm).sum() / denom)


def per_class_prf_from_cm(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.float64)

    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1, support


def macro_f1(cm: np.ndarray) -> float:
    _, _, f1, _ = per_class_prf_from_cm(cm)
    return float(np.nanmean(f1))


def weighted_f1(cm: np.ndarray) -> float:
    _, _, f1, support = per_class_prf_from_cm(cm)
    denom = support.sum()
    if denom == 0:
        return 0.0
    return float((f1 * support).sum() / denom)


def metrics_dict_from_cm(cm: np.ndarray) -> dict:
    precision, recall, f1, support = per_class_prf_from_cm(cm)
    return {
        "accuracy": accuracy_from_cm(cm),
        "macro_f1": macro_f1(cm),
        "weighted_f1": weighted_f1(cm),
        "per_class": {
            str(i): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(cm.shape[0])
        },
    }


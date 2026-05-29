from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircleParams:
    labels: np.ndarray
    g1_cx: np.ndarray
    g1_cy: np.ndarray
    g1_r: np.ndarray
    g2_cx: np.ndarray
    g2_cy: np.ndarray
    g2_r: np.ndarray
    n_train: np.ndarray


def fit_circle_least_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] < 3:
        return 0.0, 0.0, 0.0
    lhs = np.column_stack([x, y, np.ones_like(x)])
    rhs = -(x**2 + y**2)
    A, B, C = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    cx = -A / 2.0
    cy = -B / 2.0
    radius_sq = (A**2 + B**2) / 4.0 - C
    radius = math.sqrt(max(radius_sq, 0.0))
    return float(cx), float(cy), float(radius)


def fit_label_circles(
    g1_re: np.ndarray,
    g1_im: np.ndarray,
    g2_re: np.ndarray,
    g2_im: np.ndarray,
    y: np.ndarray,
) -> CircleParams:
    y = np.asarray(y, dtype=np.int64)
    labels = np.unique(y)
    labels = np.sort(labels)

    g1_cx = np.zeros((labels.shape[0],), dtype=np.float64)
    g1_cy = np.zeros((labels.shape[0],), dtype=np.float64)
    g1_r = np.zeros((labels.shape[0],), dtype=np.float64)
    g2_cx = np.zeros((labels.shape[0],), dtype=np.float64)
    g2_cy = np.zeros((labels.shape[0],), dtype=np.float64)
    g2_r = np.zeros((labels.shape[0],), dtype=np.float64)
    n_train = np.zeros((labels.shape[0],), dtype=np.int64)

    for i, lb in enumerate(labels.tolist()):
        m = y == int(lb)
        n_train[i] = int(m.sum())
        cx1, cy1, r1 = fit_circle_least_squares(g1_re[m], g1_im[m])
        cx2, cy2, r2 = fit_circle_least_squares(g2_re[m], g2_im[m])
        g1_cx[i], g1_cy[i], g1_r[i] = cx1, cy1, r1
        g2_cx[i], g2_cy[i], g2_r[i] = cx2, cy2, r2

    return CircleParams(
        labels=labels.astype(np.int64),
        g1_cx=g1_cx.astype(np.float32),
        g1_cy=g1_cy.astype(np.float32),
        g1_r=g1_r.astype(np.float32),
        g2_cx=g2_cx.astype(np.float32),
        g2_cy=g2_cy.astype(np.float32),
        g2_r=g2_r.astype(np.float32),
        n_train=n_train.astype(np.int64),
    )


def circle_params_to_table(params: CircleParams) -> np.ndarray:
    return np.column_stack(
        [
            params.labels,
            params.n_train,
            params.g1_cx,
            params.g1_cy,
            params.g1_r,
            params.g2_cx,
            params.g2_cy,
            params.g2_r,
        ]
    )


def build_residual_features(
    g1_re: np.ndarray,
    g1_im: np.ndarray,
    g2_re: np.ndarray,
    g2_im: np.ndarray,
    params: CircleParams,
) -> np.ndarray:
    g1_re = np.asarray(g1_re, dtype=np.float32)
    g1_im = np.asarray(g1_im, dtype=np.float32)
    g2_re = np.asarray(g2_re, dtype=np.float32)
    g2_im = np.asarray(g2_im, dtype=np.float32)

    n = g1_re.shape[0]
    k = params.labels.shape[0]
    out = np.zeros((n, 2 * k + 1), dtype=np.float32)

    for j in range(k):
        dx = g1_re - params.g1_cx[j]
        dy = g1_im - params.g1_cy[j]
        dist = np.sqrt(dx * dx + dy * dy)
        out[:, j] = np.abs(dist - params.g1_r[j])

    off = k
    for j in range(k):
        dx = g2_re - params.g2_cx[j]
        dy = g2_im - params.g2_cy[j]
        dist = np.sqrt(dx * dx + dy * dy)
        out[:, off + j] = np.abs(dist - params.g2_r[j])

    cross = g1_re * g2_im - g1_im * g2_re
    dot = g1_re * g2_re + g1_im * g2_im
    out[:, -1] = np.arctan2(cross, dot).astype(np.float32)
    return out


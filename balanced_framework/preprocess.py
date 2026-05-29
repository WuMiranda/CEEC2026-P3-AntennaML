from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    derived: bool = True
    it_encoding: str = "bits"


class Standardizer:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.scale_mask: np.ndarray | None = None

    def fit(self, x: np.ndarray, scale_mask: np.ndarray | None = None) -> "Standardizer":
        x = np.asarray(x, dtype=np.float32)
        if scale_mask is None:
            scale_mask = np.ones((x.shape[1],), dtype=bool)
        scale_mask = np.asarray(scale_mask, dtype=bool)
        if scale_mask.shape != (x.shape[1],):
            raise ValueError(f"scale_mask shape mismatch: expected={(x.shape[1],)}, got={scale_mask.shape}")

        mean = np.zeros((x.shape[1],), dtype=np.float32)
        std = np.ones((x.shape[1],), dtype=np.float32)
        if scale_mask.any():
            mean[scale_mask] = x[:, scale_mask].mean(axis=0)
            std[scale_mask] = x[:, scale_mask].std(axis=0) + self.eps

        self.mean = mean
        self.std = std
        self.scale_mask = scale_mask
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer is not fitted")
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / self.std

    def state_dict(self) -> dict:
        if self.mean is None or self.std is None or self.scale_mask is None:
            raise RuntimeError("Standardizer is not fitted")
        return {
            "mean": self.mean,
            "std": self.std,
            "scale_mask": self.scale_mask.astype(np.int64),
            "eps": np.array([self.eps], dtype=np.float32),
        }

    @staticmethod
    def from_state_dict(state: dict) -> "Standardizer":
        obj = Standardizer(float(np.array(state["eps"]).reshape(-1)[0]))
        obj.mean = np.array(state["mean"])
        obj.std = np.array(state["std"])
        obj.scale_mask = np.array(state.get("scale_mask", np.ones_like(obj.mean, dtype=np.int64))).astype(bool)
        return obj


def one_hot_int(x: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((x.shape[0], num_classes), dtype=np.float32)
    out[np.arange(x.shape[0]), x.astype(np.int64)] = 1.0
    return out


def it_bits(it: np.ndarray) -> np.ndarray:
    it = np.asarray(it, dtype=np.int64)
    b0 = (it & 1).astype(np.float32)
    b1 = ((it >> 1) & 1).astype(np.float32)
    b2 = ((it >> 2) & 1).astype(np.float32)
    b3 = ((it >> 3) & 1).astype(np.float32)
    return np.stack([b0, b1, b2, b3], axis=1)


def fit_freq_bin_edges(freq_train: np.ndarray, bins: int) -> np.ndarray:
    freq_train = np.asarray(freq_train, dtype=np.float32)
    if bins <= 0:
        return np.array([], dtype=np.float32)
    lo = float(np.min(freq_train))
    hi = float(np.max(freq_train))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    return np.linspace(lo, hi, bins + 1, dtype=np.float32)


def freq_to_bin_onehot(freq: np.ndarray, edges: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.float32)
    if edges.size == 0:
        return np.zeros((freq.shape[0], 0), dtype=np.float32)
    bins = edges.size - 1
    idx = np.searchsorted(edges[1:-1], freq, side="right")
    idx = np.clip(idx, 0, bins - 1)
    out = np.zeros((freq.shape[0], bins), dtype=np.float32)
    out[np.arange(freq.shape[0]), idx.astype(np.int64)] = 1.0
    return out


def build_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> tuple[np.ndarray, dict, np.ndarray]:
    g1_re = df["gammaIn1Re"].to_numpy(np.float32)
    g1_im = df["gammaIn1Im"].to_numpy(np.float32)
    g2_re = df["gammaIn2Re"].to_numpy(np.float32)
    g2_im = df["gammaIn2Im"].to_numpy(np.float32)
    freq = df["closeFreqMHz"].to_numpy(np.float32)
    it = df["itState"].to_numpy(np.int64)

    it_num = int(it.max()) + 1
    x_parts: list[np.ndarray] = []
    scale_masks: list[np.ndarray] = []

    base = np.stack([g1_re, g1_im, g2_re, g2_im, freq], axis=1)
    x_parts.append(base)
    scale_masks.append(np.ones((base.shape[1],), dtype=bool))

    if config.derived:
        g1_mag = np.sqrt(g1_re**2 + g1_im**2)
        g2_mag = np.sqrt(g2_re**2 + g2_im**2)
        g1_phase = np.arctan2(g1_im, g1_re)
        g2_phase = np.arctan2(g2_im, g2_re)
        delta_re = g1_re - g2_re
        delta_im = g1_im - g2_im
        delta_mag = g1_mag - g2_mag
        delta_phase = np.arctan2(np.sin(g1_phase - g2_phase), np.cos(g1_phase - g2_phase))
        derived = np.stack([g1_mag, g2_mag, g1_phase, g2_phase, delta_re, delta_im, delta_mag, delta_phase], axis=1)
        x_parts.append(derived)
        scale_masks.append(np.ones((derived.shape[1],), dtype=bool))

    it_encoding = str(config.it_encoding).lower().strip()
    if it_encoding == "onehot":
        it_part = one_hot_int(it, it_num)
        x_parts.append(it_part)
        scale_masks.append(np.zeros((it_part.shape[1],), dtype=bool))
    elif it_encoding == "bits":
        it_part = it_bits(it)
        x_parts.append(it_part)
        scale_masks.append(np.zeros((it_part.shape[1],), dtype=bool))
    else:
        raise ValueError(f"Unknown it_encoding: {config.it_encoding}")

    x = np.concatenate(x_parts, axis=1).astype(np.float32)
    scale_mask = np.concatenate(scale_masks, axis=0).astype(bool)

    meta = {
        "it_num": it_num,
        "derived": bool(config.derived),
        "it_encoding": it_encoding,
        "feature_dim": int(x.shape[1]),
    }
    return x, meta, scale_mask


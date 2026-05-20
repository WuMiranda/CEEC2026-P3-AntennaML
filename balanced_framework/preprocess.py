from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    derived: bool = True


class Standardizer:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + self.eps
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer is not fitted")
        return (x - self.mean) / self.std

    def state_dict(self) -> dict:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer is not fitted")
        return {"mean": self.mean, "std": self.std, "eps": np.array([self.eps], dtype=np.float32)}

    @staticmethod
    def from_state_dict(state: dict) -> "Standardizer":
        obj = Standardizer(float(np.array(state["eps"]).reshape(-1)[0]))
        obj.mean = np.array(state["mean"])
        obj.std = np.array(state["std"])
        return obj


def one_hot_int(x: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((x.shape[0], num_classes), dtype=np.float32)
    out[np.arange(x.shape[0]), x.astype(np.int64)] = 1.0
    return out


def build_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> tuple[np.ndarray, dict]:
    g1_re = df["gammaIn1Re"].to_numpy(np.float32)
    g1_im = df["gammaIn1Im"].to_numpy(np.float32)
    g2_re = df["gammaIn2Re"].to_numpy(np.float32)
    g2_im = df["gammaIn2Im"].to_numpy(np.float32)
    freq = df["closeFreqMHz"].to_numpy(np.float32)
    it = df["itState"].to_numpy(np.int64)

    it_num = int(it.max()) + 1
    x_parts = [np.stack([g1_re, g1_im, g2_re, g2_im, freq], axis=1)]

    if config.derived:
        g1_mag = np.sqrt(g1_re**2 + g1_im**2)
        g2_mag = np.sqrt(g2_re**2 + g2_im**2)
        g1_phase = np.arctan2(g1_im, g1_re)
        g2_phase = np.arctan2(g2_im, g2_re)
        delta_re = g1_re - g2_re
        delta_im = g1_im - g2_im
        delta_mag = g1_mag - g2_mag
        delta_phase = np.arctan2(np.sin(g1_phase - g2_phase), np.cos(g1_phase - g2_phase))
        x_parts.append(np.stack([g1_mag, g2_mag, g1_phase, g2_phase, delta_re, delta_im, delta_mag, delta_phase], axis=1))

    x_parts.append(one_hot_int(it, it_num))
    x = np.concatenate(x_parts, axis=1).astype(np.float32)

    meta = {"it_num": it_num, "derived": bool(config.derived), "feature_dim": int(x.shape[1])}
    return x, meta


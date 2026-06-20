from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ABLATION_FEATURES = (
    "vswr",
    "rl",
    "g1_mag",
    "g2_mag",
    "g1_phase",
    "g2_phase",
    "delta_mag",
    "delta_phase",
    "delta_im",
    "delta_re",
)


@dataclass(frozen=True)
class FeatureConfig:
    derived: bool = True
    it_encoding: str = "bits"
    feature_set: str = "full"
    use_itstate_features: bool = True
    use_closed_state_interactions: bool = False
    use_impedance_features: bool = False
    rf_features: str = "none"
    vswr_clip: float = 100.0
    exclude_features: tuple[str, ...] = ()
    z0: float = 50.0
    gamma_scale: float = 1.0


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


def freq_to_bin_index(freq: np.ndarray, edges: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.float32)
    if edges.size == 0:
        return np.zeros((freq.shape[0], 0), dtype=np.float32)
    bins = edges.size - 1
    idx = np.searchsorted(edges[1:-1], freq, side="right")
    idx = np.clip(idx, 0, bins - 1).astype(np.float32)
    return idx.reshape(-1, 1)


def decode_gamma_columns(
    df: Any,
    gamma_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = float(abs(gamma_scale))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("gamma_scale must be a finite value greater than 0")
    return (
        df["gammaIn1Re"].to_numpy(np.float32) / scale,
        df["gammaIn1Im"].to_numpy(np.float32) / scale,
        df["gammaIn2Re"].to_numpy(np.float32) / scale,
        df["gammaIn2Im"].to_numpy(np.float32) / scale,
    )


def compute_log_magnitude_ratio(
    df: Any,
    gamma_scale: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    if not np.isfinite(eps) or float(eps) <= 0.0:
        raise ValueError("log-ratio eps must be a finite value greater than 0")
    g1_re, g1_im, g2_re, g2_im = decode_gamma_columns(df, gamma_scale=gamma_scale)
    mag1 = np.hypot(g1_re, g1_im).astype(np.float64)
    mag2 = np.hypot(g2_re, g2_im).astype(np.float64)
    values = np.log((mag1 + float(eps)) / (mag2 + float(eps)))
    if not np.isfinite(values).all():
        raise ValueError("log magnitude ratio contains non-finite values")
    return values.astype(np.float32)


def fit_quantile_clip_bounds(
    train_values: np.ndarray,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999,
) -> tuple[float, float]:
    lower_quantile = float(lower_quantile)
    upper_quantile = float(upper_quantile)
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    train_values = np.asarray(train_values, dtype=np.float64).reshape(-1)
    if train_values.size == 0 or not np.isfinite(train_values).all():
        raise ValueError("training values for quantile clipping must be finite and non-empty")
    lower, upper = np.quantile(train_values, [lower_quantile, upper_quantile])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"invalid fitted clip bounds: lower={lower}, upper={upper}")
    return float(lower), float(upper)


def build_rf_features(
    g1: np.ndarray,
    g2: np.ndarray,
    mode: str,
    vswr_clip: float,
    exclude_features: tuple[str, ...] = (),
    eps: float = 1e-6,
) -> np.ndarray:
    mode = str(mode).lower().strip()
    if mode not in {"none", "vswr", "rl", "vswr_rl"}:
        raise ValueError(f"Unknown rf_features mode: {mode}")
    if mode == "none":
        return np.zeros((np.asarray(g1).shape[0], 0), dtype=np.float32)
    if not np.isfinite(vswr_clip) or float(vswr_clip) <= 1.0:
        raise ValueError("vswr_clip must be a finite value greater than 1")
    excluded = set(exclude_features)

    mag1 = np.clip(np.abs(g1).astype(np.float32), eps, 1.0 - eps)
    mag2 = np.clip(np.abs(g2).astype(np.float32), eps, 1.0 - eps)
    parts: list[np.ndarray] = []
    if mode in {"vswr", "vswr_rl"} and "vswr" not in excluded:
        vswr1 = np.minimum((1.0 + mag1) / (1.0 - mag1), float(vswr_clip))
        vswr2 = np.minimum((1.0 + mag2) / (1.0 - mag2), float(vswr_clip))
        parts.extend([vswr1, vswr2])
    if mode in {"rl", "vswr_rl"} and "rl" not in excluded:
        parts.extend([-20.0 * np.log10(mag1), -20.0 * np.log10(mag2)])
    if not parts:
        return np.zeros((np.asarray(g1).shape[0], 0), dtype=np.float32)
    return np.stack(parts, axis=1).astype(np.float32)


def build_impedance_features(
    g1: np.ndarray,
    g2: np.ndarray,
    z0: float,
    eps: float = 1e-6,
) -> np.ndarray:
    z0 = float(z0)
    gamma1 = np.asarray(g1, dtype=np.complex64)
    gamma2 = np.asarray(g2, dtype=np.complex64)

    def gamma_to_zin(gamma: np.ndarray) -> np.ndarray:
        denom = 1.0 - gamma
        denom = np.where(np.abs(denom) < eps, eps + 0j, denom)
        return z0 * (1.0 + gamma) / denom

    zin1 = gamma_to_zin(gamma1)
    zin2 = gamma_to_zin(gamma2)

    g1_r = np.real(zin1).astype(np.float32)
    g1_x = np.imag(zin1).astype(np.float32)
    g2_r = np.real(zin2).astype(np.float32)
    g2_x = np.imag(zin2).astype(np.float32)
    delta_r = g1_r - g2_r
    delta_x = g1_x - g2_x
    g1_abs_z = np.abs(zin1).astype(np.float32)
    g2_abs_z = np.abs(zin2).astype(np.float32)

    def matching_metrics(gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mag = np.clip(np.abs(gamma).astype(np.float32), eps, 1.0 - eps)
        vswr = (1.0 + mag) / (1.0 - mag)
        return_loss = -20.0 * np.log10(mag)
        return vswr.astype(np.float32), return_loss.astype(np.float32)

    g1_vswr, g1_return_loss = matching_metrics(gamma1)
    g2_vswr, g2_return_loss = matching_metrics(gamma2)

    return np.stack(
        [
            g1_r,
            g1_x,
            g2_r,
            g2_x,
            delta_r,
            delta_x,
            g1_abs_z,
            g2_abs_z,
            g1_vswr,
            g2_vswr,
            g1_return_loss,
            g2_return_loss,
        ],
        axis=1,
    ).astype(np.float32)


def build_features(
    df: Any,
    config: FeatureConfig,
) -> tuple[np.ndarray, dict, np.ndarray]:
    g1_re, g1_im, g2_re, g2_im = decode_gamma_columns(df, gamma_scale=float(config.gamma_scale))
    freq = df["closeFreqMHz"].to_numpy(np.float32)
    it = df["itState"].to_numpy(np.int64)

    it_num = int(it.max()) + 1
    x_parts: list[np.ndarray] = []
    scale_masks: list[np.ndarray] = []

    excluded = tuple(str(v).lower().strip() for v in config.exclude_features)
    unknown_excluded = sorted(set(excluded) - set(ABLATION_FEATURES))
    if unknown_excluded:
        raise ValueError(f"Unknown excluded features: {unknown_excluded}")
    excluded_set = set(excluded)

    feature_set = str(config.feature_set).lower().strip()
    if feature_set == "full":
        base = np.stack([g1_re, g1_im, g2_re, g2_im, freq], axis=1)
        x_parts.append(base)
        scale_masks.append(np.ones((base.shape[1],), dtype=bool))

        if config.derived:
            g1 = g1_re.astype(np.complex64) + 1j * g1_im.astype(np.complex64)
            g2 = g2_re.astype(np.complex64) + 1j * g2_im.astype(np.complex64)
            g1_mag = np.abs(g1).astype(np.float32)
            g2_mag = np.abs(g2).astype(np.float32)
            g1_phase = np.angle(g1).astype(np.float32)
            g2_phase = np.angle(g2).astype(np.float32)
            delta_re = g1_re - g2_re
            delta_im = g1_im - g2_im
            delta_mag = g1_mag - g2_mag
            delta_phase = np.arctan2(np.sin(g1_phase - g2_phase), np.cos(g1_phase - g2_phase))
            derived_candidates = (
                ("g1_mag", g1_mag),
                ("g2_mag", g2_mag),
                ("g1_phase", g1_phase),
                ("g2_phase", g2_phase),
                ("delta_re", delta_re),
                ("delta_im", delta_im),
                ("delta_mag", delta_mag),
                ("delta_phase", delta_phase),
            )
            kept = [values for name, values in derived_candidates if name not in excluded_set]
            if kept:
                derived = np.stack(kept, axis=1).astype(np.float32)
                x_parts.append(derived)
                scale_masks.append(np.ones((derived.shape[1],), dtype=bool))
    elif feature_set == "phys_min":
        g1 = g1_re.astype(np.complex64) + 1j * g1_im.astype(np.complex64)
        g2 = g2_re.astype(np.complex64) + 1j * g2_im.astype(np.complex64)
        g1_mag = np.abs(g1).astype(np.float32)
        g2_mag = np.abs(g2).astype(np.float32)
        delta_re = g1_re - g2_re
        delta_im = g1_im - g2_im
        delta_gamma_mag = np.sqrt(delta_re**2 + delta_im**2)
        delta_mag = g1_mag - g2_mag
        g1_phase = np.angle(g1).astype(np.float32)
        g2_phase = np.angle(g2).astype(np.float32)
        delta_phase = np.arctan2(np.sin(g1_phase - g2_phase), np.cos(g1_phase - g2_phase))
        cross = g1_re * g2_im - g1_im * g2_re
        dot = g1_re * g2_re + g1_im * g2_im
        rel_angle = np.arctan2(cross, dot)

        phys = np.stack([freq, g1_mag, g2_mag, delta_gamma_mag, delta_mag, delta_phase, rel_angle], axis=1)
        x_parts.append(phys.astype(np.float32))
        scale_masks.append(np.ones((phys.shape[1],), dtype=bool))
    else:
        raise ValueError(f"Unknown feature_set: {config.feature_set}")

    g1 = g1_re.astype(np.complex64) + 1j * g1_im.astype(np.complex64)
    g2 = g2_re.astype(np.complex64) + 1j * g2_im.astype(np.complex64)

    if bool(config.use_impedance_features):
        impedance = build_impedance_features(g1=g1, g2=g2, z0=float(config.z0))
        x_parts.append(impedance)
        scale_masks.append(np.ones((impedance.shape[1],), dtype=bool))

    rf_mode = str(config.rf_features).lower().strip()
    rf = build_rf_features(
        g1=g1,
        g2=g2,
        mode=rf_mode,
        vswr_clip=float(config.vswr_clip),
        exclude_features=excluded,
    )
    if rf.shape[1] > 0:
        x_parts.append(rf)
        scale_masks.append(np.ones((rf.shape[1],), dtype=bool))

    if bool(config.use_closed_state_interactions):
        bits = it_bits(it)
        closed_ri = np.stack([g1_re, g1_im], axis=1)
        interactions = (bits[:, :, None] * closed_ri[:, None, :]).reshape(it.shape[0], -1)
        x_parts.append(interactions.astype(np.float32))
        scale_masks.append(np.ones((interactions.shape[1],), dtype=bool))

    it_encoding = str(config.it_encoding).lower().strip()
    if bool(config.use_itstate_features):
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
        "feature_set": feature_set,
        "use_itstate_features": bool(config.use_itstate_features),
        "use_closed_state_interactions": bool(config.use_closed_state_interactions),
        "use_impedance_features": bool(config.use_impedance_features),
        "rf_features": rf_mode,
        "vswr_clip": float(config.vswr_clip),
        "exclude_features": list(excluded),
        "z0": float(config.z0),
        "gamma_scale": float(config.gamma_scale),
        "feature_dim": int(x.shape[1]),
    }
    return x, meta, scale_mask

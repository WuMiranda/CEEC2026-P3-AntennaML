from __future__ import annotations

import csv
import io
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.io import load_features_csv, load_labels_csv

from .freq_tree import FrequencyTree, apply_frequency_tree
from .geometry import CircleParams, build_residual_features, build_residual_summary_features
from .metrics import confusion_matrix, metrics_dict_from_cm
from .models import FrequencyGatedMLP, MLP
from .preprocess import (
    FeatureConfig,
    Standardizer,
    build_features,
    compute_log_magnitude_ratio,
    decode_gamma_columns,
    freq_to_bin_index,
    freq_to_bin_onehot,
)


def scalar_from_state(state: Any, name: str, default: Any) -> Any:
    if name not in state:
        return default
    arr = np.asarray(state[name])
    if arr.size == 0:
        return default
    return arr.reshape(-1)[0].item()


def load_experiment_state(model_dir: str | Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    preprocess_path = model_dir / "preprocess.npz"
    model_path = model_dir / "model.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not preprocess_path.exists():
        raise FileNotFoundError(f"Missing preprocess file: {preprocess_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    preprocess = np.load(preprocess_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    return {
        "model_dir": model_dir,
        "config": config,
        "preprocess": preprocess,
        "checkpoint": checkpoint,
    }


def _load_circle_params(path: Path) -> CircleParams:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty circle params file: {path}")

    def col(name: str, dtype: Any) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=dtype)

    return CircleParams(
        labels=col("label", np.int64),
        n_train=col("n_train", np.int64),
        g1_cx=col("g1_cx", np.float32),
        g1_cy=col("g1_cy", np.float32),
        g1_r=col("g1_r", np.float32),
        g2_cx=col("g2_cx", np.float32),
        g2_cy=col("g2_cy", np.float32),
        g2_r=col("g2_r", np.float32),
    )


def build_feature_config(config: dict[str, Any], preprocess: Any) -> FeatureConfig:
    gamma_scale = float(config.get("gamma_scale", scalar_from_state(preprocess, "gamma_scale", 1.0)))
    return FeatureConfig(
        derived=bool(config.get("derived_features", True)),
        it_encoding=str(config.get("it_encoding", "bits")),
        feature_set=str(config.get("feature_set", "full")),
        use_itstate_features=bool(config.get("use_itstate_features", True)),
        use_closed_state_interactions=bool(config.get("use_closed_state_interactions", False)),
        use_impedance_features=bool(config.get("use_impedance_features", False)),
        rf_features=str(config.get("rf_features", "none")),
        vswr_clip=float(config.get("vswr_clip", 100.0)),
        exclude_features=tuple(str(v) for v in config.get("exclude_features", [])),
        z0=float(config.get("z0", 50.0)),
        gamma_scale=gamma_scale,
    )


def load_split_indices(model_dir: str | Path) -> dict[str, np.ndarray]:
    model_dir = Path(model_dir)
    out: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        path = model_dir / f"{name}_idx.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        out[name] = np.load(path).astype(np.int64)
    return out


def build_base_model(checkpoint: dict[str, Any]) -> nn.Module:
    num_classes = int(checkpoint["num_classes"])
    in_dim = int(checkpoint["in_dim"])
    hidden_dims = tuple(int(v) for v in checkpoint.get("hidden_dims", (128, 64, 32)))
    freq_tree_depth = int(checkpoint.get("freq_tree_depth", 0))
    if freq_tree_depth > 0:
        num_leaves = int(checkpoint.get("num_leaves", 0))
        model = FrequencyGatedMLP(
            in_dim=in_dim,
            num_classes=num_classes,
            num_leaves=num_leaves,
            hidden_dims=hidden_dims,
        )
    else:
        model = MLP(in_dim=in_dim, num_classes=num_classes, hidden_dims=hidden_dims)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


class PriorAdjustedMLP(nn.Module):
    def __init__(self, model: nn.Module, adjust: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("adjust", adjust.to(dtype=torch.float32))

    def forward(self, x: torch.Tensor, leaf_id: torch.Tensor | None = None) -> torch.Tensor:
        del leaf_id
        return self.model(x) + self.adjust


class FrequencyAdjustedMLP(nn.Module):
    def __init__(self, model: nn.Module, leaf_adjust: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("leaf_adjust", leaf_adjust.to(dtype=torch.float32))

    def forward(self, x: torch.Tensor, leaf_id: torch.Tensor | None = None) -> torch.Tensor:
        if leaf_id is None:
            raise ValueError("leaf_id is required for frequency-gated models")
        leaf_id = leaf_id.to(dtype=torch.long)
        return self.model(x, leaf_id) + self.leaf_adjust[leaf_id]


def build_inference_model(state: dict[str, Any]) -> nn.Module:
    config = state["config"]
    preprocess = state["preprocess"]
    checkpoint = state["checkpoint"]
    model = build_base_model(checkpoint)
    tau = float(config.get("tau", scalar_from_state(preprocess, "tau", 0.0)))
    freq_tree_depth = int(checkpoint.get("freq_tree_depth", 0))
    num_classes = int(checkpoint["num_classes"])
    if freq_tree_depth > 0:
        leaf_priors = np.asarray(preprocess["leaf_priors"], dtype=np.float32)
        if leaf_priors.ndim != 2 or leaf_priors.shape[1] != num_classes:
            raise ValueError("Invalid saved leaf priors")
        leaf_adjust = (-tau) * np.log(leaf_priors + 1e-12)
        return FrequencyAdjustedMLP(model=model, leaf_adjust=torch.tensor(leaf_adjust, dtype=torch.float32))

    priors = np.asarray(preprocess["priors"], dtype=np.float32)
    if priors.shape != (num_classes,):
        raise ValueError(f"Saved priors shape mismatch: {priors.shape}, expected ({num_classes},)")
    adjust = (-tau) * np.log(priors + 1e-12)
    return PriorAdjustedMLP(model=model, adjust=torch.tensor(adjust, dtype=torch.float32))


def quantize_dynamic_linear_layers(model: nn.Module) -> nn.Module:
    quantize_dynamic = getattr(torch.quantization, "quantize_dynamic", None)
    if quantize_dynamic is None:
        quantize_dynamic = torch.ao.quantization.quantize_dynamic
    quantized = quantize_dynamic(deepcopy(model).cpu(), {nn.Linear}, dtype=torch.qint8)
    quantized.eval()
    return quantized


def serialized_state_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    size_mb = len(buf.getvalue()) / (1024.0 * 1024.0)
    return float(size_mb)


def _append_log_ratio_features(
    x: np.ndarray,
    features_df: Any,
    config: dict[str, Any],
    preprocess: Any,
    feat_cfg: FeatureConfig,
) -> np.ndarray:
    if not bool(config.get("use_log_magnitude_ratio", False)):
        return x
    log_ratio = compute_log_magnitude_ratio(
        features_df,
        gamma_scale=float(feat_cfg.gamma_scale),
        eps=float(config.get("log_ratio_eps", scalar_from_state(preprocess, "log_ratio_eps", 1e-6))),
    )
    lower = config.get("log_ratio_lower", None)
    upper = config.get("log_ratio_upper", None)
    if lower is None:
        lower = scalar_from_state(preprocess, "log_ratio_lower", None)
    if upper is None:
        upper = scalar_from_state(preprocess, "log_ratio_upper", None)
    if lower is not None and upper is not None:
        log_ratio = np.clip(log_ratio, float(lower), float(upper))
    return np.concatenate([x, log_ratio.astype(np.float32)[:, None]], axis=1)


def _append_frequency_features(
    x: np.ndarray,
    features_df: Any,
    config: dict[str, Any],
    preprocess: Any,
) -> np.ndarray:
    freq_edges = np.asarray(preprocess.get("freq_bin_edges", np.array([], dtype=np.float32)), dtype=np.float32)
    if freq_edges.size == 0:
        return x
    freq_all = features_df["closeFreqMHz"].to_numpy(np.float32)
    freq_bin_mode = str(config.get("freq_bin_mode", "none")).lower().strip()
    if freq_bin_mode == "onehot":
        freq_feat = freq_to_bin_onehot(freq_all, freq_edges)
    elif freq_bin_mode == "index":
        freq_feat = freq_to_bin_index(freq_all, freq_edges)
    elif freq_bin_mode == "none":
        return x
    else:
        raise ValueError(f"Unknown freq_bin_mode: {freq_bin_mode}")
    if freq_feat.shape[1] == 0:
        return x
    return np.concatenate([x, freq_feat.astype(np.float32)], axis=1)


def _append_geometry_features(
    x: np.ndarray,
    model_dir: Path,
    features_df: Any,
    config: dict[str, Any],
    feat_cfg: FeatureConfig,
) -> np.ndarray:
    geometry_mode = str(config.get("geometry_mode", "none")).lower().strip()
    if geometry_mode == "none":
        return x

    circle_path = model_dir / "circle_params.tsv"
    if not circle_path.exists():
        raise FileNotFoundError(f"Missing geometry parameters: {circle_path}")
    params = _load_circle_params(circle_path)
    g1_re, g1_im, g2_re, g2_im = decode_gamma_columns(features_df, gamma_scale=float(feat_cfg.gamma_scale))
    if geometry_mode == "full":
        geom = build_residual_features(g1_re=g1_re, g1_im=g1_im, g2_re=g2_re, g2_im=g2_im, params=params)
    elif geometry_mode == "summary":
        geom = build_residual_summary_features(g1_re=g1_re, g1_im=g1_im, g2_re=g2_re, g2_im=g2_im, params=params)
    else:
        raise ValueError(f"Unknown geometry_mode: {geometry_mode}")
    return np.concatenate([x, geom.astype(np.float32)], axis=1)


def load_preprocessed_split(
    model_dir: str | Path,
    split: str = "test",
    data_dir: str | Path | None = None,
    x_csv: str | None = None,
    y_csv: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    state = load_experiment_state(model_dir)
    model_dir = state["model_dir"]
    config = state["config"]
    preprocess = state["preprocess"]
    checkpoint = state["checkpoint"]

    split_name = str(split).lower().strip()
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")

    data_dir = Path(data_dir if data_dir is not None else config["data_dir"])
    x_csv = str(x_csv if x_csv is not None else config.get("x_csv", "x_train.csv"))
    y_csv = str(y_csv if y_csv is not None else config.get("y_csv", "y_train.csv"))

    features_df = load_features_csv(data_dir / x_csv)
    y_all = load_labels_csv(data_dir / y_csv).astype(np.int64)

    feat_cfg = build_feature_config(config, preprocess)
    x_all, _, _ = build_features(features_df, feat_cfg)
    x_all = _append_log_ratio_features(x_all, features_df, config, preprocess, feat_cfg)
    x_all = _append_frequency_features(x_all, features_df, config, preprocess)
    x_all = _append_geometry_features(x_all, model_dir, features_df, config, feat_cfg)

    in_dim = int(checkpoint["in_dim"])
    if int(x_all.shape[1]) != in_dim:
        raise ValueError(f"Feature dim mismatch: built={x_all.shape[1]}, checkpoint={in_dim}")

    scaler = Standardizer.from_state_dict(preprocess)
    x_all = scaler.transform(x_all).astype(np.float32)

    split_indices = load_split_indices(model_dir)
    idx = split_indices[split_name]

    leaf_all: np.ndarray | None = None
    freq_tree_depth = int(checkpoint.get("freq_tree_depth", 0))
    if freq_tree_depth > 0:
        tree = FrequencyTree(
            threshold=np.asarray(preprocess["freq_tree_threshold"], dtype=np.float32),
            left=np.asarray(preprocess["freq_tree_left"], dtype=np.int64),
            right=np.asarray(preprocess["freq_tree_right"], dtype=np.int64),
            is_leaf=np.asarray(preprocess["freq_tree_is_leaf"], dtype=np.int64),
            leaf_id=np.asarray(preprocess["freq_tree_leaf_id"], dtype=np.int64),
        )
        freq_all = features_df["closeFreqMHz"].to_numpy(np.float32)
        leaf_all = apply_frequency_tree(freq_all, tree).astype(np.int64)

    x_split = x_all[idx]
    y_split = y_all[idx]
    leaf_split = None if leaf_all is None else leaf_all[idx]
    return x_split, y_split, leaf_split, state


def predict_with_model(
    model: nn.Module,
    x: np.ndarray,
    leaf: np.ndarray | None = None,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(x.shape[0]))
            xb = torch.from_numpy(x[start:end]).to(dtype=torch.float32)
            if leaf is None:
                logits = model(xb)
            else:
                lb = torch.from_numpy(leaf[start:end]).to(dtype=torch.long)
                logits = model(xb, lb)
            preds.append(logits.argmax(dim=1).cpu().numpy().astype(np.int64))
    return np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.int64)


def evaluate_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    leaf: np.ndarray | None = None,
    batch_size: int = 1024,
) -> dict[str, Any]:
    if y.size == 0:
        raise ValueError("Cannot evaluate on an empty split")
    pred = predict_with_model(model=model, x=x, leaf=leaf, batch_size=batch_size)
    num_classes = int(max(int(np.max(y)), int(np.max(pred))) + 1)
    cm = confusion_matrix(y, pred, num_classes=num_classes)
    out = metrics_dict_from_cm(cm)
    out["num_samples"] = int(y.shape[0])
    return out

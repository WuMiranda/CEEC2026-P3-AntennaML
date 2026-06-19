from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from balanced_framework.freq_tree import FrequencyTree, apply_frequency_tree
from balanced_framework.geometry import CircleParams, build_residual_features
from balanced_framework.models import ClassConditionalPriorMLP, FrequencyGatedMLP, MLP
from balanced_framework.preprocess import (
    FeatureConfig,
    Standardizer,
    build_features,
    compute_class_prior_features,
    compute_log_magnitude_ratio,
    freq_to_bin_onehot,
)


FEATURE_COLUMNS = ["gammaIn1Re", "gammaIn1Im", "closeFreqMHz", "itState", "gammaIn2Re", "gammaIn2Im"]


def _scalar(state: Any, name: str, default: Any) -> Any:
    if name not in state:
        return default
    arr = np.asarray(state[name])
    if arr.size == 0:
        return default
    return arr.reshape(-1)[0].item()


def _load_circle_params(path: Path) -> CircleParams:
    table = pd.read_csv(path, sep="\t")
    return CircleParams(
        labels=table["label"].to_numpy(np.int64),
        n_train=table["n_train"].to_numpy(np.int64),
        g1_cx=table["g1_cx"].to_numpy(np.float32),
        g1_cy=table["g1_cy"].to_numpy(np.float32),
        g1_r=table["g1_r"].to_numpy(np.float32),
        g2_cx=table["g2_cx"].to_numpy(np.float32),
        g2_cy=table["g2_cy"].to_numpy(np.float32),
        g2_r=table["g2_r"].to_numpy(np.float32),
    )


def _sample_to_dataframe(sample: list[float] | tuple[float, ...] | np.ndarray | dict | pd.DataFrame) -> pd.DataFrame:
    if isinstance(sample, pd.DataFrame):
        return sample.loc[:, FEATURE_COLUMNS].copy()
    if isinstance(sample, dict):
        return pd.DataFrame([[sample[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    arr = np.asarray(sample, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features, got {arr.shape[1]}")
    return pd.DataFrame(arr, columns=FEATURE_COLUMNS)


class Predictor:
    def __init__(
        self,
        model_dir: str | Path,
        model_name: str = "model.pt",
        preprocess_name: str = "preprocess.npz",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / model_name
        self.preprocess_path = self.model_dir / preprocess_name
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model file: {self.model_path}")
        if not self.preprocess_path.exists():
            raise FileNotFoundError(f"Missing preprocess file: {self.preprocess_path}")

        self.preprocess = np.load(self.preprocess_path)
        self.scaler = Standardizer.from_state_dict(self.preprocess)
        self.tau = float(_scalar(self.preprocess, "tau", 0.0))
        self.freq_edges = np.asarray(self.preprocess.get("freq_bin_edges", np.array([], dtype=np.float32)), dtype=np.float32)
        self.use_geometry_features = bool(int(_scalar(self.preprocess, "use_geometry_features", 0)))
        self.freq_tree_depth = int(_scalar(self.preprocess, "freq_tree_depth", 0))
        self.use_log_magnitude_ratio = bool(int(_scalar(self.preprocess, "use_log_magnitude_ratio", 0)))
        self.use_class_conditional_priors = bool(int(_scalar(self.preprocess, "use_class_conditional_priors", 0)))
        self.log_ratio_eps = float(_scalar(self.preprocess, "log_ratio_eps", 1e-6))
        self.log_ratio_lower = float(_scalar(self.preprocess, "log_ratio_lower", 0.0))
        self.log_ratio_upper = float(_scalar(self.preprocess, "log_ratio_upper", 0.0))
        if self.use_log_magnitude_ratio and self.log_ratio_lower >= self.log_ratio_upper:
            raise ValueError("Invalid saved log-magnitude-ratio clipping bounds")

        derived = bool(int(_scalar(self.preprocess, "derived", 1)))
        it_encoding = "bits" if int(_scalar(self.preprocess, "it_encoding", 0)) == 0 else "onehot"
        self.feature_config = FeatureConfig(
            derived=derived,
            it_encoding=it_encoding,
            use_itstate_features=bool(int(_scalar(self.preprocess, "use_itstate_features", 1))),
            use_closed_state_interactions=bool(
                int(_scalar(self.preprocess, "use_closed_state_interactions", 0))
            ),
            use_impedance_features=bool(int(_scalar(self.preprocess, "use_impedance_features", 0))),
            rf_features=str(_scalar(self.preprocess, "rf_features", "none")),
            vswr_clip=float(_scalar(self.preprocess, "vswr_clip", 100.0)),
            exclude_features=tuple(
                str(v) for v in np.asarray(
                    self.preprocess.get("exclude_features", np.array([], dtype=str))
                ).reshape(-1).tolist()
            ),
            use_magnitude_ratio=bool(int(_scalar(self.preprocess, "use_magnitude_ratio", 0))),
            magnitude_ratio_clip=float(_scalar(self.preprocess, "magnitude_ratio_clip", 100.0)),
            use_class_conditional_priors=self.use_class_conditional_priors,
            z0=float(_scalar(self.preprocess, "z0", 50.0)),
            gamma_scale=float(_scalar(self.preprocess, "gamma_scale", 32768.0)),
        )

        self.circle_params = None
        if self.use_geometry_features:
            circle_path = self.model_dir / "circle_params.tsv"
            if not circle_path.exists():
                raise FileNotFoundError(f"Geometry model needs circle_params.tsv: {circle_path}")
            self.circle_params = _load_circle_params(circle_path)

        self.tree = None
        if self.freq_tree_depth > 0:
            self.tree = FrequencyTree(
                threshold=np.asarray(self.preprocess["freq_tree_threshold"], dtype=np.float32),
                left=np.asarray(self.preprocess["freq_tree_left"], dtype=np.int64),
                right=np.asarray(self.preprocess["freq_tree_right"], dtype=np.int64),
                is_leaf=np.asarray(self.preprocess["freq_tree_is_leaf"], dtype=np.int64),
                leaf_id=np.asarray(self.preprocess["freq_tree_leaf_id"], dtype=np.int64),
            )

        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.num_classes = int(checkpoint["num_classes"])
        self.in_dim = int(checkpoint["in_dim"])
        self.hidden_dims = tuple(int(v) for v in checkpoint.get("hidden_dims", (128, 64, 32)))
        self.num_leaves = int(checkpoint.get("num_leaves", 0))
        self.model_type = str(checkpoint.get("model_type", "mlp"))
        self.ratio_prior_classes = tuple(int(v) for v in checkpoint.get("ratio_prior_classes", (4, 6, 11)))
        self.g2_phase_prior_classes = tuple(int(v) for v in checkpoint.get("g2_phase_prior_classes", (8,)))
        self.prior_mean = np.asarray(self.preprocess.get("prior_mean", np.zeros(3)), dtype=np.float32)
        self.prior_std = np.asarray(self.preprocess.get("prior_std", np.ones(3)), dtype=np.float32)

        if self.freq_tree_depth > 0:
            self.model = FrequencyGatedMLP(
                in_dim=self.in_dim,
                num_classes=self.num_classes,
                num_leaves=self.num_leaves,
                hidden_dims=self.hidden_dims,
            )
        elif self.model_type == "class_conditional_prior":
            self.model = ClassConditionalPriorMLP(
                in_dim=self.in_dim,
                num_classes=self.num_classes,
                hidden_dims=self.hidden_dims,
                ratio_classes=self.ratio_prior_classes,
                phase_classes=self.g2_phase_prior_classes,
            )
        else:
            self.model = MLP(in_dim=self.in_dim, num_classes=self.num_classes, hidden_dims=self.hidden_dims)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def _build_matrix(self, sample: list[float] | tuple[float, ...] | np.ndarray | dict | pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        df = _sample_to_dataframe(sample)
        x, _, _, _ = build_features(df, self.feature_config)

        if self.use_log_magnitude_ratio and not self.use_class_conditional_priors:
            log_ratio = compute_log_magnitude_ratio(
                df,
                gamma_scale=self.feature_config.gamma_scale,
                eps=self.log_ratio_eps,
            )
            log_ratio = np.clip(log_ratio, self.log_ratio_lower, self.log_ratio_upper).astype(np.float32)
            x = np.concatenate([x, log_ratio[:, None]], axis=1)

        freq = df["closeFreqMHz"].to_numpy(np.float32)
        freq_oh = freq_to_bin_onehot(freq, self.freq_edges)
        if freq_oh.shape[1] > 0:
            x = np.concatenate([x, freq_oh.astype(np.float32)], axis=1)

        if self.use_geometry_features:
            assert self.circle_params is not None
            # Decode gamma the same way as in build_features
            g1_re_raw = df["gammaIn1Re"].to_numpy(np.float32)
            g1_im_raw = df["gammaIn1Im"].to_numpy(np.float32)
            g2_re_raw = df["gammaIn2Re"].to_numpy(np.float32)
            g2_im_raw = df["gammaIn2Im"].to_numpy(np.float32)
            
            gamma_scale = self.feature_config.gamma_scale
            scale = max(float(abs(gamma_scale)), 1e-6)
            g1 = (g1_re_raw.astype(np.complex64) + 1j * g1_im_raw.astype(np.complex64)) / scale
            g2 = (g2_re_raw.astype(np.complex64) + 1j * g2_im_raw.astype(np.complex64)) / scale
            
            g1_re = np.real(g1).astype(np.float32)
            g1_im = np.imag(g1).astype(np.float32)
            g2_re = np.real(g2).astype(np.float32)
            g2_im = np.imag(g2).astype(np.float32)
            
            geom = build_residual_features(
                g1_re=g1_re,
                g1_im=g1_im,
                g2_re=g2_re,
                g2_im=g2_im,
                params=self.circle_params,
            )
            x = np.concatenate([x, geom.astype(np.float32)], axis=1)

        if x.shape[1] != self.in_dim:
            raise ValueError(f"Feature dim mismatch: built={x.shape[1]}, model={self.in_dim}")

        if self.use_class_conditional_priors:
            prior = compute_class_prior_features(
                df,
                gamma_scale=self.feature_config.gamma_scale,
                log_ratio_eps=self.log_ratio_eps,
                log_ratio_lower=self.log_ratio_lower,
                log_ratio_upper=self.log_ratio_upper,
            )
            prior = (prior - self.prior_mean) / self.prior_std
        else:
            prior = np.zeros((x.shape[0], 3), dtype=np.float32)

        leaf = None
        if self.tree is not None:
            leaf = apply_frequency_tree(freq, self.tree).astype(np.int64)
        return self.scaler.transform(x), leaf, prior.astype(np.float32)

    def predict_proba(self, sample: list[float] | tuple[float, ...] | np.ndarray | dict | pd.DataFrame) -> np.ndarray:
        x, leaf, prior = self._build_matrix(sample)
        xt = torch.tensor(x, dtype=torch.float32)
        prior_t = torch.tensor(prior, dtype=torch.float32)
        with torch.no_grad():
            if self.freq_tree_depth > 0:
                if leaf is None:
                    raise RuntimeError("Frequency tree is enabled but leaf ids are missing")
                leaf_t = torch.tensor(leaf, dtype=torch.long)
                logits = self.model(xt, leaf_t)
                leaf_priors = np.asarray(self.preprocess["leaf_priors"], dtype=np.float32)
                adjust = -self.tau * np.log(leaf_priors[leaf] + 1e-12)
                logits = logits + torch.tensor(adjust, dtype=torch.float32)
            elif self.model_type == "class_conditional_prior":
                logits = self.model(xt, prior_t[:, :1], prior_t[:, 1:3])
                priors = np.asarray(self.preprocess["priors"], dtype=np.float32)
                adjust = -self.tau * np.log(priors + 1e-12)
                logits = logits + torch.tensor(adjust, dtype=torch.float32)
            else:
                logits = self.model(xt)
                priors = np.asarray(self.preprocess["priors"], dtype=np.float32)
                adjust = -self.tau * np.log(priors + 1e-12)
                logits = logits + torch.tensor(adjust, dtype=torch.float32)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, sample: list[float] | tuple[float, ...] | np.ndarray | dict | pd.DataFrame) -> int:
        probs = self.predict_proba(sample)
        return int(np.argmax(probs[0]))


def predict_one(sample: list[float] | tuple[float, ...] | np.ndarray | dict, model_dir: str | Path) -> int:
    return Predictor(model_dir=model_dir).predict(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-sample inference for balanced_framework models.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument(
        "--sample",
        type=float,
        nargs=6,
        required=True,
        metavar=("gammaIn1Re", "gammaIn1Im", "closeFreqMHz", "itState", "gammaIn2Re", "gammaIn2Im"),
    )
    parser.add_argument("--proba", action="store_true", default=False)
    args = parser.parse_args()

    predictor = Predictor(args.model_dir)
    if args.proba:
        probs = predictor.predict_proba(args.sample)[0]
        print({"pred": int(np.argmax(probs)), "proba": probs.tolist()})
    else:
        print(predictor.predict(args.sample))


if __name__ == "__main__":
    main()

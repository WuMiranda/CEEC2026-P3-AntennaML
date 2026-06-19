"""
Geometry-only 16-class classification experiment.

This script fits one eccentric circle per class for g1 and g2 on the
training set, converts every sample into 33 geometry features, trains the
selected classifiers, and writes only the final confusion matrix to OUTPUT_DIR.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

TRAIN_X_PATH = "/work2/cqw1/A/yds/datas/x_train_split.csv"
TRAIN_Y_PATH = "/work2/cqw1/A/yds/datas/y_train_split.csv"
TEST_X_PATH = "/work2/cqw1/A/yds/datas/x_test.csv"
TEST_Y_PATH = "/work2/cqw1/A/yds/datas/y_test.csv"
OUTPUT_DIR = "/work2/cqw1/A/yds/output/"

FEATURE_COLUMNS = ["g1_re", "g1_im", "freq", "itState", "g2_re", "g2_im"]
RANDOM_STATE = 42


def read_csv_auto_encoding(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """Read a CSV with several common encodings."""
    encodings = ("utf-8", "gbk", "gb18030")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **read_csv_kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to read csv: {path}")


def _columns_look_like_data(columns: Iterable[object]) -> bool:
    """Detect likely headerless files read with the first data row as header."""
    for col in columns:
        try:
            float(str(col).strip())
        except ValueError:
            return False
    return True


def read_feature_csv(path: str | Path) -> pd.DataFrame:
    """Read feature CSV and normalize columns to the required six names."""
    df = read_csv_auto_encoding(path)

    if df.shape[1] == len(FEATURE_COLUMNS) and _columns_look_like_data(df.columns):
        df = read_csv_auto_encoding(path, header=None)

    if df.shape[1] == len(FEATURE_COLUMNS) + 1:
        first_col = str(df.columns[0]).lower()
        if first_col.startswith("unnamed") or first_col in {"index", ""}:
            df = df.iloc[:, 1:].copy()

    if df.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Feature file must have 6 columns after normalization, got "
            f"{df.shape[1]} columns from {path}"
        )

    df = df.copy()
    df.columns = FEATURE_COLUMNS

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.isna().any().any():
        bad = df.isna().sum()
        bad = {col: int(count) for col, count in bad.items() if count > 0}
        raise ValueError(f"Feature file contains missing/non-numeric values: {bad}")

    return df


def read_label_csv(path: str | Path) -> np.ndarray:
    """Read one-column label CSV as int labels."""
    df = read_csv_auto_encoding(path)

    if df.shape[1] == 1 and _columns_look_like_data(df.columns):
        df = read_csv_auto_encoding(path, header=None)

    if df.shape[1] < 1:
        raise ValueError(f"Label file has no columns: {path}")

    y = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Label file contains missing/non-numeric labels: {path}")

    return y.astype(int).to_numpy()


def fit_circle(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Least-squares fit for x^2 + y^2 + A*x + B*y + C = 0.

    Returns:
        cx, cy, radius
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        raise ValueError("At least 3 points are required to fit a circle.")

    lhs = np.column_stack([x, y, np.ones_like(x)])
    rhs = -(x**2 + y**2)
    A, B, C = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    cx = -A / 2.0
    cy = -B / 2.0
    radius_sq = (A**2 + B**2) / 4.0 - C
    radius = math.sqrt(max(radius_sq, 0.0))

    return float(cx), float(cy), float(radius)


def fit_label_circles(X_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
    """Fit g1 and g2 circles for every training label."""
    params = []

    for label in sorted(np.unique(y_train)):
        mask = y_train == label
        label_df = X_train.loc[mask]

        cx1, cy1, r1 = fit_circle(
            label_df["g1_re"].to_numpy(),
            label_df["g1_im"].to_numpy(),
        )
        cx2, cy2, r2 = fit_circle(
            label_df["g2_re"].to_numpy(),
            label_df["g2_im"].to_numpy(),
        )

        params.append(
            {
                "label": int(label),
                "g1_cx": cx1,
                "g1_cy": cy1,
                "g1_radius": r1,
                "g2_cx": cx2,
                "g2_cy": cy2,
                "g2_radius": r2,
                "n_train": int(mask.sum()),
            }
        )

    return pd.DataFrame(params).sort_values("label").reset_index(drop=True)


def build_three_feature_matrix(
    X: pd.DataFrame,
    circle_params: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """Build 16 g1 residuals + 16 g2 residuals + 1 rel_angle."""
    labels = circle_params["label"].to_numpy()

    feature_blocks = []
    feature_names = []

    g1_re = X["g1_re"].to_numpy(dtype=float)
    g1_im = X["g1_im"].to_numpy(dtype=float)
    g2_re = X["g2_re"].to_numpy(dtype=float)
    g2_im = X["g2_im"].to_numpy(dtype=float)

    for _, row in circle_params.iterrows():
        label = int(row["label"])
        dist = np.sqrt((g1_re - row["g1_cx"]) ** 2 + (g1_im - row["g1_cy"]) ** 2)
        feature_blocks.append(np.abs(dist - row["g1_radius"]))
        feature_names.append(f"g1_residual_label_{label}")

    for _, row in circle_params.iterrows():
        label = int(row["label"])
        dist = np.sqrt((g2_re - row["g2_cx"]) ** 2 + (g2_im - row["g2_cy"]) ** 2)
        feature_blocks.append(np.abs(dist - row["g2_radius"]))
        feature_names.append(f"g2_residual_label_{label}")

    cross = g1_re * g2_im - g1_im * g2_re
    dot = g1_re * g2_re + g1_im * g2_im
    rel_angle = np.arctan2(cross, dot)
    feature_blocks.append(rel_angle)
    feature_names.append("rel_angle")

    features = np.column_stack(feature_blocks)

    if features.shape[1] != 2 * len(labels) + 1:
        raise RuntimeError("Unexpected geometry feature dimension.")

    return features, feature_names


def _make_models() -> Dict[str, object]:
    """Create all classifiers used in this experiment."""
    return {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "MLPClassifier": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        batch_size=256,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate_models(
    X_train_geo: np.ndarray,
    y_train: np.ndarray,
    X_test_geo: np.ndarray,
    y_test: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, dict]:
    """Train all models and collect predictions/metrics."""
    results = {}

    for model_name, model in _make_models().items():
        model.fit(X_train_geo, y_train)
        y_pred = model.predict(X_test_geo)

        results[model_name] = {
            "model": model,
            "y_pred": y_pred,
            "accuracy": accuracy_score(y_test, y_pred),
            "nmi": normalized_mutual_info_score(y_test, y_pred),
            "ari": adjusted_rand_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        }

    return results


def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str | Path,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_reports(
    output_dir: str | Path,
    results: Dict[str, dict],
    labels: np.ndarray,
) -> None:
    """Save only the final confusion matrix for the best model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_name = max(results, key=lambda name: results[name]["accuracy"])
    best_result = results[best_model_name]

    _plot_confusion_matrix(
        best_result["confusion_matrix"],
        labels,
        f"Confusion Matrix - Best Model: {best_model_name}",
        output_dir / "confusion_matrix.png",
    )

    print(f"\nBest model by accuracy: {best_model_name}")
    print(f"Accuracy: {best_result['accuracy']:.8f}")
    print(f"NMI: {best_result['nmi']:.8f}")
    print(f"ARI: {best_result['ari']:.8f}")
    print(f"Final confusion matrix saved to: {output_dir / 'confusion_matrix.png'}")


def save_circle_centers(output_dir: str | Path, circle_params: pd.DataFrame) -> None:
    """Save fitted circle centers and radii for every label."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "circle_centers.txt"

    with save_path.open("w", encoding="utf-8") as f:
        f.write("label\tn_train\tg1_cx\tg1_cy\tg1_radius\tg2_cx\tg2_cy\tg2_radius\n")
        for _, row in circle_params.iterrows():
            f.write(
                f"{int(row['label'])}\t"
                f"{int(row['n_train'])}\t"
                f"{row['g1_cx']:.10f}\t"
                f"{row['g1_cy']:.10f}\t"
                f"{row['g1_radius']:.10f}\t"
                f"{row['g2_cx']:.10f}\t"
                f"{row['g2_cy']:.10f}\t"
                f"{row['g2_radius']:.10f}\n"
            )

    print(f"Circle centers saved to: {save_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train = read_feature_csv(TRAIN_X_PATH)
    y_train = read_label_csv(TRAIN_Y_PATH)
    X_test = read_feature_csv(TEST_X_PATH)
    y_test = read_label_csv(TEST_Y_PATH)

    if len(X_train) != len(y_train):
        raise ValueError(f"Train X/y length mismatch: {len(X_train)} vs {len(y_train)}")
    if len(X_test) != len(y_test):
        raise ValueError(f"Test X/y length mismatch: {len(X_test)} vs {len(y_test)}")

    circle_params = fit_label_circles(X_train, y_train)
    save_circle_centers(OUTPUT_DIR, circle_params)
    labels = circle_params["label"].to_numpy()

    if len(labels) != 16:
        print(f"Warning: expected 16 labels, but found {len(labels)} labels: {labels}")

    X_train_geo, _ = build_three_feature_matrix(X_train, circle_params)
    X_test_geo, _ = build_three_feature_matrix(X_test, circle_params)

    results = train_and_evaluate_models(
        X_train_geo=X_train_geo,
        y_train=y_train,
        X_test_geo=X_test_geo,
        y_test=y_test,
        labels=labels,
    )

    save_reports(
        output_dir=OUTPUT_DIR,
        results=results,
        labels=labels,
    )


if __name__ == "__main__":
    main()

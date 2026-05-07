
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = (
    "gammaIn1Re",
    "gammaIn1Re",
    "gammaIn1Im",
    "closeFreqMHz",
    "itState",
    "gammaIn2Re",
    "gammaIn2Im",
)

DEFAULT_FLOAT_COLUMNS = (
    "gammaIn1Re",
    "gammaIn1Im",
    "closeFreqMHz",
    "gammaIn2Re",
    "gammaIn2Im",
)

DEFAULT_INT_COLUMNS = ("itState",)


def read_csv_with_fallback(file_path: str | Path, encodings: Sequence[str] = ("utf-8", "gbk", "gb18030")) -> pd.DataFrame:
    file_path = Path(file_path)
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    if last_err is None:
        raise RuntimeError(f"Failed to read csv: {file_path}")
    raise last_err


def load_features_csv(
    file_path: str | Path,
    expected_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    float_columns: Sequence[str] = DEFAULT_FLOAT_COLUMNS,
    int_columns: Sequence[str] = DEFAULT_INT_COLUMNS,
    encodings: Sequence[str] = ("utf-8", "gbk", "gb18030"),
) -> pd.DataFrame:
    df = read_csv_with_fallback(file_path, encodings=encodings)
    df = _normalize_feature_columns(df, expected_columns)
    df = _coerce_feature_dtypes(df, float_columns=float_columns, int_columns=int_columns)
    _assert_no_missing_or_inf(df, columns=expected_columns)
    for c in int_columns:
        df[c] = df[c].astype(np.int64)
    return df


def load_labels_csv(file_path: str | Path, encodings: Sequence[str] = ("utf-8", "gbk", "gb18030")) -> np.ndarray:
    y_df = read_csv_with_fallback(file_path, encodings=encodings)
    if y_df.shape[1] == 0:
        raise ValueError(f"Label csv has no columns: {file_path}")
    if y_df.shape[1] > 1:
        y_series = y_df.iloc[:, 0]
    else:
        y_series = y_df.iloc[:, 0]
    y = pd.to_numeric(y_series, errors="coerce")
    if y.isna().any():
        bad = int(y.isna().sum())
        raise ValueError(f"Label csv contains {bad} non-numeric / missing values: {file_path}")
    return y.astype(np.int64).to_numpy()


def load_small_trainset(data_dir: str | Path = "train_data") -> tuple[pd.DataFrame, np.ndarray]:
    data_dir = Path(data_dir)
    x_path = data_dir / "x_train_small.csv"
    y_path = data_dir / "y_train_small.csv"
    return load_features_csv(x_path), load_labels_csv(y_path)


def load_small_predset(data_dir: str | Path = "train_data") -> tuple[pd.DataFrame, np.ndarray]:
    data_dir = Path(data_dir)
    x_path = data_dir / "x_pred_small.csv"
    y_path = data_dir / "y_pred_small.csv"
    return load_features_csv(x_path), load_labels_csv(y_path)


def to_feature_array(
    features_df: pd.DataFrame,
    columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    cols = list(columns)
    if list(features_df.columns) != cols:
        features_df = _normalize_feature_columns(features_df, cols)
    x = features_df.loc[:, cols].to_numpy(dtype=dtype, na_value=np.nan)
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Features contain NaN/inf after conversion")
    return x


def _normalize_feature_columns(df: pd.DataFrame, expected_columns: Sequence[str]) -> pd.DataFrame:
    expected_columns = list(expected_columns)

    if list(df.columns) == expected_columns:
        return df.copy()

    cols = [str(c) for c in df.columns]
    if df.shape[1] == len(expected_columns) + 1:
        if cols[0].startswith("Unnamed") or cols[0].lower() in ("index",):
            df = df.iloc[:, 1:].copy()
            cols = [str(c) for c in df.columns]

    if df.shape[1] == len(expected_columns):
        if set(expected_columns).issubset(set(df.columns)):
            return df.loc[:, expected_columns].copy()
        out = df.copy()
        out.columns = expected_columns
        return out

    raise ValueError(
        "Feature csv columns mismatch. "
        f"expected={list(expected_columns)}; got={list(df.columns)}; shape={df.shape}"
    )


def _coerce_feature_dtypes(
    df: pd.DataFrame,
    float_columns: Sequence[str],
    int_columns: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()

    for c in float_columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in int_columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].round().astype("Int64")

    return out


def _assert_no_missing_or_inf(df: pd.DataFrame, columns: Iterable[str]) -> None:
    cols = list(columns)
    sub = df.loc[:, cols]
    arr = sub.to_numpy(dtype=float, na_value=np.nan)

    if np.isinf(arr).any():
        raise ValueError("Found inf/-inf in features")

    if np.isnan(arr).any():
        na_counts = sub.isna().sum()
        bad = {k: int(v) for k, v in na_counts.items() if int(v) > 0}
        raise ValueError(f"Found missing values in features: {bad}")

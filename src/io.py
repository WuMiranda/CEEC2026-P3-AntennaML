
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_FEATURE_COLUMNS = (
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


class _SeriesLike:
    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values)

    def to_numpy(self, dtype: np.dtype | None = None) -> np.ndarray:
        if dtype is None:
            return np.asarray(self._values)
        return np.asarray(self._values, dtype=dtype)


class SimpleFeatureFrame:
    def __init__(self, data: dict[str, np.ndarray], columns: Sequence[str]):
        self._data = {str(k): np.asarray(v) for k, v in data.items()}
        self.columns = [str(c) for c in columns]
        if not self.columns:
            self.shape = (0, 0)
        else:
            n_rows = int(np.asarray(self._data[self.columns[0]]).shape[0])
            self.shape = (n_rows, len(self.columns))

    def __getitem__(self, key: str) -> _SeriesLike:
        return _SeriesLike(self._data[str(key)])


def read_csv_with_fallback(
    file_path: str | Path,
    encodings: Sequence[str] = ("utf-8", "gbk", "gb18030"),
) -> tuple[list[str], list[list[str]]]:
    file_path = Path(file_path)
    last_err: Exception | None = None
    for enc in encodings:
        try:
            with file_path.open("r", encoding=enc, newline="") as f:
                rows = list(csv.reader(f))
            if not rows:
                return [], []
            header = [str(c) for c in rows[0]]
            return header, rows[1:]
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
) -> SimpleFeatureFrame:
    columns, rows = read_csv_with_fallback(file_path, encodings=encodings)
    columns, rows = _normalize_feature_columns(columns, rows, expected_columns)
    return _build_feature_frame(
        columns=columns,
        rows=rows,
        float_columns=float_columns,
        int_columns=int_columns,
    )


def load_labels_csv(file_path: str | Path, encodings: Sequence[str] = ("utf-8", "gbk", "gb18030")) -> np.ndarray:
    _, rows = read_csv_with_fallback(file_path, encodings=encodings)
    if not rows:
        raise ValueError(f"Label csv has no columns: {file_path}")
    values: list[int] = []
    bad = 0
    for row in rows:
        if not row:
            bad += 1
            continue
        try:
            value = int(round(float(row[0])))
        except (TypeError, ValueError):
            bad += 1
            continue
        values.append(value)
    if bad > 0:
        raise ValueError(f"Label csv contains {bad} non-numeric / missing values: {file_path}")
    return np.asarray(values, dtype=np.int64)


def load_small_trainset(data_dir: str | Path = "train_data") -> tuple[SimpleFeatureFrame, np.ndarray]:
    data_dir = Path(data_dir)
    x_path = data_dir / "x_train_small.csv"
    y_path = data_dir / "y_train_small.csv"
    return load_features_csv(x_path), load_labels_csv(y_path)


def load_small_predset(data_dir: str | Path = "train_data") -> tuple[SimpleFeatureFrame, np.ndarray]:
    data_dir = Path(data_dir)
    x_path = data_dir / "x_pred_small.csv"
    y_path = data_dir / "y_pred_small.csv"
    return load_features_csv(x_path), load_labels_csv(y_path)


def to_feature_array(
    features_df: SimpleFeatureFrame,
    columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    cols = list(columns)
    if list(features_df.columns) != cols:
        raise ValueError(
            "Feature columns mismatch. "
            f"expected={cols}; got={list(features_df.columns)}"
        )
    x = np.stack([features_df[c].to_numpy(dtype=dtype) for c in cols], axis=1)
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Features contain NaN/inf after conversion")
    return x


def _normalize_feature_columns(
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    expected_columns: Sequence[str],
) -> tuple[list[str], list[list[str]]]:
    expected_columns = [str(c) for c in expected_columns]
    cols = [str(c) for c in columns]
    out_rows = [list(row) for row in rows]

    if cols == expected_columns:
        return cols, out_rows

    if len(cols) == len(expected_columns) + 1:
        first = cols[0]
        if first.startswith("Unnamed") or first.lower() == "index":
            cols = cols[1:]
            out_rows = [row[1:] for row in out_rows]

    if len(cols) == len(expected_columns):
        if set(expected_columns).issubset(set(cols)):
            index_map = [cols.index(name) for name in expected_columns]
            reordered = []
            for row in out_rows:
                reordered.append([row[idx] if idx < len(row) else "" for idx in index_map])
            return expected_columns, reordered
        return expected_columns, [row[: len(expected_columns)] for row in out_rows]

    raise ValueError(
        "Feature csv columns mismatch. "
        f"expected={expected_columns}; got={cols}; shape=({len(out_rows)}, {len(cols)})"
    )


def _build_feature_frame(
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    float_columns: Sequence[str],
    int_columns: Sequence[str],
) -> SimpleFeatureFrame:
    cols = [str(c) for c in columns]
    raw: dict[str, list[str]] = {c: [] for c in cols}
    for row in rows:
        padded = list(row) + [""] * max(0, len(cols) - len(row))
        for idx, col in enumerate(cols):
            raw[col].append(padded[idx])

    data: dict[str, np.ndarray] = {}
    bad_counts: dict[str, int] = {}

    for c in cols:
        values = raw[c]
        if c in float_columns:
            arr, bad = _coerce_float_column(values)
            data[c] = arr.astype(np.float32)
        elif c in int_columns:
            arr, bad = _coerce_int_column(values)
            data[c] = arr.astype(np.int64)
        else:
            arr = np.asarray(values)
            bad = 0
            data[c] = arr
        if bad > 0:
            bad_counts[c] = int(bad)

    if bad_counts:
        raise ValueError(f"Found missing or non-numeric values in features: {bad_counts}")

    _assert_no_missing_or_inf(data, columns=tuple(cols))
    return SimpleFeatureFrame(data=data, columns=cols)


def _coerce_float_column(values: Sequence[str]) -> tuple[np.ndarray, int]:
    out = np.zeros((len(values),), dtype=np.float64)
    bad = 0
    for i, value in enumerate(values):
        try:
            out[i] = float(value)
        except (TypeError, ValueError):
            out[i] = np.nan
            bad += 1
    return out, bad


def _coerce_int_column(values: Sequence[str]) -> tuple[np.ndarray, int]:
    out = np.zeros((len(values),), dtype=np.float64)
    bad = 0
    for i, value in enumerate(values):
        try:
            out[i] = round(float(value))
        except (TypeError, ValueError):
            out[i] = np.nan
            bad += 1
    return out, bad


def _assert_no_missing_or_inf(data: dict[str, np.ndarray], columns: Iterable[str]) -> None:
    bad: dict[str, int] = {}
    for c in columns:
        arr = np.asarray(data[str(c)], dtype=np.float64)
        invalid = int(np.isnan(arr).sum() + np.isinf(arr).sum())
        if invalid > 0:
            bad[str(c)] = invalid
    if bad:
        raise ValueError(f"Found missing values in features: {bad}")

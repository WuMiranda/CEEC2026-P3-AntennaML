from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ExperimentSummary:
    name: str
    split_mode: str
    tau: float | None
    test_accuracy: float
    test_macro_f1: float
    test_weighted_f1: float
    test_per_class_f1: dict[int, float]
    test_per_class_recall: dict[int, float]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _try_get_tau_from_config(config: dict) -> float | None:
    tau = config.get("tau", None)
    if tau is None:
        return None
    try:
        return float(tau)
    except Exception:
        return None


def load_experiment(out_dir: Path) -> ExperimentSummary:
    metrics_path = out_dir / "metrics.json"
    config_path = out_dir / "config.json"

    metrics = _read_json(metrics_path)
    cfg = _read_json(config_path) if config_path.exists() else {}

    test = metrics["test"]
    per_class = test["per_class"]
    per_class_f1 = {int(k): float(v["f1"]) for k, v in per_class.items()}
    per_class_recall = {int(k): float(v["recall"]) for k, v in per_class.items()}

    return ExperimentSummary(
        name=out_dir.name,
        split_mode=str(metrics.get("split_mode", cfg.get("split_mode", ""))),
        tau=_try_get_tau_from_config(cfg),
        test_accuracy=float(test["accuracy"]),
        test_macro_f1=float(test["macro_f1"]),
        test_weighted_f1=float(test["weighted_f1"]),
        test_per_class_f1=per_class_f1,
        test_per_class_recall=per_class_recall,
    )


def _mean_for_classes(values: dict[int, float], classes: list[int]) -> float:
    xs = [values[c] for c in classes if c in values]
    if not xs:
        return 0.0
    return float(np.mean(xs))


def _save_bar(
    out_path: Path,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float]]],
) -> None:
    n = len(labels)
    m = len(series)
    x = np.arange(n)
    width = 0.8 / max(m, 1)

    fig = plt.figure(figsize=(max(10, n * 1.2), 5))
    ax = fig.add_subplot(1, 1, 1)
    for i, (name, ys) in enumerate(series):
        ax.bar(x + (i - (m - 1) / 2) * width, ys, width=width, label=name)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_heatmap(
    out_path: Path,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    values: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(max(10, len(col_labels) * 1.6), max(6, len(row_labels) * 0.5)))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(values, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticklabels(row_labels)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_dir", type=str, default="outputs")
    p.add_argument("--out_dir", type=str, default="outputs/_analysis")
    p.add_argument("--minority", type=int, nargs="*", default=[4, 5, 6, 8, 10, 11, 13])
    p.add_argument("--pattern", type=str, default="0514_*_seed42")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_dirs = sorted([p for p in outputs_dir.glob(args.pattern) if p.is_dir()])
    exps = [load_experiment(p) for p in exp_dirs]

    labels = [e.name for e in exps]
    overall = {
        "test_accuracy": [e.test_accuracy for e in exps],
        "test_macro_f1": [e.test_macro_f1 for e in exps],
        "test_weighted_f1": [e.test_weighted_f1 for e in exps],
    }

    minority = [int(x) for x in args.minority]
    minority_f1 = [_mean_for_classes(e.test_per_class_f1, minority) for e in exps]
    minority_recall = [_mean_for_classes(e.test_per_class_recall, minority) for e in exps]

    summary = []
    for e, mf1, mrec in zip(exps, minority_f1, minority_recall):
        summary.append(
            {
                "name": e.name,
                "split_mode": e.split_mode,
                "tau": e.tau,
                "test_accuracy": e.test_accuracy,
                "test_macro_f1": e.test_macro_f1,
                "test_weighted_f1": e.test_weighted_f1,
                "minority_mean_f1": mf1,
                "minority_mean_recall": mrec,
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _save_bar(
        out_path=out_dir / "overall_metrics.png",
        title="Overall Metrics (Test)",
        labels=labels,
        series=[
            ("accuracy", overall["test_accuracy"]),
            ("macro_f1", overall["test_macro_f1"]),
            ("weighted_f1", overall["test_weighted_f1"]),
        ],
    )

    _save_bar(
        out_path=out_dir / "minority_metrics.png",
        title=f"Minority Metrics (Test), classes={minority}",
        labels=labels,
        series=[
            ("minority_mean_f1", minority_f1),
            ("minority_mean_recall", minority_recall),
        ],
    )

    row_labels = [str(c) for c in minority]
    col_labels = labels
    f1_mat = np.array([[exps[j].test_per_class_f1.get(int(c), 0.0) for j in range(len(exps))] for c in minority], dtype=float)
    _save_heatmap(
        out_path=out_dir / "minority_f1_heatmap.png",
        title="Per-class F1 Heatmap (Minority, Test)",
        row_labels=row_labels,
        col_labels=col_labels,
        values=f1_mat,
    )


if __name__ == "__main__":
    main()


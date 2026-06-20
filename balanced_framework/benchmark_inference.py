from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .runtime_utils import (
    build_inference_model,
    evaluate_model,
    load_preprocessed_split,
    quantize_dynamic_linear_layers,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark FP32 and INT8 inference for a trained experiment.")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--x_csv", type=str, default=None)
    p.add_argument("--y_csv", type=str, default=None)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup_runs", type=int, default=200)
    p.add_argument("--timed_runs", type=int, default=5000)
    p.add_argument("--latency_batch_size", type=int, default=1)
    return p.parse_args()


def _set_threads(threads: int) -> int:
    threads = max(1, int(threads))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return int(torch.get_num_threads())


def _measure_latency(
    model: torch.nn.Module,
    x: np.ndarray,
    leaf: np.ndarray | None,
    warmup_runs: int,
    timed_runs: int,
    batch_size: int,
) -> dict[str, Any]:
    if x.shape[0] == 0:
        raise ValueError("Cannot benchmark an empty split")
    if batch_size <= 0:
        raise ValueError("latency_batch_size must be > 0")

    model.eval()
    n = int(x.shape[0])
    batch_size = int(batch_size)
    warmup_runs = max(0, int(warmup_runs))
    timed_runs = max(1, int(timed_runs))

    with torch.no_grad():
        for i in range(warmup_runs):
            start = (i * batch_size) % n
            end = min(start + batch_size, n)
            xb = torch.from_numpy(x[start:end]).to(dtype=torch.float32)
            if leaf is None:
                _ = model(xb)
            else:
                lb = torch.from_numpy(leaf[start:end]).to(dtype=torch.long)
                _ = model(xb, lb)

        times_ms = np.zeros((timed_runs,), dtype=np.float64)
        for i in range(timed_runs):
            start = (i * batch_size) % n
            end = min(start + batch_size, n)
            xb = torch.from_numpy(x[start:end]).to(dtype=torch.float32)
            lb = None if leaf is None else torch.from_numpy(leaf[start:end]).to(dtype=torch.long)
            t0 = time.perf_counter_ns()
            if lb is None:
                _ = model(xb)
            else:
                _ = model(xb, lb)
            t1 = time.perf_counter_ns()
            times_ms[i] = (t1 - t0) / 1_000_000.0

    return {
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "batch_size": batch_size,
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
    }


def _environment_report(args: argparse.Namespace, state: dict[str, Any], actual_threads: int) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": int(os.cpu_count() or 0),
        "threads": actual_threads,
        "benchmark_device": "cpu",
        "model_dir": str(state["model_dir"]),
        "split": str(args.split),
        "warmup_runs": int(args.warmup_runs),
        "timed_runs": int(args.timed_runs),
        "latency_batch_size": int(args.latency_batch_size),
        "eval_batch_size": int(args.eval_batch_size),
    }


def _write_environment_text(out_dir: Path, env: dict[str, Any]) -> None:
    lines = [
        f"platform: {env['platform']}",
        f"python: {env['python']}",
        f"numpy: {env['numpy']}",
        f"torch: {env['torch']}",
        f"cpu: {env['cpu']}",
        f"cpu_count: {env['cpu_count']}",
        f"threads: {env['threads']}",
        f"benchmark_device: {env['benchmark_device']}",
        f"model_dir: {env['model_dir']}",
        f"split: {env['split']}",
        f"warmup_runs: {env['warmup_runs']}",
        f"timed_runs: {env['timed_runs']}",
        f"latency_batch_size: {env['latency_batch_size']}",
        f"eval_batch_size: {env['eval_batch_size']}",
    ]
    (out_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    actual_threads = _set_threads(args.threads)
    x, y, leaf, state = load_preprocessed_split(
        model_dir=args.model_dir,
        split=args.split,
        data_dir=args.data_dir,
        x_csv=args.x_csv,
        y_csv=args.y_csv,
    )

    out_dir = Path(args.out_dir) if args.out_dir else (state["model_dir"] / "benchmark_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_model = build_inference_model(state).cpu().eval()
    int8_model = quantize_dynamic_linear_layers(fp32_model)

    fp32_metrics = evaluate_model(fp32_model, x=x, y=y, leaf=leaf, batch_size=int(args.eval_batch_size))
    int8_metrics = evaluate_model(int8_model, x=x, y=y, leaf=leaf, batch_size=int(args.eval_batch_size))

    fp32_latency = _measure_latency(
        model=fp32_model,
        x=x,
        leaf=leaf,
        warmup_runs=int(args.warmup_runs),
        timed_runs=int(args.timed_runs),
        batch_size=int(args.latency_batch_size),
    )
    int8_latency = _measure_latency(
        model=int8_model,
        x=x,
        leaf=leaf,
        warmup_runs=int(args.warmup_runs),
        timed_runs=int(args.timed_runs),
        batch_size=int(args.latency_batch_size),
    )

    accuracy_compare = {
        "split": str(args.split),
        "num_samples": int(y.shape[0]),
        "fp32": fp32_metrics,
        "int8_dynamic": int8_metrics,
        "delta": {
            "accuracy": float(int8_metrics["accuracy"] - fp32_metrics["accuracy"]),
            "macro_f1": float(int8_metrics["macro_f1"] - fp32_metrics["macro_f1"]),
            "weighted_f1": float(int8_metrics["weighted_f1"] - fp32_metrics["weighted_f1"]),
        },
    }

    latency_compare = {
        "split": str(args.split),
        "fp32": fp32_latency,
        "int8_dynamic": int8_latency,
        "speedup_vs_fp32": float(fp32_latency["mean_ms"] / max(int8_latency["mean_ms"], 1e-12)),
    }

    env = _environment_report(args=args, state=state, actual_threads=actual_threads)
    _write_environment_text(out_dir, env)
    (out_dir / "accuracy_compare.json").write_text(
        json.dumps(accuracy_compare, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "latency_compare.json").write_text(
        json.dumps(latency_compare, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "environment": env,
                "accuracy_compare": accuracy_compare,
                "latency_compare": latency_compare,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_lines = [
        "# Inference Benchmark",
        "",
        f"- split: `{args.split}`",
        f"- num_samples: `{y.shape[0]}`",
        f"- fp32 accuracy: `{fp32_metrics['accuracy']:.6f}`",
        f"- int8 accuracy: `{int8_metrics['accuracy']:.6f}`",
        f"- fp32 macro_f1: `{fp32_metrics['macro_f1']:.6f}`",
        f"- int8 macro_f1: `{int8_metrics['macro_f1']:.6f}`",
        f"- fp32 mean_ms: `{fp32_latency['mean_ms']:.6f}`",
        f"- int8 mean_ms: `{int8_latency['mean_ms']:.6f}`",
        f"- speedup_vs_fp32: `{latency_compare['speedup_vs_fp32']:.6f}`",
        "",
        "## Method",
        "",
        "- Latency is measured on CPU only.",
        "- Input data uses the saved experiment split and saved preprocessing parameters.",
        "- Each run measures a full forward pass of the final inference model, including prior-logit adjustment.",
        "- Statistics report mean/std/min/max/p50/p95/p99 in milliseconds.",
    ]
    (out_dir / "benchmark_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"benchmark saved to {out_dir}")
    print(
        "fp32_acc={fp32_acc:.6f} int8_acc={int8_acc:.6f} fp32_mean_ms={fp32_ms:.6f} int8_mean_ms={int8_ms:.6f}".format(
            fp32_acc=fp32_metrics["accuracy"],
            int8_acc=int8_metrics["accuracy"],
            fp32_ms=fp32_latency["mean_ms"],
            int8_ms=int8_latency["mean_ms"],
        )
    )


if __name__ == "__main__":
    main()

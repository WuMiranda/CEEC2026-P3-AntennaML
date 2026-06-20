from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .runtime_utils import (
    build_base_model,
    build_inference_model,
    load_experiment_state,
    quantize_dynamic_linear_layers,
    serialized_state_size_mb,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate params, MACs, FLOPs, and model size for a trained experiment.")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def _profile_linear_stack(model: nn.Module, input_dim: int) -> dict[str, Any]:
    layer_stats: list[dict[str, Any]] = []
    current_dim = int(input_dim)
    total_macs = 0
    total_flops = 0

    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, nn.Linear):
            in_features = int(module.in_features)
            out_features = int(module.out_features)
            bias_ops = out_features if module.bias is not None else 0
            macs = in_features * out_features
            flops = 2 * macs + bias_ops
            layer_stats.append(
                {
                    "name": name,
                    "type": "Linear",
                    "in_features": in_features,
                    "out_features": out_features,
                    "macs": int(macs),
                    "flops": int(flops),
                }
            )
            total_macs += macs
            total_flops += flops
            current_dim = out_features
        elif isinstance(module, nn.BatchNorm1d):
            num_features = int(module.num_features)
            flops = 4 * num_features
            layer_stats.append(
                {
                    "name": name,
                    "type": "BatchNorm1d",
                    "num_features": num_features,
                    "macs": 0,
                    "flops": int(flops),
                }
            )
            total_flops += flops
            current_dim = num_features
        elif isinstance(module, nn.ReLU):
            flops = current_dim
            layer_stats.append(
                {
                    "name": name,
                    "type": "ReLU",
                    "num_features": int(current_dim),
                    "macs": 0,
                    "flops": int(flops),
                }
            )
            total_flops += flops

    return {
        "layers": layer_stats,
        "macs": int(total_macs),
        "flops": int(total_flops),
    }


def main() -> None:
    args = parse_args()
    state = load_experiment_state(args.model_dir)
    model_dir = state["model_dir"]
    checkpoint = state["checkpoint"]
    config = state["config"]

    out_dir = Path(args.out_dir) if args.out_dir else (model_dir / "profile_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model = build_base_model(checkpoint)
    inference_model = build_inference_model(state)
    quant_model = quantize_dynamic_linear_layers(inference_model)

    input_dim = int(checkpoint["in_dim"])
    num_classes = int(checkpoint["num_classes"])
    hidden_dims = [int(v) for v in checkpoint.get("hidden_dims", (128, 64, 32))]

    total_params = int(sum(p.numel() for p in base_model.parameters()))
    trainable_params = int(sum(p.numel() for p in base_model.parameters() if p.requires_grad))
    complexity = _profile_linear_stack(base_model, input_dim=input_dim)

    report = {
        "model_dir": str(model_dir),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "hidden_dims": hidden_dims,
        "model_type": "frequency_gated_mlp" if int(checkpoint.get("freq_tree_depth", 0)) > 0 else "mlp",
        "estimation_method": {
            "params": "Counts all learnable parameters in the saved PyTorch model.",
            "macs": "Counts only Linear-layer multiply-accumulate operations for single-sample forward inference (batch=1).",
            "flops": "Counts Linear FLOPs as 2*MACs plus bias adds, and also includes BatchNorm1d/ReLU elementwise inference FLOPs.",
            "note": "Latency should be measured separately on the target hardware; MACs/FLOPs are architecture estimates and do not depend on retraining.",
        },
        "fp32": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "macs": int(complexity["macs"]),
            "flops": int(complexity["flops"]),
            "serialized_state_size_mb": serialized_state_size_mb(inference_model),
        },
        "int8_dynamic": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "macs": int(complexity["macs"]),
            "flops": int(complexity["flops"]),
            "serialized_state_size_mb": serialized_state_size_mb(quant_model),
            "quantization_scope": "Dynamic quantization over Linear layers on CPU.",
        },
        "layerwise": complexity["layers"],
        "training_config_excerpt": {
            "tau": float(config.get("tau", 0.0)),
            "feature_dim": int(config.get("feature_dim", input_dim)),
            "rf_features": str(config.get("rf_features", "none")),
            "use_log_magnitude_ratio": bool(config.get("use_log_magnitude_ratio", False)),
            "use_impedance_features": bool(config.get("use_impedance_features", False)),
        },
    }

    json_path = out_dir / "profile_summary.json"
    md_path = out_dir / "profile_summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = [
        "# Model Profile",
        "",
        f"- model_dir: `{model_dir}`",
        f"- input_dim: `{input_dim}`",
        f"- num_classes: `{num_classes}`",
        f"- hidden_dims: `{hidden_dims}`",
        f"- fp32 params: `{total_params}`",
        f"- fp32 MACs: `{complexity['macs']}`",
        f"- fp32 FLOPs: `{complexity['flops']}`",
        f"- fp32 size(MB): `{report['fp32']['serialized_state_size_mb']:.6f}`",
        f"- int8 size(MB): `{report['int8_dynamic']['serialized_state_size_mb']:.6f}`",
        "",
        "## Method",
        "",
        f"- params: {report['estimation_method']['params']}",
        f"- macs: {report['estimation_method']['macs']}",
        f"- flops: {report['estimation_method']['flops']}",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"profile saved to {json_path}")
    print(
        "fp32 params={params} macs={macs} flops={flops} fp32_size_mb={fp32:.6f} int8_size_mb={int8:.6f}".format(
            params=total_params,
            macs=complexity["macs"],
            flops=complexity["flops"],
            fp32=report["fp32"]["serialized_state_size_mb"],
            int8=report["int8_dynamic"]["serialized_state_size_mb"],
        )
    )


if __name__ == "__main__":
    main()

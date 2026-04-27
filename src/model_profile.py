# 在 train.py 训练循环结束后，或 eval.py 加载最佳权重后调用
# 依赖安装：pip install ptflops onnx onnxruntime
# updated: 4.27(wyx)

import os
import torch
import numpy as np
import onnx
import onnxruntime as ort
from ptflops import get_model_complexity_info

def profile_and_export(model, input_dim: int, save_dir: str, device: str = "cpu"):
    """
    统一输出：参数量、MACs、ONNX 文件、精度校验结果
    建议在训练结束后或评估阶段调用
    """
    model.eval()
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    # 1. 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. MACs 计算（ptflops 对 1D 表格特征支持良好）
    # 注意：ptflops 输入形状为不含 batch 的 tuple，表格特征为 (input_dim,)
    macs_str, params_str = get_model_complexity_info(
        model, (input_dim,), as_strings=True, print_per_layer_stat=False
    )

    # 3. ONNX 导出（固定单样本形状，贴合赛题“静态前向”要求）
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32).to(device)
    onnx_path = os.path.join(save_dir, "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=15,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes=None,  # 比赛推荐固定 batch=1，避免图结构复杂化拖慢单样本推理
        export_params=True,
        do_constant_folding=True,  # 关键：折叠 BatchNorm/常量，显著减小体积与推理耗时
        verbose=False
    )

    # 4. ONNX 精度校验（防止导出后数值漂移导致提交失败）
    ort_session = ort.InferenceSession(onnx_path)
    onnx_out = ort_session.run(None, {"features": dummy_input.cpu().numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy_input).cpu().numpy()
    
    max_diff = float(np.max(np.abs(onnx_out - torch_out)))
    if max_diff > 1e-5:
        print(f"⚠️ ONNX 导出精度偏差较大: {max_diff:.2e}，请检查 model.eval() 与 do_constant_folding")

    # 5. 返回结构化结果（日志/报告引用）
    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "macs": macs_str,
        "onnx_size_mb": round(os.path.getsize(onnx_path) / 1024 / 1024, 3),
        "onnx_max_diff": max_diff
    }
    print(f"模型统计: Params={total_params:,} | MACs={macs_str} | ONNX={stats['onnx_size_mb']}MB | Diff={max_diff:.2e}")
    return stats
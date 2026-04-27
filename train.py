# 仅更新使用到集成函数 src/model_profile.py 的部分，应位于训练脚本末尾 （4.27/wyx）
best_model_path = "outputs/exp1/best_model.pth"
model.load_state_dict(torch.load(best_model_path, map_location="cpu"))

from src.model_profiler import profile_and_export
stats = profile_and_export(
    model=model,
    input_dim=FEATURE_DIM,  # 从 A 提供的特征方案中获取
    save_dir="outputs/exp1/",
    device="cpu"
)

# 实验日志供技术报告
with open("outputs/exp1/metrics.csv", "a") as f:
    f.write(f"{stats['total_params']},{stats['macs']},{stats['onnx_size_mb']},{stats['onnx_max_diff']}\n")
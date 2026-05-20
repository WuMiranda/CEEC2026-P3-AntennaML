# Balanced Framework 使用说明

位置：
- 训练与评估框架代码：`balanced_framework/`
- 依赖声明：`setup.py`
- CSV 读取与列对齐：`src/io.py`

快速开始：

```bash
python -m balanced_framework.train_balanced --data_dir 0514 --out_dir outputs/balanced_ce_0514
```

输出（`--out_dir`）：
- `model.pt`：最优模型权重（按验证集 macro-F1 选优）
- `preprocess.npz`：标准化参数、训练集先验等（推理复现用）
- `metrics.json`：accuracy / macro-F1 / weighted-F1 / per-class 指标
- `confusion_val.png`、`confusion_test.png`：混淆矩阵图
- `config.json`：训练配置与超参

更多参数说明见 [balanced_framework/README.md](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/balanced_framework/README.md)。


# Balanced Framework（长尾分类可复现训练）

本目录提供一个“可复现 + 更工程化”的训练/评估框架，用于解决类别不均衡（少数类样本少）带来的性能问题。

核心策略：
- 均衡采样：按类别频次生成 `WeightedRandomSampler`
- Class-Balanced 权重（Effective Number of Samples）
- Logit Adjustment（按训练集先验修正 logits）
- 以 `macro-F1` 作为“少数类友好”的主指标，并保存最优模型

## 依赖

建议新建非 base 环境（示例）：
- Python >= 3.10
- numpy
- pandas
- torch
- matplotlib

依赖清单见项目根目录的 [setup.py](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/setup.py)（你自行在目标环境安装即可）。

## 输入数据格式

默认读取 `--data_dir` 下的：
- `x_train.csv`
- `y_train.csv`

特征列名与编码容错读取由 [io.py](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/src/io.py) 负责。

## 训练（6/2/2 划分）

从仓库根目录运行：

```bash
python -m balanced_framework.train_balanced --data_dir 0514 --out_dir outputs/balanced_ce_0514
```

常用参数：
- `--split 0.6 0.2 0.2`：train/val/test 比例（默认 6/2/2）
- `--hierarchical_col itState`：层级划分（默认开启；在每个 label 内按 itState 分桶后再划分）
- `--epochs 50`
- `--batch_size 256`
- `--tau 1.0`：Logit Adjustment 系数（0 表示关闭）

## 输出产物

输出目录（`--out_dir`）包含：
- `model.pt`：最优模型参数（以 val macro-F1 选优）
- `preprocess.npz`：标准化参数、itState 类别数、训练集先验等（推理复现用）
- `metrics.json`：accuracy / macro-F1 / weighted-F1 / per-class 指标
- `confusion_val.png`、`confusion_test.png`：混淆矩阵图
- `config.json`：本次训练的配置（seed、split、超参等）


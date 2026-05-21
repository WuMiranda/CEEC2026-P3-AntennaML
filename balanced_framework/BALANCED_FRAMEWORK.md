# Balanced Framework 使用说明

## 框架简介

本框架面向“类别不均衡/长尾分类”的工程训练需求，目标是在不改变数据标注的前提下，提升少数类的可学习性与最终预测效果，并且把训练过程做成可复现、可追溯的实验闭环。

典型场景：

- 少数类样本量只有几百，大类样本量达到几千甚至更多
- 直接用普通交叉熵训练会偏向大类，导致少数类 recall/F1 很差

设计原则：

- 可复现：固定随机种子、固定数据切分方式、保存完整配置与预处理参数
- 不数据泄露：所有会“学习数据分布”的步骤（标准化、类别先验/权重等）只在 train split 上拟合
- 少数类友好：以 macro-F1 为主指标选模，并通过重加权/重采样/先验修正改善长尾

方法概览（训练时同时生效）：

- 均衡采样：`WeightedRandomSampler`，让少数类在训练 batch 中出现更频繁
- Class-Balanced 权重：基于 Effective Number of Samples 计算类别权重，缓解长尾梯度主导
- Logit Adjustment：按训练集类别先验修正 logits，降低模型对大类的先验偏置（`--tau` 控制强度）

数据切分策略：

- 默认做 6/2/2（train/val/test），并支持“层级切分”
- 层级切分含义：在每个 label 内，再按某个层级字段（默认 `itState`）分桶后分别切分，减少层级因素导致的分布偏移

输出产物规范：

- 每次实验输出到独立 `--out_dir`，包含模型、预处理参数、配置、指标与混淆矩阵图

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

## 输出文件

- `metrics.json`
  - `val.macro_f1`：选模主指标（更关注少数类）
  - `test.macro_f1`：最终对外汇报主指标之一
  - `per_class`：逐类 precision/recall/f1/support，直接定位哪些类仍是短板
- `confusion_test.png`
  - 用于展示少数类主要被错分到哪些大类，说明“错误模式”与后续改进方向
- `config.json`
  - 用于复现实验（seed、split、是否层级切分、tau/beta 等）
- `model.pt` + `preprocess.npz`
  - 用于部署/推理复现（模型参数 + 标准化参数 + 先验修正参数）

更多参数说明见 [balanced\_framework/README.md](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/balanced_framework/README.md)。

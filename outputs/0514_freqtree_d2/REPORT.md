# 0514_freqtree_d2 实验报告

本目录为一次“频率树（仅依赖 closeFreqMHz）→ 路由到多 Head MLP（共享 trunk）”的训练与评估产物，包含完整复现所需的配置、切分索引、模型与指标。

## 1. 实验配置（来自 config.json）

- 数据：`0514/x_train.csv` + `0514/y_train.csv`
- 切分：6/2/2，`seed=42`
- 切分策略：`hierarchical_within_label:itState`（在每个 label 内再按 itState 分桶切分）
- 特征：
  - `itState`：`bits`（4bit）
  - 频率分箱：`freq_bins=10`（分箱边界仅在 train split 拟合）
  - 派生物理量：`derived_features=true`
  - 复平面几何残差：`use_geometry_features=false`
- 长尾策略：
  - `beta=0.999`（Class-Balanced Effective Number）
  - `tau=1.0`（Logit Adjustment 强度）
- 频率树：
  - `freq_tree_depth=2`（4 个叶子频段）
  - `freq_tree_min_leaf=200`
  - `freq_tree_candidates=128`
- 训练：
  - `epochs=50`
  - `batch_size=256`
  - `lr=1e-3`
  - `weight_decay=1e-4`
  - `device_auto=cuda`

## 2. 频率树结构（来自 freq_tree.json）

该树仅使用 `closeFreqMHz` 分裂，阈值如下（深度 2 → 4 个叶子）：

- 根节点：`f <= 748` vs `f > 748`
- 左子树：`f <= 727` vs `727 < f <= 748`
- 右子树：`f <= 839` vs `f > 839`

因此 4 个频段叶子可理解为：

- Leaf 0：`f <= 727`
- Leaf 1：`727 < f <= 748`
- Leaf 2：`748 < f <= 839`
- Leaf 3：`f > 839`

## 3. 总体指标（来自 metrics.json）

### Val

- accuracy：0.8770
- macro-F1：0.8057
- weighted-F1：0.8819

### Test

- accuracy：0.8790
- macro-F1：0.8108
- weighted-F1：0.8833

## 4. 少数类表现（基于 minority classes：4, 5, 6, 8, 10, 11, 13）

### Test（少数类逐类）

| label | support | precision | recall | f1 |
|---:|---:|---:|---:|---:|
| 4 | 238 | 0.3174 | 0.6429 | 0.4250 |
| 5 | 617 | 0.8436 | 0.8217 | 0.8325 |
| 6 | 842 | 0.8476 | 0.7268 | 0.7826 |
| 8 | 652 | 0.6166 | 0.5230 | 0.5660 |
| 10 | 155 | 0.3597 | 0.9677 | 0.5245 |
| 11 | 444 | 0.7321 | 0.8986 | 0.8069 |
| 13 | 484 | 0.9396 | 0.9959 | 0.9669 |

- 少数类平均 recall：约 0.7967
- 少数类平均 f1：约 0.7006

观察要点（用于汇报解释）：
- label 10：recall 很高但 precision 偏低，说明该类“宁可多报也要抓住”，适合讨论 tau/leaf 内 priors 的进一步调参。
- label 4：precision 与 f1 最弱，属于“最难少数类”，适合在混淆矩阵中定位主要误分去向。

## 5. 可视化产物（建议汇报时展示）

- `confusion_val.png`：验证集混淆矩阵
- `confusion_test.png`：测试集混淆矩阵

建议在汇报中：
- 先给总体指标（accuracy / macro-F1 / weighted-F1）
- 再给少数类（4/5/6/8/10/11/13）的逐类表格或热力图
- 最后用 `confusion_test.png` 指出最难类别（如 4、10）主要错到哪些大类

## 6. 可复现与组内统一对比（关键文件）

本次切分索引已保存，可被任何组员脚本复用以确保对比公平：

- `train_idx.npy`
- `val_idx.npy`
- `test_idx.npy`

规则：所有“拟合”类操作（标准化参数、频率分箱边界、圆参数、树阈值等）只允许在 train split 上 fit；val/test 只能 transform/路由。

## 7. 交付物（推理复现所需）

- `model.pt`：最优模型（按 val macro-F1 选优）
- `preprocess.npz`：标准化参数、频率分箱边界、频率树数组、leaf priors 等


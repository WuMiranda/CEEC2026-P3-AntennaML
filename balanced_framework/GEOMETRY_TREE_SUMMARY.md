# 复平面几何特征与树模型计划

本文件用于把以下三部分串起来，形成可复用的技术叙事与后续落地计划：

- 物理背景与复平面规律（[说明.md](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/%E8%AF%B4%E6%98%8E.md) 与 [complex\_plane](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/complex_plane)）
- 当前长尾训练框架与已有结果（[balanced\_framework](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/balanced_framework) 与 [outputs](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/outputs)）
- 基于复平面“圆/圆心”特征引入树模型的稳健做法与实施步骤（参考 [tmp.py](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/tmp.py)）

## 1. 背景串起来：从物理到特征

数据包含两种工作模式下的反射系数：

- 闭合态：(\Gamma\_1=\Gamma\_{1Re}+j\Gamma\_{1Im})
- 默认态：(\Gamma\_2=\Gamma\_{2Re}+j\Gamma\_{2Im})
- 频率：`closeFreqMHz`
- 控制位：`itState`（0\~15）

物理上反射系数：
\[
\Gamma=\frac{Z\_{in}-Z\_0}{Z\_{in}+Z\_0}
]
\[
\Gamma = |\Gamma|e^{j\theta}=\mathrm{Re}(\Gamma)+j\mathrm{Im}(\Gamma)
]

[complex\_plane](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/complex_plane) 的可视化表明，不同 label 在复平面上呈现出不同的聚集形态/轨迹特征。对“复平面同心/偏心圆”进行参数化是合理的特征工程方向，例如：
\[
(\Gamma\_{Re}-x\_0)^2+(\Gamma\_{Im}-y\_0)^2=\rho^2
]

这类几何参数（圆心/半径）或“到圆的残差距离”可以把复杂的点云结构压缩成更易分类的数值特征。

## 2. 提到的“30圆心 + 频率”与 tmp.py 的对应关系

[tmp.py](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/tmp.py) 的做法是（监督式几何特征）：

- 对每个 label，在训练集上分别拟合两条圆：
  - (\Gamma\_1) 平面拟合一条圆，得到 ((c\_{x1},c\_{y1},r\_1))
  - (\Gamma\_2) 平面拟合一条圆，得到 ((c\_{x2},c\_{y2},r\_2))
- 对任意样本，构造特征：
  - 对每个 label 的圆，计算点到圆的“半径残差”：
    \[
    \mathrm{residual} = \left|\sqrt{(x-c\_x)^2+(y-c\_y)^2}-r\right|
    ]
  - (\Gamma\_1) 产生 16 维残差、(\Gamma\_2) 产生 16 维残差（若 label=16 类），共 32 维
  - 额外加入 (\Gamma\_1) 与 (\Gamma\_2) 的相对角度（tmp.py 的 `rel_angle`）

因此它本质上是“2×(每类1个圆心/半径) + 频率/角度”等几何特征，与“30 个圆心”思路高度一致，只是：

- tmp.py 是“按 label 拟合圆”（监督式）
- 更好的是无监督聚类圆心”（监督风险更低）

## 4. itState 的稳健表达：从 16 one-hot 改为 4 个二进制开关位

[说明.md](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/%E8%AF%B4%E6%98%8E.md) 指出 `itState` 实质是 4bit（0000\~1111）的拓扑控制位。

因此更物理、更紧凑的表达是：
\[
b\_0 = itState & 1,\quad b\_1=(itState>>1)&1,\quad b\_2=(itState>>2)&1,\quad b\_3=(itState>>3)&1
]

收益：

- 维度更低、冗余更少
- 更利于树模型学习“某一位控制开关与几何特征的交互”

## 5. 与当前 balanced\_framework 的关系：已有结果在回答什么

当前 [train\_balanced.py](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/balanced_framework/train_balanced.py) 是“通用表格学习 + 长尾处理”的基线闭环：

- 均衡采样（WeightedRandomSampler）
- Class-Balanced 类权重（Effective Number of Samples）
- Logit Adjustment（(\tau) 控制强度）

4 组实验的作用是做 2×2 消融：

- 是否层级切分：`hier-off` vs `hier-itState`
- logit adjustment 强度：(\tau=0.5) vs (\tau=1.0)

从已汇总的结果看：

- `tau=0.5` 在两种切分下都更稳
- 层级切分能进一步提升少数类均值指标，但会牺牲整体 accuracy（典型 trade-off）

## 6. 为什么要接入树模型

在“复平面几何特征（圆心/残差距离）+ 频率 + itState(4bit)”这一类表格特征上：

- 树模型更擅长学习阈值/分段规则与高阶交互
- 对冗余高维（例如 30\~40 个几何残差特征）更鲁棒，会自动选择最有效的特征
- 可解释性更强（feature importance/分裂规则）更适合答辩

## 7. 接入树模型的落地计划

阶段 A（不改现有训练闭环，只新增树模型脚本）：

- 新增 `balanced_framework/train_tree.py`
  - 复用 `src/io.py` 做 CSV 读入与校验
  - 复用 `balanced_framework/splits.py` 的 split 与层级切分逻辑
  - 在 train split 上拟合 `circle_params`（按 tmp.py 的 `fit_label_circles`）
  - 用 `build_three_feature_matrix` 构造几何特征 + 加入 `freq` + `itState(4bit)`
  - 训练树模型（优先建议梯度提升树；若环境受限可先 RandomForest）
  - 输出目录结构对齐：`metrics.json`、`confusion_test.png`、`config.json`、`circle_params.*`

阶段 B（把“几何特征”也接入当前 MLP 框架，做模型对比）：

- 在 `preprocess.py` 增加开关：选择“原始特征/派生特征/几何残差特征”
- 用同一个 split 与 seed 比较：
  - MLP + 长尾三件套
  - Tree + class\_weight/sample\_weight（对应长尾）

阶段 C（降低监督式几何特征的过拟合风险）：

- 将“按 label 拟合圆”替换/补充为“无监督聚类中心”（如 KMeans 15+15）
- 或者把圆拟合改成“按 itState 分桶拟合圆”，再把残差作为特征（更贴近物理控制位）

## 8. 建议的实验矩阵（最少成本，最大信息量）

固定：

- seed 与 split（例如 0.6/0.2/0.2）
- minority classes（4/5/6/8/10/11/13）

对比维度：

- 模型：MLP vs Tree
- 特征：原始/派生 vs 几何残差（tmp.py） vs 无监督圆心
- 切分：hier-off vs hier-itState

汇报指标：

- overall：accuracy / macro-F1 / weighted-F1
- minority：少数类平均 F1/Recall + 少数类逐类热力图（用 `analyze_outputs.py`）

## 9. 与组员统一 split（同 seed）再复现：具体操作口径

为了让“几何特征/树模型/MLP”在同一划分上公平对比，建议统一使用本框架的切分结果：

- 运行任意一次训练后（例如 `train_balanced.py`），输出目录会自动保存：
  - `train_idx.npy`
  - `val_idx.npy`
  - `test_idx.npy`

它们就是对原始 CSV 行号的索引，可被任何组员脚本复用。组员迁移方式：
- 用同一份 `x_train.csv/y_train.csv` 读入得到原始顺序数组 \(X\) 与 \(y\)
- 用 `np.load(train_idx.npy)` 取子集 \(X_{train},y_{train}\)，同理得到 val/test
- 所有“拟合”类操作（圆参数、聚类中心、标准化等）只允许在 train 子集上 fit

这样可以确保：
- 各方案对比只反映“特征/模型”差异，而不被切分随机性干扰
- 可复现与可追溯（索引文件就是切分的事实来源）

## 10. 框架已实现的增强选项（物理特征 + itState 4bit + 频率分箱 + 几何残差）

从现在起，[train_balanced.py](file:///e:/OneDrive/%E7%A0%94%E7%94%B5%E8%B5%9B/CEEC2026-P3-AntennaML/balanced_framework/train_balanced.py) 支持以下增强选项：

- `itState` 编码方式：
  - `--it_encoding bits`（默认）：将 itState 转为 4bit（\(b_0..b_3\)），更贴近“4个控制位”的物理含义
  - `--it_encoding onehot`：保留 16 维 one-hot（兼容旧设定）
- 频率分箱（避免泄露：只用 train split 拟合分箱边界）：
  - `--freq_bins 10`（默认）：基于 train 的频率范围做等距分箱并 one-hot
  - `--no_freq_bins`：关闭分箱（只保留连续频率特征并由标准化处理）
- 几何残差特征（参考 tmp.py，避免泄露：只用 train split 拟合圆参数）：
  - `--geometry_mode full`：在特征中追加“到每个 label 圆的残差 + 相对角度”（兼容参数 `--use_geometry_features` 等价于 full）
  - 输出目录会保存 `circle_params.tsv`，记录每个 label 的圆心与半径

同时，为避免对 one-hot/bit 等离散特征做不必要的标准化，预处理使用“按列掩码标准化”：只对连续特征列做 z-score，离散列保持原值。

### 运行示例

- 基线（itState=4bit，freq_bins=10，派生物理量=开启）：
\`\`\`bash
python -m balanced_framework.train_balanced --data_dir 0514 --out_dir outputs/0514_phys_base
\`\`\`

- 开几何残差（追加 tmp.py 思路特征，维度较高）：
\`\`\`bash
python -m balanced_framework.train_balanced --data_dir 0514 --geometry_mode full --out_dir outputs/0514_phys_geo
\`\`\`

- 关闭频率分箱（只用连续频率 + 标准化）：
\`\`\`bash
python -m balanced_framework.train_balanced --data_dir 0514 --no_freq_bins --out_dir outputs/0514_phys_nobin
\`\`\`

- 少特征版本（每个特征具备明确物理含义）：
  - `feature_set=phys_min`：保留频率 + 反射强度/变化幅度 + 相位差/相对角
  - `freq_bin_mode=index`：仅增加一个“频段编号”，避免 one-hot 维度膨胀

\`\`\`bash
python -m balanced_framework.train_balanced --data_dir 0514 --feature_set phys_min --it_encoding bits --freq_bin_mode index --freq_bins 10 --geometry_mode none --out_dir outputs/0514_phys_min
\`\`\`

- 几何残差的“压缩版”（保留物理意义、减少维度）：
  - `geometry_mode=summary`：只保留 g1/g2 到各类圆的最小残差、次小残差与 gap（6 维），用于表达“最接近哪个原型以及区分度”

\`\`\`bash
python -m balanced_framework.train_balanced --data_dir 0514 --feature_set phys_min --geometry_mode summary --freq_bin_mode index --freq_bins 10 --out_dir outputs/0514_phys_min_geom_summary
\`\`\`

## 11. 频率树嫁接到 MLP 前：按频段分段建模（与物理意义一致）

### 11.1 动机（为什么分叉必须基于频率）

在天线/匹配网络系统里，频率变化会直接改变输入阻抗与反射轨迹形态，因此“同一组几何特征在不同频段的判别规则可能不同”。若强行用单一模型覆盖全频段，模型往往会学到折中边界，导致少数类或边界类被牺牲。

因此引入“频率树（Frequency Tree）”作为 gating：仅用 `closeFreqMHz`（`x_train.csv` 第三列）学习一棵一维二叉树，把样本按频段切成若干子域，再在每个子域上用 MLP head 学习更合适的分类边界。

### 11.2 模型形式（Tree → MLP）

该结构可以理解为“按频率分段的 Mixture-of-Experts”，但 gating 是确定性的树路由：
- Tree 节点仅包含阈值 \(t\)，规则为：
  \[
  f \le t \Rightarrow \text{left},\quad f > t \Rightarrow \text{right}
  \]
- 落到某个 leaf 后，使用对应的 MLP head 做分类；同时各 leaf 共享同一条 trunk（特征抽取层），避免训练成本随 leaf 数线性爆炸。

### 11.3 训练方式（避免泄露）

- 频率树只用 train split 的 \((f_{train}, y_{train})\) 拟合（CART 的一维分裂，指标为 Gini）
- val/test 只做路由，不参与任何阈值学习
- 输出保存：
  - `freq_tree.json`：树结构
  - `preprocess.npz`：树数组与 leaf priors（推理复现用）

### 11.4 长尾处理如何对齐到 leaf

当启用频率树时，本框架会在每个 leaf 内统计训练集类别分布，并计算：
- leaf priors（带平滑）：用于 leaf 的 Logit Adjustment
- leaf 的 Class-Balanced 权重：用于 loss 的样本加权

这样可以保证：每个频段子域内，仍然针对长尾分布进行纠偏。

### 11.5 运行示例

\`\`\`bash
python -m balanced_framework.train_balanced --data_dir 0514 --freq_tree_depth 2 --out_dir outputs/0514_freqtree_d2
\`\`\`

推荐先固定 `tau=0.5` 并对比：
- `--freq_tree_depth 0`（关闭 tree，单模型）
- `--freq_tree_depth 1/2`（浅层分段）

并且强制与组内统一 split 对齐：复用 `train_idx.npy/val_idx.npy/test_idx.npy`。

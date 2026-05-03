import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体（防止中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =====================
# 1. 读取数据 + 数据质量检查
# =====================
print("=" * 60)
print("1. 读取数据及数据质量检查")
print("=" * 60)

# 读取数据（尝试不同编码）
try:
    X = pd.read_csv("x_train.csv", encoding="utf-8")
except UnicodeDecodeError:
    try:
        X = pd.read_csv("x_train.csv", encoding="gbk")
        print("使用GBK编码读取数据")
    except:
        X = pd.read_csv("x_train.csv", encoding="gb18030")
        print("使用GB18030编码读取数据")

y = pd.read_csv("y_train.csv", encoding="gbk")  # 标签文件通常不会有问题
y = y.values.squeeze()

print(f"原始数据形状: X={X.shape}, y={y.shape}")

# 给列命名（根据赛题描述）
X.columns = ['gammaIn1Re', 'gammaIn1Im', 'closeFreqMHz',
             'itState', 'gammaIn2Re', 'gammaIn2Im']

# ===== 数据质量检查 =====
print("\n【数据质量检查】")

# 检查缺失值
missing_count = X.isnull().sum()
if missing_count.sum() > 0:
    print(f"⚠ 发现缺失值:\n{missing_count[missing_count > 0]}")
    # 缺失值处理：用中位数填充
    for col in X.columns:
        if missing_count[col] > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"  - {col}: 用中位数 {median_val:.4f} 填充 {missing_count[col]} 个缺失值")
else:
    print("✓ 无缺失值")

# 检查无穷值
inf_count = np.isinf(X.values).sum()
if inf_count > 0:
    print(f"⚠ 发现无穷值 {inf_count} 个，替换为NaN并用中位数填充")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
else:
    print("✓ 无无穷值")

# 检查异常值（使用IQR方法识别，但不删除，仅标记）
print("\n【异常值检测（IQR方法）】")
outlier_summary = {}
for col in X.columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
    outlier_summary[col] = {
        '异常值数量': outliers,
        '占比': f"{outliers / len(X) * 100:.2f}%",
        '下界': f"{lower_bound:.4f}",
        '上界': f"{upper_bound:.4f}"
    }

    if outliers > 0:
        print(f"  {col}: {outliers} 个异常值 ({outliers / len(X) * 100:.2f}%)")
        # 对异常值进行截尾处理（Winsorize）
        X[col] = X[col].clip(lower_bound, upper_bound)

print("\n✓ 异常值已进行截尾处理（Winsorize）")

# 数据基本统计
print("\n【数据基本统计】")
print(X.describe().round(4))

# 类别分布检查
unique_classes, class_counts = np.unique(y, return_counts=True)
print(f"\n【类别分布】")
print(f"总类别数: {len(unique_classes)}")
print(f"最小类别样本数: {class_counts.min()}, 最大类别样本数: {class_counts.max()}")
if class_counts.max() / class_counts.min() > 3:
    print("⚠ 类别不平衡，建议使用加权损失函数或过采样")
else:
    print("✓ 类别分布相对均衡")

# =====================
# 2. 特征工程（增强版）
# =====================
print("\n" + "=" * 60)
print("2. 特征工程（增强版）")
print("=" * 60)

X_np = X.values
g1_re, g1_im = X_np[:, 0], X_np[:, 1]  # gammaIn1 实部/虚部
freq = X_np[:, 2]  # 闭环中心频点
it_state = X_np[:, 3].astype(int)  # 开关状态（分类特征）
g2_re, g2_im = X_np[:, 4], X_np[:, 5]  # gammaIn2 实部/虚部

# ----- 基础特征 -----
g1_mag = np.sqrt(g1_re ** 2 + g1_im ** 2)  # |Γ1| 反射系数幅度
g2_mag = np.sqrt(g2_re ** 2 + g2_im ** 2)  # |Γ2| 反射系数幅度
g1_phase = np.arctan2(g1_im, g1_re)  # ∠Γ1 反射系数相位
g2_phase = np.arctan2(g2_im, g2_re)  # ∠Γ2 反射系数相位

# ----- 物理意义特征 -----
# 1. VSWR（电压驻波比）: VSWR = (1+|Γ|)/(1-|Γ|)
# 反映阻抗失配程度，VSWR=1表示完全匹配
epsilon = 1e-8  # 防止除零
g1_vswr = (1 + g1_mag) / (1 - np.clip(g1_mag, 0, 1 - epsilon) + epsilon)
g2_vswr = (1 + g2_mag) / (1 - np.clip(g2_mag, 0, 1 - epsilon) + epsilon)

# 2. 反射损耗（Return Loss）: RL = -20*log10(|Γ|)
# 反射损耗越大越好，表示反射能量越少
g1_rl = -20 * np.log10(np.clip(g1_mag, epsilon, 1))
g2_rl = -20 * np.log10(np.clip(g2_mag, epsilon, 1))

# 3. 差值特征
delta_mag = g1_mag - g2_mag
delta_phase = g1_phase - g2_phase
delta_vswr = g1_vswr - g2_vswr
delta_rl = g1_rl - g2_rl

# 4. 比值特征
ratio_mag = g1_mag / (g2_mag + epsilon)
ratio_vswr = g1_vswr / (g2_vswr + epsilon)

# 5. 交叉特征
mag_product = g1_mag * g2_mag
rl_sum = g1_rl + g2_rl

# 6. 频点相关特征（如果有多次测量，freq可能不是常数）
freq_normalized = (freq - np.mean(freq)) / (np.std(freq) + epsilon)

# ----- 构建连续特征矩阵 -----
X_continuous = np.stack([
    # 原始特征
    g1_re, g1_im,  # gammaIn1 实部/虚部（2维）
    g2_re, g2_im,  # gammaIn2 实部/虚部（2维）
    freq_normalized,  # 归一化频点（1维）

    # 幅度/相位
    g1_mag, g2_mag,  # 反射系数幅度（2维）
    g1_phase, g2_phase,  # 反射系数相位（2维）

    # VSWR和反射损耗
    g1_vswr, g2_vswr,  # 电压驻波比（2维）
    g1_rl, g2_rl,  # 反射损耗（2维）

    # 差值特征
    delta_mag,  # 幅度差（1维）
    delta_phase,  # 相位差（1维）
    delta_vswr,  # VSWR差（1维）
    delta_rl,  # 反射损耗差（1维）

    # 比值特征
    ratio_mag,  # 幅度比（1维）
    ratio_vswr,  # VSWR比（1维）

    # 交叉特征
    mag_product,  # 幅度乘积（1维）
    rl_sum,  # 反射损耗和（1维）
], axis=1)

print(f"连续特征维度: {X_continuous.shape[1]}")

# ----- 处理分类特征 itState -----
# itState是开关状态，属于离散分类特征，进行One-Hot编码
print(f"itState 取值范围: {np.unique(it_state)}")
print(f"itState 唯一值个数: {len(np.unique(it_state))}")

# One-Hot编码
it_state_reshaped = it_state.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
it_onehot = encoder.fit_transform(it_state_reshaped)
print(f"itState One-Hot编码后维度: {it_onehot.shape[1]}")

# ----- 合并所有特征 -----
X_feat = np.hstack([X_continuous, it_onehot])
print(f"最终特征维度: {X_feat.shape[1]}")

num_classes = len(unique_classes)
print(f"类别数: {num_classes}")

# =====================
# 3. 划分数据（增加测试集）
# =====================
print("\n" + "=" * 60)
print("3. 划分数据集")
print("=" * 60)

# 先分出测试集（10%）
X_temp, X_test, y_temp, y_test = train_test_split(
    X_feat, y, test_size=0.1, random_state=42, stratify=y
)

# 再从剩余数据分出验证集（最终：训练70%，验证20%，测试10%）
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2222, random_state=42, stratify=y_temp  # 0.2222*0.9=0.2
)

print(f"训练集: {X_train.shape} ({X_train.shape[0] / len(X_feat) * 100:.1f}%)")
print(f"验证集: {X_val.shape} ({X_val.shape[0] / len(X_feat) * 100:.1f}%)")
print(f"测试集: {X_test.shape} ({X_test.shape[0] / len(X_feat) * 100:.1f}%)")

# 检查划分后的类别分布
print("\n【划分后类别分布】")
for name, y_data in [("训练集", y_train), ("验证集", y_val), ("测试集", y_test)]:
    unique, counts = np.unique(y_data, return_counts=True)
    print(f"{name}: {dict(zip(unique, counts))}")

# =====================
# 4. 标准化
# =====================
print("\n" + "=" * 60)
print("4. 标准化处理")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"训练集均值范围: [{X_train_scaled.mean(axis=0).min():.4f}, {X_train_scaled.mean(axis=0).max():.4f}]")
print(f"训练集标准差范围: [{X_train_scaled.std(axis=0).min():.4f}, {X_train_scaled.std(axis=0).max():.4f}]")

# 转换为PyTorch张量
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# =====================
# 5. 模型构建（考虑类别不平衡）
# =====================
print("\n" + "=" * 60)
print("5. 构建模型")
print("=" * 60)


class MLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = in_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))  # 批归一化提升训练稳定性
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


input_dim = X_train_t.shape[1]
model = MLP(input_dim, num_classes, hidden_dims=[128, 64, 32], dropout=0.3)


# 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


total_params = count_parameters(model)
print(f"模型参数量: {total_params:,}")

# 检查是否需要类别加权损失
class_counts = np.bincount(y_train)
if class_counts.max() / (class_counts.min() + 1e-8) > 2:
    print("⚠ 检测到类别不平衡，使用加权交叉熵损失")
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
else:
    loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=8
)

# =====================
# 6. 训练
# =====================
print("\n" + "=" * 60)
print("6. 开始训练")
print("=" * 60)

train_losses = []
val_accs = []
test_accs = []
best_val_acc = 0.0
patience = 15
patience_counter = 0

num_epochs = 100

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)

    # 验证阶段
    model.eval()
    with torch.no_grad():
        # 验证集
        logits_val = model(X_val_t)
        preds_val = logits_val.argmax(dim=1)
        val_acc = (preds_val == y_val_t).float().mean().item()
        val_accs.append(val_acc)

        # 测试集
        logits_test = model(X_test_t)
        preds_test = logits_test.argmax(dim=1)
        test_acc = (preds_test == y_test_t).float().mean().item()
        test_accs.append(test_acc)

    # 学习率调度
    scheduler.step(val_acc)

    # 早停检查
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        torch.save(model.state_dict(), "best_mlp_model.pt")
        print(f"✓ 保存最佳模型 (val_acc={val_acc:.4f}, test_acc={test_acc:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 5 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch + 1:3d}/{num_epochs}: loss={avg_loss:.4f}, "
              f"val_acc={val_acc:.4f}, test_acc={test_acc:.4f}, "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

    if patience_counter >= patience:
        print(f"\n早停触发！在第 {epoch + 1} 轮停止训练")
        break

print(f"\n训练完成！")
print(f"最佳验证准确率: {best_val_acc:.4f}")
print(f"对应测试准确率: {best_test_acc:.4f}")

# =====================
# 7. 绘制结果图
# =====================
print("\n" + "=" * 60)
print("7. 绘制结果图")
print("=" * 60)

model.load_state_dict(torch.load("best_mlp_model.pt"))
model.eval()

with torch.no_grad():
    # 测试集最终预测
    final_preds_test = model(X_test_t).argmax(dim=1).numpy()
    # 验证集预测
    final_preds_val = model(X_val_t).argmax(dim=1).numpy()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- 图1：训练损失曲线 ---
ax1 = axes[0, 0]
ax1.plot(range(1, len(train_losses) + 1), train_losses, color='#E74C3C',
         linewidth=2, marker='o', markersize=2, label='训练损失')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12, color='#E74C3C')
ax1.tick_params(axis='y', labelcolor='#E74C3C')
ax1.set_title('训练损失曲线', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# --- 图2：验证/测试准确率曲线 ---
ax2 = axes[0, 1]
ax2.plot(range(1, len(val_accs) + 1), val_accs, color='#3498DB',
         linewidth=2, marker='s', markersize=2, label='验证准确率')
ax2.plot(range(1, len(test_accs) + 1), test_accs, color='#2ECC71',
         linewidth=2, marker='^', markersize=2, label='测试准确率')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('准确率', fontsize=12)
ax2.set_title('验证/测试准确率曲线', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axhline(y=best_val_acc, color='#3498DB', linestyle='--', alpha=0.5)
ax2.axhline(y=best_test_acc, color='#2ECC71', linestyle='--', alpha=0.5)

# --- 图3：验证集混淆矩阵 ---
ax3 = axes[1, 0]
cm_val = confusion_matrix(y_val, final_preds_val)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=[f'类{i}' for i in range(num_classes)],
            yticklabels=[f'类{i}' for i in range(num_classes)],
            cbar_kws={'label': '样本数'})
ax3.set_xlabel('预测标签', fontsize=12)
ax3.set_ylabel('真实标签', fontsize=12)
ax3.set_title('验证集混淆矩阵', fontsize=14, fontweight='bold')

# --- 图4：测试集混淆矩阵 ---
ax4 = axes[1, 1]
cm_test = confusion_matrix(y_test, final_preds_test)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=[f'类{i}' for i in range(num_classes)],
            yticklabels=[f'类{i}' for i in range(num_classes)],
            cbar_kws={'label': '样本数'})
ax4.set_xlabel('预测标签', fontsize=12)
ax4.set_ylabel('真实标签', fontsize=12)
ax4.set_title('测试集混淆矩阵', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('training_results.png', dpi=200, bbox_inches='tight')
print("图表已保存为 training_results.png")
plt.show()

# =====================
# 8. 分类报告
# =====================
print("\n" + "=" * 60)
print("8. 验证集分类报告")
print("=" * 60)
print(classification_report(y_val, final_preds_val))

print("\n" + "=" * 60)
print("9. 测试集分类报告")
print("=" * 60)
print(classification_report(y_test, final_preds_test))

# 特征重要性分析（简单方法：计算第一层权重的绝对值）
print("\n" + "=" * 60)
print("10. 特征重要性分析（第一层权重）")
print("=" * 60)
first_layer_weights = model.net[0].weight.data.abs().mean(dim=0).numpy()
feature_names = (
        ['g1_re', 'g1_im', 'g2_re', 'g2_im', 'freq_norm',
         'g1_mag', 'g2_mag', 'g1_phase', 'g2_phase',
         'g1_vswr', 'g2_vswr', 'g1_rl', 'g2_rl',
         'delta_mag', 'delta_phase', 'delta_vswr', 'delta_rl',
         'ratio_mag', 'ratio_vswr', 'mag_product', 'rl_sum'] +
        [f'itState_{i}' for i in range(it_onehot.shape[1])]
)

# 按重要性排序并显示Top-10
feature_importance = list(zip(feature_names, first_layer_weights))
feature_importance.sort(key=lambda x: x[1], reverse=True)
print("Top-10 重要特征:")
for i, (name, importance) in enumerate(feature_importance[:10], 1):
    print(f"  {i:2d}. {name:15s}: {importance:.4f}")
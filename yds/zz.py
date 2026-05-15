import os
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
# 路径设置
# =====================
TRAIN_X_PATH = "/work2/dhy/cqw/B/yds/datas/x_train_split.csv"
TRAIN_Y_PATH = "/work2/dhy/cqw/B/yds/datas/y_train_split.csv"

TEST_X_PATH = "/work2/dhy/cqw/B/yds/datas/x_val.csv"
TEST_Y_PATH = "/work2/dhy/cqw/B/yds/datas/y_val.csv"

OUTPUT_DIR = "/work2/dhy/cqw/B/yds/results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_mlp_model.pt")
RESULT_FIG_PATH = os.path.join(OUTPUT_DIR, "training_results.png")

# =====================
# 1. 读取数据 + 数据质量检查
# =====================
print("=" * 60)
print("1. 读取数据及数据质量检查")
print("=" * 60)

# ---------- 读取训练集 ----------
try:
    X_train_df = pd.read_csv(TRAIN_X_PATH, encoding="utf-8")
except UnicodeDecodeError:
    try:
        X_train_df = pd.read_csv(TRAIN_X_PATH, encoding="gbk")
        print("训练集使用GBK编码读取")
    except:
        X_train_df = pd.read_csv(TRAIN_X_PATH, encoding="gb18030")
        print("训练集使用GB18030编码读取")

y_train_df = pd.read_csv(TRAIN_Y_PATH, encoding="gbk")
y_train_all = y_train_df.values.squeeze()

# ---------- 读取测试集 ----------
try:
    X_test_df = pd.read_csv(TEST_X_PATH, encoding="utf-8")
except UnicodeDecodeError:
    try:
        X_test_df = pd.read_csv(TEST_X_PATH, encoding="gbk")
        print("测试集使用GBK编码读取")
    except:
        X_test_df = pd.read_csv(TEST_X_PATH, encoding="gb18030")
        print("测试集使用GB18030编码读取")

y_test_df = pd.read_csv(TEST_Y_PATH, encoding="gbk")
y_test = y_test_df.values.squeeze()

print(f"训练集形状: X={X_train_df.shape}, y={y_train_all.shape}")
print(f"测试集形状: X={X_test_df.shape}, y={y_test.shape}")

# 给列命名
column_names = [
    'gammaIn1Re', 'gammaIn1Im', 'closeFreqMHz',
    'itState', 'gammaIn2Re', 'gammaIn2Im'
]

X_train_df.columns = column_names
X_test_df.columns = column_names

# =====================
# 数据清洗函数
# =====================
def clean_dataframe(X):
    print("\n【数据质量检查】")

    # 缺失值检查
    missing_count = X.isnull().sum()

    if missing_count.sum() > 0:
        print(f"⚠ 发现缺失值:\n{missing_count[missing_count > 0]}")

        for col in X.columns:
            if missing_count[col] > 0:
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
                print(f"  - {col}: 用中位数 {median_val:.4f} 填充")
    else:
        print("✓ 无缺失值")

    # 无穷值检查
    inf_count = np.isinf(X.values).sum()

    if inf_count > 0:
        print(f"⚠ 发现无穷值 {inf_count} 个")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)
    else:
        print("✓ 无无穷值")

    # 异常值截尾
    print("\n【异常值检测（IQR方法）】")

    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()

        if outliers > 0:
            print(f"{col}: {outliers} 个异常值")

        X[col] = X[col].clip(lower_bound, upper_bound)

    print("✓ 异常值已截尾处理")

    return X

# 清洗训练/测试数据
print("\n处理训练集...")
X_train_df = clean_dataframe(X_train_df)

print("\n处理测试集...")
X_test_df = clean_dataframe(X_test_df)

# =====================
# 2. 特征工程
# =====================
print("\n" + "=" * 60)
print("2. 特征工程")
print("=" * 60)

def feature_engineering(X, fit_encoder=False, encoder=None):
    X_np = X.values

    g1_re, g1_im = X_np[:, 0], X_np[:, 1]
    freq = X_np[:, 2]
    it_state = X_np[:, 3].astype(int)
    g2_re, g2_im = X_np[:, 4], X_np[:, 5]

    epsilon = 1e-8

    # 幅度/相位
    g1_mag = np.sqrt(g1_re ** 2 + g1_im ** 2)
    g2_mag = np.sqrt(g2_re ** 2 + g2_im ** 2)

    g1_phase = np.arctan2(g1_im, g1_re)
    g2_phase = np.arctan2(g2_im, g2_re)

    # VSWR
    g1_vswr = (1 + g1_mag) / (1 - np.clip(g1_mag, 0, 1 - epsilon) + epsilon)
    g2_vswr = (1 + g2_mag) / (1 - np.clip(g2_mag, 0, 1 - epsilon) + epsilon)

    # RL
    g1_rl = -20 * np.log10(np.clip(g1_mag, epsilon, 1))
    g2_rl = -20 * np.log10(np.clip(g2_mag, epsilon, 1))

    # 差值特征
    delta_mag = g1_mag - g2_mag
    delta_phase = g1_phase - g2_phase
    delta_vswr = g1_vswr - g2_vswr
    delta_rl = g1_rl - g2_rl

    # 比值特征
    ratio_mag = g1_mag / (g2_mag + epsilon)
    ratio_vswr = g1_vswr / (g2_vswr + epsilon)

    # 交叉特征
    mag_product = g1_mag * g2_mag
    rl_sum = g1_rl + g2_rl

    # 频率归一化
    freq_normalized = (freq - np.mean(freq)) / (np.std(freq) + epsilon)

    # 连续特征
    X_continuous = np.stack([
        g1_re, g1_im,
        g2_re, g2_im,
        freq_normalized,

        g1_mag, g2_mag,
        g1_phase, g2_phase,

        g1_vswr, g2_vswr,
        g1_rl, g2_rl,

        delta_mag,
        delta_phase,
        delta_vswr,
        delta_rl,

        ratio_mag,
        ratio_vswr,

        mag_product,
        rl_sum,
    ], axis=1)

    # One-hot
    it_state_reshaped = it_state.reshape(-1, 1)

    if fit_encoder:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore'
        )
        it_onehot = encoder.fit_transform(it_state_reshaped)
    else:
        it_onehot = encoder.transform(it_state_reshaped)

    # 拼接
    X_feat = np.hstack([X_continuous, it_onehot])

    return X_feat, encoder, it_onehot.shape[1]

# 训练集fit encoder
X_train_feat, encoder, onehot_dim = feature_engineering(
    X_train_df,
    fit_encoder=True
)

# 测试集复用encoder
X_test_feat, _, _ = feature_engineering(
    X_test_df,
    fit_encoder=False,
    encoder=encoder
)

print(f"最终特征维度: {X_train_feat.shape[1]}")

# 类别数
unique_classes = np.unique(y_train_all)
num_classes = len(unique_classes)

print(f"类别数: {num_classes}")

# =====================
# 3. 划分训练/验证集
# =====================
print("\n" + "=" * 60)
print("3. 划分训练/验证集")
print("=" * 60)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_feat,
    y_train_all,
    test_size=0.2,
    random_state=42,
    stratify=y_train_all
)

X_test = X_test_feat

print(f"训练集: {X_train.shape}")
print(f"验证集: {X_val.shape}")
print(f"测试集: {X_test.shape}")

# =====================
# 4. 标准化
# =====================
print("\n" + "=" * 60)
print("4. 标准化")
print("=" * 60)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 转Tensor
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train_t, y_train_t)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True
)

# =====================
# 5. 模型
# =====================
print("\n" + "=" * 60)
print("5. 构建模型")
print("=" * 60)

class MLP(nn.Module):

    def __init__(
        self,
        in_dim,
        num_classes,
        hidden_dims=[128, 64, 32],
        dropout=0.3
    ):
        super().__init__()

        layers = []
        prev_dim = in_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

input_dim = X_train_t.shape[1]

model = MLP(
    input_dim,
    num_classes,
    hidden_dims=[128, 64, 32],
    dropout=0.3
)

# 类别权重
class_counts = np.bincount(y_train)

if class_counts.max() / (class_counts.min() + 1e-8) > 2:
    print("⚠ 使用加权交叉熵")

    class_weights = 1.0 / torch.tensor(
        class_counts,
        dtype=torch.float32
    )

    class_weights = (
        class_weights / class_weights.sum() * num_classes
    )

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

else:
    loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=8
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
best_test_acc = 0.0

patience = 15
patience_counter = 0

num_epochs = 100

for epoch in range(num_epochs):

    # ========= 训练 =========
    model.train()

    epoch_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = loss_fn(logits, batch_y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)

    # ========= 验证 =========
    model.eval()

    with torch.no_grad():

        # 验证集
        logits_val = model(X_val_t)

        preds_val = logits_val.argmax(dim=1)

        val_acc = (
            (preds_val == y_val_t)
            .float()
            .mean()
            .item()
        )

        val_accs.append(val_acc)

        # 测试集
        logits_test = model(X_test_t)

        preds_test = logits_test.argmax(dim=1)

        test_acc = (
            (preds_test == y_test_t)
            .float()
            .mean()
            .item()
        )

        test_accs.append(test_acc)

    scheduler.step(val_acc)

    # 保存最佳模型
    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_test_acc = test_acc

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            f"✓ 保存最佳模型 "
            f"(val_acc={val_acc:.4f}, "
            f"test_acc={test_acc:.4f})"
        )

        patience_counter = 0

    else:
        patience_counter += 1

    if epoch % 5 == 0 or epoch == num_epochs - 1:

        print(
            f"Epoch {epoch + 1:3d}/{num_epochs}: "
            f"loss={avg_loss:.4f}, "
            f"val_acc={val_acc:.4f}, "
            f"test_acc={test_acc:.4f}"
        )

    if patience_counter >= patience:

        print(f"\n早停触发！第 {epoch + 1} 轮停止")

        break

print("\n训练完成！")
print(f"最佳验证准确率: {best_val_acc:.4f}")
print(f"对应测试准确率: {best_test_acc:.4f}")

# =====================
# 7. 绘图
# =====================
print("\n" + "=" * 60)
print("7. 绘制结果图")
print("=" * 60)

model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

with torch.no_grad():

    final_preds_test = (
        model(X_test_t)
        .argmax(dim=1)
        .numpy()
    )

    final_preds_val = (
        model(X_val_t)
        .argmax(dim=1)
        .numpy()
    )

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 12)
)

# Loss
ax1 = axes[0, 0]

ax1.plot(train_losses)

ax1.set_title("训练损失")

# Accuracy
ax2 = axes[0, 1]

ax2.plot(val_accs, label='val')
ax2.plot(test_accs, label='test')

ax2.legend()

ax2.set_title("验证/测试准确率")

# 验证集混淆矩阵
ax3 = axes[1, 0]

cm_val = confusion_matrix(
    y_val,
    final_preds_val
)

sns.heatmap(
    cm_val,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax3
)

ax3.set_title("验证集混淆矩阵")

# 测试集混淆矩阵
ax4 = axes[1, 1]

cm_test = confusion_matrix(
    y_test,
    final_preds_test
)

sns.heatmap(
    cm_test,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax4
)

ax4.set_title("测试集混淆矩阵")

plt.tight_layout()

plt.savefig(
    RESULT_FIG_PATH,
    dpi=200,
    bbox_inches='tight'
)

print(f"图已保存: {RESULT_FIG_PATH}")

plt.show()

# =====================
# 8. 分类报告
# =====================
print("\n" + "=" * 60)
print("8. 验证集分类报告")
print("=" * 60)

print(
    classification_report(
        y_val,
        final_preds_val
    )
)

print("\n" + "=" * 60)
print("9. 测试集分类报告")
print("=" * 60)

print(
    classification_report(
        y_test,
        final_preds_test
    )
)

# =====================
# 10. 特征重要性
# =====================
print("\n" + "=" * 60)
print("10. 特征重要性")
print("=" * 60)

first_layer_weights = (
    model.net[0]
    .weight
    .data
    .abs()
    .mean(dim=0)
    .numpy()
)

feature_names = (
    [
        'g1_re',
        'g1_im',
        'g2_re',
        'g2_im',
        'freq_norm',

        'g1_mag',
        'g2_mag',

        'g1_phase',
        'g2_phase',

        'g1_vswr',
        'g2_vswr',

        'g1_rl',
        'g2_rl',

        'delta_mag',
        'delta_phase',
        'delta_vswr',
        'delta_rl',

        'ratio_mag',
        'ratio_vswr',

        'mag_product',
        'rl_sum'
    ]
    +
    [f'itState_{i}' for i in range(onehot_dim)]
)

feature_importance = list(
    zip(feature_names, first_layer_weights)
)

feature_importance.sort(
    key=lambda x: x[1],
    reverse=True
)

print("Top-10 重要特征:")

for i, (name, importance) in enumerate(
    feature_importance[:10],
    1
):

    print(
        f"{i:2d}. "
        f"{name:15s}: "
        f"{importance:.4f}"
    )
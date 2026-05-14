import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# 解决中文显示问题
# =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# =====================
# 1. 加载数据
# =====================
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# =====================
# 2. 标准化
# =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# =====================
# 3. 构建模型
# =====================
class MLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dims=[128,64,32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim,h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim,num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

num_classes = len(np.unique(y_train))
input_dim = X_train_t.shape[1]
model = MLP(input_dim,num_classes)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# =====================
# 4. 训练循环 + 记录指标
# =====================
num_epochs = 50
train_losses = []
val_accs = []
test_accs = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits,batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    # 验证集
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_preds = val_logits.argmax(dim=1)
        val_acc = (val_preds == y_val_t).float().mean().item()
        val_accs.append(val_acc)

        test_logits = model(X_test_t)
        test_preds = test_logits.argmax(dim=1)
        test_acc = (test_preds == y_test_t).float().mean().item()
        test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{num_epochs}: loss={train_losses[-1]:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

# =====================
# 5. 保存模型
# =====================
torch.save(model.state_dict(),"mlp_model.pt")
print("模型训练完成并保存为 mlp_model.pt")

# =====================
# 6. 绘制训练损失 & 准确率曲线
# =====================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label="训练损失", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("训练损失曲线")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accs, label="验证准确率", marker='s')
plt.plot(test_accs, label="测试准确率", marker='^')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("验证/测试准确率曲线")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=200)
plt.show()

# =====================
# 7. 混淆矩阵
# =====================
cm_val = confusion_matrix(y_val, val_preds.numpy())
cm_test = confusion_matrix(y_test, test_preds.numpy())

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues')
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("验证集混淆矩阵")

plt.subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens')
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("测试集混淆矩阵")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=200)
plt.show()

# =====================
# 8. 分类报告
# =====================
print("验证集分类报告:")
print(classification_report(y_val, val_preds.numpy()))
print("\n测试集分类报告:")
print(classification_report(y_test, test_preds.numpy()))
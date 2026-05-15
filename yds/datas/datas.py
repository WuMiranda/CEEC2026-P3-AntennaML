import pandas as pd
from sklearn.model_selection import train_test_split
import os

# =========================
# 输入文件路径
# =========================
x_path = "/work2/dhy/cqw/B/yds/datasets/x_train.csv"
y_path = "/work2/dhy/cqw/B/yds/datasets/y_train.csv"

# =========================
# 输出文件夹
# =========================
output_dir = "/work2/dhy/cqw/B/yds/datas"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 读取数据
# =========================
x_df = pd.read_csv(x_path, encoding="gbk")
y_df = pd.read_csv(y_path, encoding="gbk")

# =========================
# 检查行数是否一致
# =========================
assert len(x_df) == len(y_df), "x_train.csv 与 y_train.csv 行数不一致！"

# =========================
# 数据总数
# =========================
total_samples = len(x_df)
print(f"总样本数: {total_samples}")

# 目标划分数量
train_num = 58061
val_num = 19354
test_num = 19353

# 检查总数
assert train_num + val_num + test_num == total_samples, "划分数量之和不等于总样本数！"

# =========================
# 先划分训练集
# =========================
x_train, x_temp, y_train, y_temp = train_test_split(
    x_df,
    y_df,
    train_size=train_num,
    random_state=42,
    shuffle=True
)

# =========================
# 再划分验证集和测试集
# =========================
x_val, x_test, y_val, y_test = train_test_split(
    x_temp,
    y_temp,
    train_size=val_num,
    test_size=test_num,
    random_state=42,
    shuffle=True
)

# =========================
# 保存结果
# =========================
x_train.to_csv(f"{output_dir}/x_train_split.csv", index=False)
y_train.to_csv(f"{output_dir}/y_train_split.csv", index=False)

x_val.to_csv(f"{output_dir}/x_val.csv", index=False)
y_val.to_csv(f"{output_dir}/y_val.csv", index=False)

x_test.to_csv(f"{output_dir}/x_test.csv", index=False)
y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

# =========================
# 输出信息
# =========================
print("\n划分完成：")
print(f"训练集: {len(x_train)}")
print(f"验证集: {len(x_val)}")
print(f"测试集: {len(x_test)}")

print("\n文件已保存到：")
print(output_dir)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#将特征一和特征四画在复数平面上，观察不同Label的聚类分布。
# =========================
# 路径
# =========================
TRAIN_X_PATH = '/work2/dhy/cqw/B/yds/datasets/x_train.csv'
TRAIN_Y_PATH = '/work2/dhy/cqw/B/yds/datasets/y_train.csv'
OUT_DIR = '/work2/dhy/cqw/B/yds/output/complex_plane/'

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 读取数据
# =========================
x = pd.read_csv(TRAIN_X_PATH, encoding='gbk')
y = pd.read_csv(TRAIN_Y_PATH, encoding='gbk')

data = pd.concat([x, y], axis=1)

data.columns = [
    'gamma1_re', 'gamma1_im',
    'close_freq',
    'itState',
    'gamma2_re', 'gamma2_im',
    'label'
]

# =========================
# 按 label 分图
# =========================
labels = sorted(data['label'].unique())

for lb in labels:
    subset = data[data['label'] == lb]

    plt.figure(figsize=(7,7))

    # ===== gamma1 =====
    plt.scatter(
        subset['gamma1_re'],
        subset['gamma1_im'],
        s=10,
        alpha=0.6,
        label='gamma1 (closed)'
    )

    # ===== gamma2 =====
    plt.scatter(
        subset['gamma2_re'],
        subset['gamma2_im'],
        s=10,
        alpha=0.6,
        label='gamma2 (default)'
    )

    # =========================
    # 坐标轴设置（复平面）
    # =========================
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)

    plt.xlabel("Real Part")
    plt.ylabel("Imag Part")
    plt.title(f"Complex Plane Distribution (Label {lb})")
    plt.legend()

    plt.axis('equal')  # 保证圆形不变形

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'label_{lb}_complex_plane.png'))
    plt.close()

print("Done. Saved to:", OUT_DIR)
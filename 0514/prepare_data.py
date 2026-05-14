import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# =====================
# 1. 读取原始数据
# =====================

# 尝试自动检测多种编码
for enc in ["utf-8", "gbk", "gb18030"]:
    try:
        X = pd.read_csv("x_train.csv", encoding=enc)
        print(f"成功读取 x_train.csv，编码={enc}")
        break
    except Exception as e:
        print(f"尝试编码 {enc} 失败: {e}")
else:
    raise ValueError("无法读取 x_train.csv，请确认文件编码格式")

# y_train 同理
for enc in ["utf-8", "gbk", "gb18030"]:
    try:
        y = pd.read_csv("y_train.csv", encoding=enc).values.squeeze()
        print(f"成功读取 y_train.csv，编码={enc}")
        break
    except Exception as e:
        print(f"尝试编码 {enc} 失败: {e}")
else:
    raise ValueError("无法读取 y_train.csv，请确认文件编码格式")
# 列名
X.columns = ['gammaIn1Re', 'gammaIn1Im', 'closeFreqMHz',
             'itState', 'gammaIn2Re', 'gammaIn2Im']

# =====================
# 2. 特征工程（增强版）
# =====================
X_np = X.values
g1_re, g1_im = X_np[:, 0], X_np[:, 1]
freq = X_np[:, 2]
it_state = X_np[:, 3].astype(int)
g2_re, g2_im = X_np[:, 4], X_np[:, 5]

# 基础特征
g1_mag = np.sqrt(g1_re ** 2 + g1_im ** 2)
g2_mag = np.sqrt(g2_re ** 2 + g2_im ** 2)
g1_phase = np.arctan2(g1_im, g1_re)
g2_phase = np.arctan2(g2_im, g2_re)

epsilon = 1e-8
g1_vswr = (1 + g1_mag) / (1 - np.clip(g1_mag, 0, 1 - epsilon) + epsilon)
g2_vswr = (1 + g2_mag) / (1 - np.clip(g2_mag, 0, 1 - epsilon) + epsilon)
g1_rl = -20 * np.log10(np.clip(g1_mag, epsilon, 1))
g2_rl = -20 * np.log10(np.clip(g2_mag, epsilon, 1))

delta_mag = g1_mag - g2_mag
delta_phase = g1_phase - g2_phase
delta_vswr = g1_vswr - g2_vswr
delta_rl = g1_rl - g2_rl

ratio_mag = g1_mag / (g2_mag + epsilon)
ratio_vswr = g1_vswr / (g2_vswr + epsilon)
mag_product = g1_mag * g2_mag
rl_sum = g1_rl + g2_rl
freq_norm = (freq - np.mean(freq)) / (np.std(freq) + epsilon)

# 合并连续特征
X_cont = np.stack([
    g1_re, g1_im, g2_re, g2_im, freq_norm,
    g1_mag, g2_mag, g1_phase, g2_phase,
    g1_vswr, g2_vswr, g1_rl, g2_rl,
    delta_mag, delta_phase, delta_vswr, delta_rl,
    ratio_mag, ratio_vswr, mag_product, rl_sum
], axis=1)

# 分类特征 One-Hot
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
it_onehot = encoder.fit_transform(it_state.reshape(-1, 1))

# 最终特征
X_feat = np.hstack([X_cont, it_onehot])

# =====================
# 3. 划分数据集
# =====================
X_temp, X_test, y_temp, y_test = train_test_split(
    X_feat, y, test_size=0.1, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2222, random_state=42, stratify=y_temp
)

# =====================
# 4. 保存为独立文件
# =====================
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("数据集已保存为 Numpy 文件：X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy")
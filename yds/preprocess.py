import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COLUMN_NAMES = [
    'gammaIn1Re', 'gammaIn1Im', 'closeFreqMHz',
    'itState', 'gammaIn2Re', 'gammaIn2Im'
]

BASE_FEATURE_NAMES = [
    'g1_re',
    'g1_im',
    'g2_re',
    'g2_im',
    'freq',
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
    'rl_sum',
]


class Preprocessor:

    def __init__(self):
        self.clean_stats = None
        self.encoder = None
        self.scaler = None
        self.onehot_dim = None
        self.feature_names = None
        self.freq_mean = None
        self.freq_std = None

    def fit(self, X):
        X = ensure_columns(X)
        self.clean_stats = fit_clean_stats(X)
        X_clean = transform_clean_dataframe(X, self.clean_stats, verbose=True)

        self.freq_mean = float(X_clean['closeFreqMHz'].mean())
        self.freq_std = float(X_clean['closeFreqMHz'].std() + 1e-8)

        X_feat, self.encoder, self.onehot_dim = feature_engineering(
            X_clean,
            fit_encoder=True,
            encoder=None
        )
        self.scaler = StandardScaler()
        self.scaler.fit(X_feat)

        self.feature_names = get_feature_names(self.encoder, self.onehot_dim)
        return self

    def transform(self, X, verbose=False):
        if self.clean_stats is None or self.encoder is None or self.scaler is None:
            raise RuntimeError('Preprocessor must be fitted before transform.')

        X = ensure_columns(X)
        X_clean = transform_clean_dataframe(X, self.clean_stats, verbose=verbose)
        X_feat, _, _ = feature_engineering(
            X_clean,
            fit_encoder=False,
            encoder=self.encoder
        )
        return self.scaler.transform(X_feat)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def read_csv_with_encoding(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            print(f'{path} 使用GBK编码读取')
            return pd.read_csv(path, encoding='gbk')
        except UnicodeDecodeError:
            print(f'{path} 使用GB18030编码读取')
            return pd.read_csv(path, encoding='gb18030')


def load_xy(x_path, y_path):
    X = read_csv_with_encoding(x_path)
    y_df = pd.read_csv(y_path, encoding='gbk')
    y = y_df.values.squeeze()
    X = ensure_columns(X)
    return X, y


def ensure_columns(X):
    X = X.copy()
    X.columns = COLUMN_NAMES
    return X


def fit_clean_stats(X):
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    medians = X.median()
    filled = X.fillna(medians)

    lower_bounds = {}
    upper_bounds = {}

    for col in filled.columns:
        Q1 = filled[col].quantile(0.25)
        Q3 = filled[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bounds[col] = Q1 - 1.5 * IQR
        upper_bounds[col] = Q3 + 1.5 * IQR

    return {
        'medians': medians.to_dict(),
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
    }


def transform_clean_dataframe(X, clean_stats, verbose=False):
    X = X.copy()

    if verbose:
        print('\n【数据质量检查】')

    missing_count = X.isnull().sum()

    if verbose:
        if missing_count.sum() > 0:
            print(f'⚠ 发现缺失值:\n{missing_count[missing_count > 0]}')
        else:
            print('✓ 无缺失值')

    inf_count = np.isinf(X.values).sum()

    if verbose:
        if inf_count > 0:
            print(f'⚠ 发现无穷值 {inf_count} 个')
        else:
            print('✓ 无无穷值')

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(clean_stats['medians'])

    if verbose:
        print('\n【异常值检测（IQR方法）】')

    for col in X.columns:
        lower_bound = clean_stats['lower_bounds'][col]
        upper_bound = clean_stats['upper_bounds'][col]
        outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()

        if verbose and outliers > 0:
            print(f'{col}: {outliers} 个异常值')

        X[col] = X[col].clip(lower_bound, upper_bound)

    if verbose:
        print('✓ 异常值已按训练集统计量截尾处理')

    return X


def make_one_hot_encoder():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown='ignore')


def feature_engineering(X, fit_encoder=False, encoder=None):
    X_np = X.values

    g1_re, g1_im = X_np[:, 0], X_np[:, 1]
    freq = X_np[:, 2]
    it_state = X_np[:, 3].astype(int)
    g2_re, g2_im = X_np[:, 4], X_np[:, 5]

    epsilon = 1e-8

    g1_mag = np.sqrt(g1_re ** 2 + g1_im ** 2)
    g2_mag = np.sqrt(g2_re ** 2 + g2_im ** 2)

    g1_phase = np.arctan2(g1_im, g1_re)
    g2_phase = np.arctan2(g2_im, g2_re)

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

    X_continuous = np.stack([
        g1_re, g1_im,
        g2_re, g2_im,
        freq,
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

    it_state_reshaped = it_state.reshape(-1, 1)

    if fit_encoder:
        encoder = make_one_hot_encoder()
        it_onehot = encoder.fit_transform(it_state_reshaped)
    else:
        if encoder is None:
            raise ValueError('encoder is required when fit_encoder=False')
        it_onehot = encoder.transform(it_state_reshaped)

    X_feat = np.hstack([X_continuous, it_onehot])

    return X_feat, encoder, it_onehot.shape[1]


def get_feature_names(encoder, onehot_dim):
    if encoder is not None and hasattr(encoder, 'categories_'):
        onehot_names = [f'itState_{int(v)}' for v in encoder.categories_[0]]
    else:
        onehot_names = [f'itState_{i}' for i in range(onehot_dim)]

    return BASE_FEATURE_NAMES + onehot_names

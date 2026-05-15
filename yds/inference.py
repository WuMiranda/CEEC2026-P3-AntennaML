import os

import numpy as np
import pandas as pd
import torch

from model import MLP
from preprocess import COLUMN_NAMES, Preprocessor


DEFAULT_MODEL_PATH = '/work2/dhy/cqw/B/yds/results/best_mlp_model.pt'
DEFAULT_PREPROCESSOR_PATH = '/work2/dhy/cqw/B/yds/results/preprocessor.pkl'


class Predictor:

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        preprocessor_path=DEFAULT_PREPROCESSOR_PATH
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'模型文件不存在: {model_path}')
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f'预处理器文件不存在: {preprocessor_path}')

        self.preprocessor = Preprocessor.load(preprocessor_path)
        self.checkpoint = torch.load(model_path, map_location='cpu')

        self.model = MLP(
            self.checkpoint['input_dim'],
            self.checkpoint['num_classes'],
            hidden_dims=self.checkpoint['hidden_dims'],
            dropout=self.checkpoint['dropout']
        )
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, sample):
        X = sample_to_dataframe(sample)
        X_scaled = self.preprocessor.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()

        return pred

    def predict_with_proba(self, sample):
        X = sample_to_dataframe(sample)
        X_scaled = self.preprocessor.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred = int(np.argmax(probs))

        return pred, probs


def sample_to_dataframe(sample):
    if isinstance(sample, pd.DataFrame):
        return sample[COLUMN_NAMES].copy()

    if isinstance(sample, dict):
        return pd.DataFrame([[sample[col] for col in COLUMN_NAMES]], columns=COLUMN_NAMES)

    arr = np.asarray(sample, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != len(COLUMN_NAMES):
        raise ValueError(f'输入特征维度应为 {len(COLUMN_NAMES)}，实际为 {arr.shape[1]}')

    return pd.DataFrame(arr, columns=COLUMN_NAMES)


def predict_one(sample, model_path=DEFAULT_MODEL_PATH, preprocessor_path=DEFAULT_PREPROCESSOR_PATH):
    predictor = Predictor(model_path, preprocessor_path)
    return predictor.predict(sample)


if __name__ == '__main__':
    example = {
        'gammaIn1Re': 0.0,
        'gammaIn1Im': 0.0,
        'closeFreqMHz': 0.0,
        'itState': 0,
        'gammaIn2Re': 0.0,
        'gammaIn2Im': 0.0,
    }

    predictor = Predictor()
    pred, probs = predictor.predict_with_proba(example)
    print(f'预测类别: {pred}')
    print(f'类别概率: {probs}')

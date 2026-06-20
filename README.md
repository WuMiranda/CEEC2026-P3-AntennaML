# 项目说明

用于手机天线工作状态识别的轻量级训练与评估框架。

## 主要功能
- 训练 `N--` / `N-+` 版本模型
- 支持固定切分复用
- 支持特征消融、VSWR/RL、对数幅值比等特征
- 支持模型参数量 / MACs / FLOPs 统计
- 支持 FP32 / INT8 dynamic 推理测速

## 主入口
训练：
```bash
python -m balanced_framework.train_balanced --help
```

模型统计：
```bash
python -m balanced_framework.profile_model --help
```

推理测速：
```bash
python -m balanced_framework.benchmark_inference --help
```

## 当前模型说明
当前主线模型为轻量级 MLP，并结合长尾均衡训练策略与类别先验 logit 校正。
其中：
- `N--`：删除前期单变量消融中不利于性能的特征
- `N-+`：在 `N--` 基础上加入对数幅值比特征

## 典型输出
- `config.json`
- `metrics.json`
- `model.pt`
- `preprocess.npz`
- `profile_reports/`
- `benchmark_reports/`

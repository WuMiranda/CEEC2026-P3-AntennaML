## baseline——同步修改版

### 需要学会

RandomForestClassifier
XGBoost
LightGBM
MLPClassifier / PyTorch MLP

### 尝试的baseline

RandomForestClassifier
XGBoost
LightGBM
MLPClassifier / PyTorch MLP

### 一步步来

1.数据类型是什么？表格、传感器/时序、图像、文本、多模态数据：表格数据。

2.表格数据就优先尝试树模型，再到mlp。其他：线性回归、逻辑回归、随机森林、梯度提升树XGBoost / LightGBM / CatBoost、mlp、模型融合。

问师姐要表格学习的资料

3.搭建baseline：把数据读进来 （数据格式能不能读进去）→ 整理成模型能吃的格式（模型能不能训练） → 训练一个最基础模型 → 验证效果 （评价指标能不能算）→ 生成预测结果/提交结果（画图、保存模型、交叉验证流程能不能跑）。

4.生成模拟数据。人为设置隐藏规律

模拟训练集：用来训练模型
模拟验证集：用来算准确率、F1、混淆矩阵（打印）

5.搭建好了随机森林的baseline，模拟数据的生成是个问题，用什么作为指标

project/
│
├── main.py              # 主入口：控制整个流程
├── config.py            # 参数配置
├── data.py              # 数据读取 / 模拟数据生成
├── preprocess.py        # 预处理
├── models.py            # 不同模型
├── train.py             # 训练逻辑
├── evaluate.py          # 指标计算
└── utils.py             # 工具函数，可暂时没有
参数要求高
6.完整代码文件，训练训练集1，将训练好的模型导出pth/onnx，写inference.py，用分割的另一份数据看看效果。
### 提交的文件
一个轻量模型文件，比如 model.onnx 或 model.pth
一个 inference.py
可能还要有：标准化参数文件、类别映射说明、requirements.txt
### 数据情况
样本数：96768
原始输入维度：6
标签类别：15 类
没有缺失值
itState 只有 16 个取值，明显是离散状态
closeFreqMHz 有 72 个不同频点
标签分布不均衡，最少类 775，最多类 15974
处理目标：能体现物理合理性、能让轻量模型稳定训练、不能引入太多冗余维度，避免和“小模型、低计算量”的评分方向冲突
### 打算做的（存档）（要用class吗？）
先一个train.py和一个model.py

import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import MLP
from preprocess import Preprocessor, load_xy


warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

TRAIN_X_PATH = '/work2/dhy/cqw/B/yds/datas/x_train_split.csv'
TRAIN_Y_PATH = '/work2/dhy/cqw/B/yds/datas/y_train_split.csv'

OUTPUT_DIR = '/work2/dhy/cqw/B/yds/results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_mlp_model.pt')
PREPROCESSOR_PATH = os.path.join(OUTPUT_DIR, 'preprocessor.pkl')
RESULT_FIG_PATH = os.path.join(OUTPUT_DIR, 'training_results.png')

RANDOM_STATE = 42
BATCH_SIZE = 64
NUM_EPOCHS = 100
PATIENCE = 15
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits, y_true):
    preds = logits.argmax(dim=1)
    acc = (preds == y_true).float().mean().item()
    return acc, preds


def main():
    set_seed(RANDOM_STATE)

    print('=' * 60)
    print('1. 读取数据')
    print('=' * 60)

    X_all_df, y_all = load_xy(TRAIN_X_PATH, TRAIN_Y_PATH)

    print(f'完整训练数据形状: X={X_all_df.shape}, y={y_all.shape}')

    print('\n' + '=' * 60)
    print('2. 划分训练/验证集')
    print('=' * 60)

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_all_df,
        y_all,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_all
    )

    print(f'训练集: X={X_train_df.shape}, y={y_train.shape}')
    print(f'验证集: X={X_val_df.shape}, y={y_val.shape}')

    print('\n' + '=' * 60)
    print('3. 预处理：训练集统计参数，验证集复用参数')
    print('=' * 60)

    preprocessor = Preprocessor()

    print('\n处理训练集...')
    preprocessor.fit(X_train_df)
    X_train_scaled = preprocessor.transform(X_train_df)

    print('\n处理验证集...')
    X_val_scaled = preprocessor.transform(X_val_df, verbose=True)

    preprocessor.save(PREPROCESSOR_PATH)
    print(f'预处理器已保存: {PREPROCESSOR_PATH}')
    print(f'最终特征维度: {X_train_scaled.shape[1]}')
    print(f'训练集频率均值: {preprocessor.freq_mean:.6f}')
    print(f'训练集频率标准差: {preprocessor.freq_std:.6f}')

    unique_classes = np.unique(y_all)
    num_classes = len(unique_classes)

    print(f'类别数: {num_classes}')

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print('\n' + '=' * 60)
    print('4. 构建模型')
    print('=' * 60)

    input_dim = X_train_t.shape[1]

    model = MLP(
        input_dim,
        num_classes,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT
    )

    class_counts = np.bincount(y_train)

    if class_counts.max() / (class_counts.min() + 1e-8) > 2:
        print('⚠ 使用加权交叉熵')

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
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=8
    )

    print('\n' + '=' * 60)
    print('5. 开始训练')
    print('=' * 60)

    train_losses = []
    val_accs = []

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
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

        model.eval()

        with torch.no_grad():
            logits_val = model(X_val_t)
            val_acc, preds_val = accuracy_from_logits(logits_val, y_val_t)
            val_accs.append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'input_dim': input_dim,
                    'num_classes': num_classes,
                    'hidden_dims': HIDDEN_DIMS,
                    'dropout': DROPOUT,
                    'feature_names': preprocessor.feature_names,
                },
                BEST_MODEL_PATH
            )

            print(f'✓ 保存最佳模型 (val_acc={val_acc:.4f})')

            patience_counter = 0

        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            print(
                f'Epoch {epoch + 1:3d}/{NUM_EPOCHS}: '
                f'loss={avg_loss:.4f}, '
                f'val_acc={val_acc:.4f}'
            )

        if patience_counter >= PATIENCE:
            print(f'\n早停触发！第 {epoch + 1} 轮停止')
            break

    print('\n训练完成！')
    print(f'最佳验证准确率: {best_val_acc:.4f}')

    print('\n' + '=' * 60)
    print('6. 绘制验证结果图')
    print('=' * 60)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        final_preds_val = (
            model(X_val_t)
            .argmax(dim=1)
            .numpy()
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    ax1 = axes[0]
    ax1.plot(train_losses)
    ax1.set_title('训练损失')

    ax2 = axes[1]
    ax2.plot(val_accs, label='val')
    ax2.legend()
    ax2.set_title('验证准确率')

    plt.tight_layout()

    plt.savefig(
        RESULT_FIG_PATH,
        dpi=200,
        bbox_inches='tight'
    )

    print(f'图已保存: {RESULT_FIG_PATH}')

    print('\n' + '=' * 60)
    print('7. 验证集分类报告')
    print('=' * 60)

    print(
        classification_report(
            y_val,
            final_preds_val
        )
    )

    print('\n' + '=' * 60)
    print('8. 验证集混淆矩阵')
    print('=' * 60)

    cm_val = confusion_matrix(
        y_val,
        final_preds_val
    )

    print(cm_val)

    cm_fig_path = os.path.join(OUTPUT_DIR, 'val_confusion_matrix.png')
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_val,
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    plt.title('验证集混淆矩阵')
    plt.tight_layout()
    plt.savefig(cm_fig_path, dpi=200, bbox_inches='tight')
    print(f'验证集混淆矩阵图已保存: {cm_fig_path}')

    print('\n' + '=' * 60)
    print('9. 特征重要性')
    print('=' * 60)

    first_layer_weights = (
        model.net[0]
        .weight
        .data
        .abs()
        .mean(dim=0)
        .numpy()
    )

    feature_importance = list(
        zip(preprocessor.feature_names, first_layer_weights)
    )

    feature_importance.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print('Top-10 重要特征:')

    for i, (name, importance) in enumerate(
        feature_importance[:10],
        1
    ):
        print(
            f'{i:2d}. '
            f'{name:15s}: '
            f'{importance:.4f}'
        )


if __name__ == '__main__':
    main()

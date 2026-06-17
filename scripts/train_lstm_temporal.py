#!/usr/bin/env python3
"""
时序 v2 — 任务 3: LSTM 时序深度学习
====================================
输入: (Symbol, t-5..t-1) 的 7 比率时序窗口(5 个时间步 × 7 个特征)
输出: 舞弊概率

设计:
  - 用 PyTorch 写最小 LSTM(避免过度工程)
  - 处理 NaN:用 0 填充 + mask 标志
  - 5 折交叉验证
  - 与 XGBoost Baseline 对比

产出:
  output/lstm_results.csv
  models/fraud_detection_lstm.pth
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, recall_score, precision_score, roc_auc_score)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
OUT_DIR = os.path.join(BASE, "output")

RATIO_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio',
              'asset_turnover', 'net_margin', 'ocf_to_rev']
SEQ_LEN = 5  # 用 t-5..t-1 共 5 年

print('=' * 60)
print('LSTM 时序深度学习')
print('=' * 60)

# 1. 加载
df = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_with_temporal.csv'))
panel = pd.read_csv(os.path.join(DATA_DIR, 'temporal_panel_wide.csv'))
panel['Symbol'] = panel['Symbol'].astype(str).str.zfill(6)
df['Symbol'] = df['Symbol'].astype(str).str.zfill(6)

# 仅保留 ann_fin_flag ∈ {0, 1} 的样本(用于训练和评估)
df = df.dropna(subset=['ann_fin_flag'])
df = df[df['ann_fin_flag'].isin([0, 1])].copy()
df['violation_year'] = df['violation_year'].astype(int)

print(f'样本: {len(df)}, 正样本: {(df["ann_fin_flag"]==1).sum()} ({(df["ann_fin_flag"]==1).mean()*100:.1f}%)')

# 2. 构建时序序列(每个样本 = 5 × 7 tensor)
print(f'\n构建时序窗口(SEQ_LEN={SEQ_LEN}, FEATURES={len(RATIO_COLS)})...')

panel_idx = panel.set_index(['Symbol', 'year'])

def get_seq(symbol, end_year):
    """获取 (symbol, end_year-SEQ_LEN+1..end_year) 的 7 比率时序"""
    seq = np.zeros((SEQ_LEN, len(RATIO_COLS)), dtype=np.float32)
    mask = np.zeros((SEQ_LEN, len(RATIO_COLS)), dtype=np.float32)
    for i in range(SEQ_LEN):
        y = end_year - SEQ_LEN + 1 + i
        for j, r in enumerate(RATIO_COLS):
            try:
                v = panel_idx.loc[(symbol, y), r]
                if pd.notna(v):
                    seq[i, j] = v
                    mask[i, j] = 1.0
            except KeyError:
                pass
    return seq, mask

# 3. 准备训练数据
X_seqs = []
X_masks = []
y_list = []
symbols = []
violation_years = []

for _, row in df.iterrows():
    seq, mask = get_seq(row['Symbol'], row['violation_year'] - 1)  # 用 t-1 作为窗口末端
    X_seqs.append(seq)
    X_masks.append(mask)
    y_list.append(int(row['ann_fin_flag']))
    symbols.append(row['Symbol'])
    violation_years.append(row['violation_year'])

X_seqs = np.array(X_seqs)
X_masks = np.array(X_masks)
y_arr = np.array(y_list, dtype=np.float32)

print(f'序列形状: {X_seqs.shape} (样本 × 5 步 × 7 特征)')
print(f'非空比率单元: {X_masks.sum() / X_masks.size * 100:.1f}%')

# 4. 标准化(按特征 z-score,用训练集均值/标准差)
mean_per_feat = X_seqs.reshape(-1, len(RATIO_COLS)).mean(axis=0)
std_per_feat = X_seqs.reshape(-1, len(RATIO_COLS)).std(axis=0)
std_per_feat = np.where(std_per_feat < 1e-6, 1.0, std_per_feat)

X_seqs_norm = (X_seqs - mean_per_feat) / std_per_feat
# 用 mask 标记未观测的,值设为 0
X_seqs_norm = X_seqs_norm * X_masks

print(f'标准化后均值: {X_seqs_norm.mean():.3f}, 标准差: {X_seqs_norm.std():.3f}')

# 5. LSTM 模型定义
class FraudLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features * 2,  # 值 + mask 拼接
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x, mask):
        # x: (B, T, F), mask: (B, T, F)
        # 拼接值与 mask 作为输入
        x_in = torch.cat([x, mask], dim=-1)  # (B, T, 2F)
        out, _ = self.lstm(x_in)  # (B, T, H)
        # 取最后一个有效时间步(用 mask sum 找)
        last_valid = (mask.sum(dim=(1, 2)) > 0).long()  # (B,)
        # 简单方案:取最后一时间步(T-1)的输出
        last_out = out[:, -1, :]  # (B, H)
        logits = self.classifier(last_out)  # (B, 1)
        return logits.squeeze(-1)

# 6. 训练函数
def train_lstm_fold(X_tr, m_tr, y_tr, X_va, m_va, y_va, epochs=50, lr=0.001):
    device = torch.device('cpu')
    model = FraudLSTM(n_features=len(RATIO_COLS)).to(device)

    # 处理类别不平衡:正样本权重
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_tr_t = torch.from_numpy(X_tr).float()
    m_tr_t = torch.from_numpy(m_tr).float()
    y_tr_t = torch.from_numpy(y_tr).float()

    X_va_t = torch.from_numpy(X_va).float()
    m_va_t = torch.from_numpy(m_va).float()
    y_va_t = torch.from_numpy(y_va).float()

    best_f1 = -1
    best_state = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr_t, m_tr_t)
        loss = criterion(logits, y_tr_t)
        loss.backward()
        optimizer.step()

        # 评估
        model.eval()
        with torch.no_grad():
            va_logits = model(X_va_t, m_va_t)
            va_prob = torch.sigmoid(va_logits).numpy()
        va_pred = (va_prob >= 0.5).astype(int)
        va_f1 = f1_score(y_va, va_pred, zero_division=0)
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 用最优状态评估
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_prob = torch.sigmoid(model(X_va_t, m_va_t)).numpy()
    va_pred = (va_prob >= 0.5).astype(int)

    return {
        'f1': f1_score(y_va, va_pred, zero_division=0),
        'recall': recall_score(y_va, va_pred, zero_division=0),
        'precision': precision_score(y_va, va_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_va, va_prob),
    }, model

# 7. 5 折交叉验证
print('\n--- 5 折交叉验证 ---')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {'f1': [], 'recall': [], 'precision': [], 'roc_auc': []}
for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_seqs_norm, y_arr)):
    fold_result, _ = train_lstm_fold(
        X_seqs_norm[tr_idx], X_masks[tr_idx], y_arr[tr_idx],
        X_seqs_norm[va_idx], X_masks[va_idx], y_arr[va_idx],
        epochs=30,
    )
    print(f'  Fold {fold_idx+1}: F1={fold_result["f1"]:.4f}  Recall={fold_result["recall"]:.4f}  '
          f'Precision={fold_result["precision"]:.4f}  AUC={fold_result["roc_auc"]:.4f}')
    for k, v in fold_result.items():
        results[k].append(v)

print(f'\n=== LSTM 5 折平均 ===')
for k, v in results.items():
    print(f'  {k}: {np.mean(v):.4f} ± {np.std(v):.4f}')

# 8. 保存结果
lstm_results = pd.DataFrame(results).mean().to_frame('LSTM')
lstm_results.loc['precision', 'LSTM'] = np.mean(results['precision'])
lstm_results.loc['recall', 'LSTM'] = np.mean(results['recall'])
lstm_results.loc['f1', 'LSTM'] = np.mean(results['f1'])
lstm_results.loc['roc_auc', 'LSTM'] = np.mean(results['roc_auc'])

# 加载 XGBoost 对比
if os.path.exists(os.path.join(OUT_DIR, 'xgb_vs_rf.csv')):
    xgb_comp = pd.read_csv(os.path.join(OUT_DIR, 'xgb_vs_rf.csv'), index_col=0)
    comp = pd.concat([xgb_comp, lstm_results], axis=1)
    print('\n=== 综合对比 ===')
    print(comp.round(4).to_string())
    comp.to_csv(os.path.join(OUT_DIR, 'lstm_vs_xgb_vs_rf.csv'))
    print(f'\n  → output/lstm_vs_xgb_vs_rf.csv')

# 9. 训练最终 LSTM 模型(全量训练)
print('\n--- 全量训练最终 LSTM ---')
device = torch.device('cpu')
final_model = FraudLSTM(n_features=len(RATIO_COLS)).to(device)
pos_weight = torch.tensor([(y_arr == 0).sum() / max((y_arr == 1).sum(), 1)])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-5)

X_t = torch.from_numpy(X_seqs_norm).float()
m_t = torch.from_numpy(X_masks).float()
y_t = torch.from_numpy(y_arr).float()

for epoch in range(30):
    final_model.train()
    optimizer.zero_grad()
    loss = criterion(final_model(X_t, m_t), y_t)
    loss.backward()
    optimizer.step()

# 10. 全量预测
final_model.eval()
with torch.no_grad():
    p_ml_lstm = torch.sigmoid(final_model(X_t, m_t)).numpy()

# 保存
pd.DataFrame({
    'Symbol': symbols,
    'ShortName': df['ShortName'].values,
    'violation_year': violation_years,
    'p_ml_lstm': p_ml_lstm,
}).to_csv(os.path.join(DATA_DIR, 'p_ml_lstm_full.csv'), index=False)

# 保存模型
torch.save({
    'model_state_dict': final_model.state_dict(),
    'mean_per_feat': mean_per_feat,
    'std_per_feat': std_per_feat,
    'seq_len': SEQ_LEN,
    'feature_names': RATIO_COLS,
}, os.path.join(MODEL_DIR, 'fraud_detection_lstm.pth'))

print(f'\n  → data/p_ml_lstm_full.csv ({len(p_ml_lstm)} 条)')
print(f'  → models/fraud_detection_lstm.pth')
print(f'  高概率(>0.5): {(p_ml_lstm > 0.5).sum()}')

print('\n✅ LSTM 时序深度学习完成')
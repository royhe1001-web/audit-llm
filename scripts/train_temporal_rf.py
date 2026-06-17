#!/usr/bin/env python3
"""
时序特征工程 — 任务 3: 时序 vs baseline 模型对比
================================================
对比:
  A. Baseline:仅用 7 个单年财务比率
  B. 时序增强:7 单年 + 56 维时序特征

输出:
  output/temporal_vs_baseline.csv (对比表)
  models/fraud_detection_rf_temporal.pkl (胜出模型)
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, recall_score, precision_score, roc_auc_score,
                              classification_report)
import joblib

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
OUT_DIR = os.path.join(BASE, "output")

RATIO_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio',
              'asset_turnover', 'net_margin', 'ocf_to_rev']

print('=' * 60)
print('时序 vs Baseline 模型对比')
print('=' * 60)

# 1. 加载数据
df = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_with_temporal.csv'))
df_clean = df.dropna(subset=['roe', 'ann_related']).copy()
df_clean = df_clean[df_clean['ann_related'].isin([0, 1])]
target = 'ann_fin_flag'
df_clean = df_clean.dropna(subset=[target])
df_clean = df_clean[df_clean[target].isin([0, 1])]
y = df_clean[target].astype(int)
print(f'可用样本: {len(df_clean)}, 正样本: {y.sum()} ({y.mean()*100:.1f}%)')

# 2. 准备两组特征
X_baseline = df_clean[RATIO_COLS].copy()

# 时序特征列(除 Symbol, violation_year 外,都是数值特征)
exclude = ['Symbol', 'violation_year', 'feature_year', target, 'ann_related',
           'third_party_flag', 'industry', 'ShortName', 'area', 'list_date',
           'list_year', 'list_years']
temporal_cols = [c for c in df_clean.columns
                 if c not in exclude
                 and df_clean[c].dtype in ['int64', 'float64']
                 and c not in RATIO_COLS]
X_temporal = df_clean[RATIO_COLS + temporal_cols].copy()
print(f'Baseline 特征数: {X_baseline.shape[1]}')
print(f'时序增强特征数: {X_temporal.shape[1]} (新增 {len(temporal_cols)} 维)')

# 3. 训练与评估
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_eval(X, label):
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    print(f'\n--- 模型 {label}: {X.shape[1]} 维特征 ---')
    scores = {
        'f1': cross_val_score(pipe, X, y, cv=cv, scoring='f1', n_jobs=-1),
        'recall': cross_val_score(pipe, X, y, cv=cv, scoring='recall', n_jobs=-1),
        'precision': cross_val_score(pipe, X, y, cv=cv, scoring='precision', n_jobs=-1),
        'roc_auc': cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1),
    }
    for k, v in scores.items():
        print(f'  {k}: {v.mean():.4f} ± {v.std():.4f}')
    return pipe, scores

pipe_a, scores_a = train_eval(X_baseline, 'A: Baseline (7 比率)')
pipe_b, scores_b = train_eval(X_temporal, 'B: 时序增强 (63 维)')

# 4. 对比
print('\n' + '=' * 60)
print('对比结果')
print('=' * 60)
comparison = pd.DataFrame({
    'A_Baseline_7列': {k: v.mean() for k, v in scores_a.items()},
    'B_时序增强_63列': {k: v.mean() for k, v in scores_b.items()},
}).T
comparison = comparison[['precision', 'recall', 'f1', 'roc_auc']]
comparison['delta_f1'] = comparison['f1'] - comparison.loc['A_Baseline_7列', 'f1']
comparison['delta_recall'] = comparison['recall'] - comparison.loc['A_Baseline_7列', 'recall']
print(comparison.round(4).to_string())

# 5. 提升分析
f1_gain = scores_b['f1'].mean() - scores_a['f1'].mean()
recall_gain = scores_b['recall'].mean() - scores_a['recall'].mean()
print(f'\n时序增强带来的提升:')
print(f'  F1: {f1_gain:+.4f} ({"提升" if f1_gain > 0 else "下降"})')
print(f'  Recall: {recall_gain:+.4f} ({"提升" if recall_gain > 0 else "下降"})')

# 6. 保存对比
comparison.to_csv(os.path.join(OUT_DIR, 'temporal_vs_baseline.csv'))
print(f'\n  → output/temporal_vs_baseline.csv')

# 7. 用胜出模型全量重训 + 保存
print('\n' + '=' * 60)
print('胜出模型全量重训')
print('=' * 60)

if f1_gain > 0:
    winner = 'B_时序增强'
    pipe_winner = pipe_b
    X_winner = X_temporal
    print(f'胜出: B (时序增强), F1 提升 {f1_gain:.4f}')
else:
    winner = 'A_Baseline'
    pipe_winner = pipe_a
    X_winner = X_baseline
    print(f'胜出: A (Baseline, 时序未带来提升), F1 持平/略降 {f1_gain:.4f}')

pipe_winner.fit(X_winner, y)
out_path = os.path.join(MODEL_DIR, 'fraud_detection_rf_temporal.pkl')
joblib.dump({
    'pipeline': pipe_winner,
    'feature_names': list(X_winner.columns),
    'winner': winner,
    'f1_gain': f1_gain,
}, out_path)
print(f'  → {out_path}')

# 8. 用胜出模型对全量数据预测
print('\n--- 应用到全量数据 ---')
df_full = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_with_temporal.csv'))
X_full = df_full[X_winner.columns].copy()
p_ml_temporal = pipe_winner.predict_proba(X_full)[:, 1]

out_csv = os.path.join(DATA_DIR, 'p_ml_temporal_full.csv')
pd.DataFrame({
    'Symbol': df_full['Symbol'],
    'ShortName': df_full['ShortName'],
    'violation_year': df_full['violation_year'],
    'p_ml_temporal': p_ml_temporal,
}).to_csv(out_csv, index=False)
print(f'  → {out_csv}')
print(f'  高概率(>0.5): {(p_ml_temporal > 0.5).sum()}')
print(f'  中概率(0.3-0.5): {((p_ml_temporal > 0.3) & (p_ml_temporal <= 0.5)).sum()}')

print('\n✅ 时序 vs Baseline 对比完成')
#!/usr/bin/env python3
"""
阶段四·优化 3: SMOTE 采样 + RF 重训
====================================
对比:原 RF(class_weight='balanced') vs SMOTE 增强 RF
目标:提升正样本(舞弊)召回率,降低漏报
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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
OUT_DIR = os.path.join(BASE, "output")

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']

print('=' * 60)
print('SMOTE 采样 + RF 重训')
print('=' * 60)

# ============================================================
# 1. 加载与准备数据
# ============================================================
df = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_combined.csv'))
df_clean = df.dropna(subset=['roe', 'ann_related']).copy()
df_clean = df_clean[df_clean['ann_related'].isin([0, 1])]
target = 'ann_fin_flag'
df_clean = df_clean.dropna(subset=[target])
df_clean = df_clean[df_clean[target].isin([0, 1])]
y = df_clean[target].astype(int)
X = df_clean[FIN_COLS].copy()

print(f'样本: {len(X)}, 正样本: {y.sum()} ({y.mean()*100:.1f}%)')

# ============================================================
# 2. 模型 A: 原版(class_weight='balanced')
# ============================================================
print('\n--- 模型 A: 原版 RF(class_weight=balanced) ---')

pipe_a = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_a = {
    'f1': cross_val_score(pipe_a, X, y, cv=cv, scoring='f1', n_jobs=-1),
    'recall': cross_val_score(pipe_a, X, y, cv=cv, scoring='recall', n_jobs=-1),
    'precision': cross_val_score(pipe_a, X, y, cv=cv, scoring='precision', n_jobs=-1),
    'roc_auc': cross_val_score(pipe_a, X, y, cv=cv, scoring='roc_auc', n_jobs=-1),
}
for k, v in scores_a.items():
    print(f'  {k}: {v.mean():.4f} ± {v.std():.4f}')

# ============================================================
# 3. 模型 B: SMOTE + RF(无 class_weight)
# ============================================================
print('\n--- 模型 B: SMOTE + RF ---')

pipe_b = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('clf', RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=10,
        random_state=42, n_jobs=-1)),
])

scores_b = {
    'f1': cross_val_score(pipe_b, X, y, cv=cv, scoring='f1', n_jobs=-1),
    'recall': cross_val_score(pipe_b, X, y, cv=cv, scoring='recall', n_jobs=-1),
    'precision': cross_val_score(pipe_b, X, y, cv=cv, scoring='precision', n_jobs=-1),
    'roc_auc': cross_val_score(pipe_b, X, y, cv=cv, scoring='roc_auc', n_jobs=-1),
}
for k, v in scores_b.items():
    print(f'  {k}: {v.mean():.4f} ± {v.std():.4f}')

# ============================================================
# 4. 模型 C: SMOTE + RF + class_weight 微调
# ============================================================
print('\n--- 模型 C: SMOTE + RF + class_weight=balanced_subsample ---')

pipe_c = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('clf', RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=10,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1)),
])

scores_c = {
    'f1': cross_val_score(pipe_c, X, y, cv=cv, scoring='f1', n_jobs=-1),
    'recall': cross_val_score(pipe_c, X, y, cv=cv, scoring='recall', n_jobs=-1),
    'precision': cross_val_score(pipe_c, X, y, cv=cv, scoring='precision', n_jobs=-1),
    'roc_auc': cross_val_score(pipe_c, X, y, cv=cv, scoring='roc_auc', n_jobs=-1),
}
for k, v in scores_c.items():
    print(f'  {k}: {v.mean():.4f} ± {v.std():.4f}')

# ============================================================
# 5. 对比表
# ============================================================
print('\n' + '=' * 60)
print('模型对比')
print('=' * 60)

comparison = pd.DataFrame({
    '原版_RF(class_weight=balanced)': {k: v.mean() for k, v in scores_a.items()},
    'SMOTE+RF': {k: v.mean() for k, v in scores_b.items()},
    'SMOTE+RF+balanced_subsample': {k: v.mean() for k, v in scores_c.items()},
}).T
comparison = comparison[['precision', 'recall', 'f1', 'roc_auc']]
print(comparison.round(4).to_string())

# 保存对比
comparison.to_csv(os.path.join(OUT_DIR, 'model_comparison_smote.csv'))
print(f'\n  → output/model_comparison_smote.csv')

# ============================================================
# 6. 选最优模型,全量重训 + 保存
# ============================================================
print('\n' + '=' * 60)
print('选择最优模型并全量重训')
print('=' * 60)

best_name = comparison['f1'].idxmax()
best_pipe = {'原版_RF(class_weight=balanced)': pipe_a,
             'SMOTE+RF': pipe_b,
             'SMOTE+RF+balanced_subsample': pipe_c}[best_name]
print(f'  最优模型: {best_name}')
print(f'  F1: {comparison.loc[best_name, "f1"]:.4f}')

best_pipe.fit(X, y)
out_path = os.path.join(MODEL_DIR, 'fraud_detection_rf_smote.pkl')
joblib.dump(best_pipe, out_path)
print(f'  → {out_path}')

# ============================================================
# 7. 用最优模型对全量数据预测,产出 p_ml 增强版
# ============================================================
print('\n--- 应用最优模型到全量数据 ---')

# 重新加载全量(包括无标签的)
df_full = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_combined.csv'))
X_full = df_full[FIN_COLS].copy()
p_ml_new = best_pipe.predict_proba(X_full)[:, 1]

df_full['p_ml_smote'] = p_ml_new
out_csv = os.path.join(DATA_DIR, 'p_ml_smote_full.csv')
df_full[['Symbol', 'ShortName', 'violation_year', 'p_ml_smote']].to_csv(out_csv, index=False)
print(f'  → {out_csv}')
print(f'  高概率(>0.5)公司: {(p_ml_new > 0.5).sum()}')
print(f'  中概率(0.3-0.5): {((p_ml_new > 0.3) & (p_ml_new <= 0.5)).sum()}')
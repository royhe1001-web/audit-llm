#!/usr/bin/env python3
"""
扩展 C 快速训练: 仅用 pledge_stat 验证效果
============================================
对比:
  A. v2.2 Baseline(XGBoost + 7 比率)
  C. + pledge_stat(质押数据)

不保存模型,只评估效果(等 fina_indicator 拉完再跑完整 v2.3 训练)
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, recall_score, precision_score, roc_auc_score)
from xgboost import XGBClassifier

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")

ORIG_7 = ['roe', 'roa', 'debt_ratio', 'current_ratio',
          'asset_turnover', 'net_margin', 'ocf_to_rev']

print('=' * 60)
print('快速训练: XGBoost + pledge_stat 验证')
print('=' * 60)

# 1. 加载主表
df = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_combined.csv'))
df['Symbol'] = df['Symbol'].astype(str).str.zfill(6)
df['violation_year'] = df['violation_year'].astype(int)
df['feature_year'] = df['violation_year'] - 1
print(f'主表: {df.shape}')

# 2. 合并 pledge_stat
pled = pd.read_csv(os.path.join(DATA_DIR, 'pledge_stat_full.csv'))
pled['Symbol'] = pled['ts_code'].str.split('.').str[0]
pled['end_date'] = pled['end_date'].astype(str)
pled['feature_year'] = pled['end_date'].str[:4].astype(int)

# 关键字段
pled_keep = ['Symbol', 'feature_year', 'pledge_count', 'unrest_pledge',
             'rest_pledge', 'total_share', 'pledge_ratio']
pled_keep = [c for c in pled_keep if c in pled.columns]
pled = pled[pled_keep].copy()

# 去重(取每年最后一条)
pled = pled.sort_values(['Symbol', 'feature_year']).drop_duplicates(
    subset=['Symbol', 'feature_year'], keep='last')

df = df.merge(pled, on=['Symbol', 'feature_year'], how='left')
print(f'合并 pledge 后: {df.shape}')
pled_extra = [c for c in pled_keep if c not in ['Symbol', 'feature_year']]
print(f'新增字段: {pled_extra}')
print(f'pledge_ratio 覆盖: {df["pledge_ratio"].notna().sum()}/{len(df)} ({df["pledge_ratio"].notna().mean()*100:.1f}%)')

# 3. 训练样本
df_clean = df.dropna(subset=['roe', 'ann_related']).copy()
df_clean = df_clean[df_clean['ann_related'].isin([0, 1])]
target = 'ann_fin_flag'
df_clean = df_clean.dropna(subset=[target])
df_clean = df_clean[df_clean[target].isin([0, 1])]
y = df_clean[target].astype(int)
print(f'\n样本: {len(df_clean)}, 正样本: {y.sum()} ({y.mean()*100:.1f}%)')

# 4. 两组特征
X_a = df_clean[ORIG_7].copy()
X_c = df_clean[ORIG_7 + pled_extra].copy()
print(f'\nA. Baseline: {X_a.shape[1]} 列')
print(f'C. + pledge: {X_c.shape[1]} 列')

# 5. 训练
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def make_xgb(y_data):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=(y_data == 0).sum() / max((y_data == 1).sum(), 1),
            random_state=42, n_jobs=-1, eval_metric='logloss'))
    ])

def evaluate(pipe, X, y_data, label):
    scores = {
        'f1': cross_val_score(pipe, X, y_data, cv=cv, scoring='f1', n_jobs=-1),
        'recall': cross_val_score(pipe, X, y_data, cv=cv, scoring='recall', n_jobs=-1),
        'precision': cross_val_score(pipe, X, y_data, cv=cv, scoring='precision', n_jobs=-1),
        'roc_auc': cross_val_score(pipe, X, y_data, cv=cv, scoring='roc_auc', n_jobs=-1),
    }
    print(f'\n--- {label}: {X.shape[1]} 维 ---')
    for k, v in scores.items():
        print(f'  {k}: {v.mean():.4f} ± {v.std():.4f}')
    return scores

results = {}
results['A_baseline'] = evaluate(make_xgb(y), X_a, y, 'A: Baseline (7 比率)')
results['C_pledge'] = evaluate(make_xgb(y), X_c, y, 'C: + pledge_stat')

# 6. 对比
print('\n' + '=' * 60)
print('对比结果')
print('=' * 60)
comp = pd.DataFrame({k: {m: v[m].mean() for m in ['precision', 'recall', 'f1', 'roc_auc']}
                      for k, v in results.items()}).T
comp = comp[['precision', 'recall', 'f1', 'roc_auc']]
print(comp.round(4).to_string())

a_f1 = results['A_baseline']['f1'].mean()
c_f1 = results['C_pledge']['f1'].mean()
delta = c_f1 - a_f1
print(f'\npledge 提升: F1 {a_f1:.4f} → {c_f1:.4f} (Δ={delta:+.4f})')

if delta > 0:
    print(f'✅ pledge_stat 带来正向提升 +{delta:.4f}')
else:
    print(f'⚠️ pledge_stat 未带来提升 ({delta:+.4f})')

print('\n✅ 快速训练完成')
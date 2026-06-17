#!/usr/bin/env python3
"""
扩展 1: 在 2,634 样本(时序扩展数据集)上重训 v2.3
====================================================
数据集: data/fraud_features_with_temporal.csv(2,634 样本,75.2% 正样本)
对比:
  A. v2.2 baseline(7 比率)
  B. + fina_indicator
  C. + fina + pledge
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
import joblib

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
OUT_DIR = os.path.join(BASE, "output")

ORIG_7 = ['roe', 'roa', 'debt_ratio', 'current_ratio',
          'asset_turnover', 'net_margin', 'ocf_to_rev']

print('=' * 60)
print('v2.3 重训 — 2,634 样本(时序扩展)')
print('=' * 60)

# 1. 加载
df = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_with_temporal.csv'))
df['Symbol'] = df['Symbol'].astype(str).str.zfill(6)
df['violation_year'] = df['violation_year'].astype(int)
df['feature_year'] = df['violation_year'] - 1
print(f'主表: {df.shape}')

# 2. 合并 fina_indicator
print('\n[1] 合并 fina_indicator...')
fina = pd.read_csv(os.path.join(DATA_DIR, 'fina_indicator_full.csv'))
fina['Symbol'] = fina['ts_code'].str.split('.').str[0]
fina['feature_year'] = fina['end_date'].astype(str).str[:4].astype(int)
fina_extra = [c for c in fina.columns
              if c not in ['ts_code', 'ann_date', 'end_date', 'update_flag', 'Symbol', 'feature_year']]
for c in ['roe', 'roa', 'debt_ratio', 'current_ratio']:
    if c in fina_extra:
        fina_extra.remove(c)
fina = fina[['Symbol', 'feature_year'] + fina_extra].drop_duplicates(
    subset=['Symbol', 'feature_year'], keep='last')
df = df.merge(fina, on=['Symbol', 'feature_year'], how='left')
print(f'  合并后: {df.shape} (新增 {len(fina_extra)} 列)')

# 3. 合并 pledge_stat
print('\n[2] 合并 pledge_stat...')
pled = pd.read_csv(os.path.join(DATA_DIR, 'pledge_stat_full.csv'))
pled['Symbol'] = pled['ts_code'].str.split('.').str[0]
pled['feature_year'] = pled['end_date'].astype(str).str[:4].astype(int)
pled_keep = ['Symbol', 'feature_year', 'pledge_count', 'unrest_pledge',
             'rest_pledge', 'total_share', 'pledge_ratio']
pled = pled[pled_keep].drop_duplicates(
    subset=['Symbol', 'feature_year'], keep='last')
df = df.merge(pled, on=['Symbol', 'feature_year'], how='left')
pled_extra = [c for c in pled_keep if c not in ['Symbol', 'feature_year']]
print(f'  合并后: {df.shape} (新增 {len(pled_extra)} 列)')

# 4. 训练样本(2,634 样本,与 v2.2 baseline 同口径)
df_clean = df.dropna(subset=['roe', 'ann_related']).copy()
df_clean = df_clean[df_clean['ann_related'].isin([0, 1])]
target = 'ann_fin_flag'
df_clean = df_clean.dropna(subset=[target])
df_clean = df_clean[df_clean[target].isin([0, 1])]
y = df_clean[target].astype(int)
print(f'\n样本: {len(df_clean)}, 正样本: {y.sum()} ({y.mean()*100:.1f}%)')

# 5. 三组特征(用 df_clean.columns 实际存在的列,避免 merge 后列名冲突)
fina_cols = [c for c in fina_extra if c in df_clean.columns]
pled_cols = [c for c in pled_extra if c in df_clean.columns]
print(f'\n实际可用的 fina 列: {len(fina_cols)} / {len(fina_extra)}')
print(f'实际可用的 pledge 列: {len(pled_cols)} / {len(pled_extra)}')

X_baseline = df_clean[ORIG_7].copy()
X_with_fina = df_clean[ORIG_7 + fina_cols].copy()
X_with_all = df_clean[ORIG_7 + fina_cols + pled_cols].copy()
print(f'\nA. Baseline: {X_baseline.shape[1]} 列')
print(f'B. + fina_indicator: {X_with_fina.shape[1]} 列')
print(f'C. + fina + pledge: {X_with_all.shape[1]} 列')

# 6. 训练
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
results['A_baseline'] = evaluate(make_xgb(y), X_baseline, y, 'A: Baseline (7 比率)')
results['B_with_fina'] = evaluate(make_xgb(y), X_with_fina, y, 'B: + fina_indicator')
results['C_with_all'] = evaluate(make_xgb(y), X_with_all, y, 'C: + fina + pledge')

# 7. 对比
print('\n' + '=' * 60)
print('v2.3 模型对比(2,634 样本)')
print('=' * 60)
comp = pd.DataFrame({k: {m: v[m].mean() for m in ['precision', 'recall', 'f1', 'roc_auc']}
                      for k, v in results.items()}).T
comp = comp[['precision', 'recall', 'f1', 'roc_auc']]
print(comp.round(4).to_string())
comp.to_csv(os.path.join(OUT_DIR, 'v23_model_comparison_2634.csv'))

a_f1 = results['A_baseline']['f1'].mean()
b_f1 = results['B_with_fina']['f1'].mean()
c_f1 = results['C_with_all']['f1'].mean()
print(f'\n提升:')
print(f'  v2.2 baseline(7 比率):F1={a_f1:.4f}')
print(f'  + fina_indicator:F1={b_f1:.4f} (Δ={b_f1-a_f1:+.4f})')
print(f'  + fina + pledge:F1={c_f1:.4f} (Δ={c_f1-a_f1:+.4f})')

# 8. 选最优 + 保存
best_f1 = comp['f1'].max()
best_name = comp['f1'].idxmax()
print(f'\n胜出: {best_name} (F1={best_f1:.4f})')

X_winner = {'A_baseline': X_baseline,
            'B_with_fina': X_with_fina,
            'C_with_all': X_with_all}[best_name]

# 实际使用的特征列表(去重)
all_features = list(dict.fromkeys(X_winner.columns.tolist()))
pipe_winner = make_xgb(y)
pipe_winner.fit(X_winner, y)

# Feature importance
xgb_clf = pipe_winner.named_steps['clf']
imp = pd.DataFrame({
    'feature': X_winner.columns,
    'importance': xgb_clf.feature_importances_,
}).sort_values('importance', ascending=False)
imp['rank'] = range(1, len(imp) + 1)
print('\n=== Feature Importance TOP 20 ===')
print(imp.head(20).to_string(index=False))
imp.to_csv(os.path.join(OUT_DIR, 'v23_feature_importance_2634.csv'), index=False)

# 9. 保存模型(覆盖 v2.3 1554 版本)
joblib.dump(pipe_winner, os.path.join(MODEL_DIR, 'fraud_detection_xgb_combined.pkl'))
print(f'\n  → models/fraud_detection_xgb_combined.pkl (覆盖 v2.3 1554 版)')

# 更新 metadata
metadata = {
    'pipeline_path': 'models/fraud_detection_xgb_combined.pkl',
    'model_type': 'XGBoost',
    'version': 'v2.3.1',
    'sample_count': len(df_clean),
    'f1_cv': best_f1,
    'feature_names': all_features,
    'n_features': X_winner.shape[1],
    'training_data': 'fraud_features_with_temporal.csv',
}
joblib.dump(metadata, os.path.join(MODEL_DIR, 'fraud_detection_xgb_combined_metadata.pkl'))

print('\n✅ v2.3 重训完成(2,634 样本)')
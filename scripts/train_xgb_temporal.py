#!/usr/bin/env python3
"""
时序 v2 — 任务 2: XGBoost + feature_importance
==============================================
用 XGBoost 替代 RF,看是否能利用时序特征 + 输出 feature_importance

对比:
  A. RF + 7 单年(Baseline)
  B. RF + 63 时序(temporal_v1 失败)
  C. XGBoost + 7 单年
  D. XGBoost + 63 时序

输出:
  output/xgb_vs_rf.csv (4 模型对比)
  output/xgb_feature_importance.csv (XGBoost 特征重要性)
  models/fraud_detection_xgb_temporal.pkl (胜出 XGBoost)
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, recall_score, precision_score, roc_auc_score)
from xgboost import XGBClassifier
import joblib

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
OUT_DIR = os.path.join(BASE, "output")

RATIO_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio',
              'asset_turnover', 'net_margin', 'ocf_to_rev']

print('=' * 60)
print('XGBoost + feature_importance 对比')
print('=' * 60)

# 1. 加载数据
df = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_with_temporal.csv'))
df_clean = df.dropna(subset=['roe', 'ann_related']).copy()
df_clean = df_clean[df_clean['ann_related'].isin([0, 1])]
target = 'ann_fin_flag'
df_clean = df_clean.dropna(subset=[target])
df_clean = df_clean[df_clean[target].isin([0, 1])]
y = df_clean[target].astype(int)
print(f'样本: {len(df_clean)}, 正样本: {y.sum()} ({y.mean()*100:.1f}%)')

# 2. 准备特征
X_baseline = df_clean[RATIO_COLS].copy()
exclude = ['Symbol', 'violation_year', 'feature_year', target, 'ann_related',
           'third_party_flag', 'industry', 'ShortName', 'area', 'list_date',
           'list_year', 'list_years']
temporal_cols = [c for c in df_clean.columns
                 if c not in exclude
                 and df_clean[c].dtype in ['int64', 'float64']
                 and c not in RATIO_COLS]
X_temporal = df_clean[RATIO_COLS + temporal_cols].copy()
print(f'Baseline 特征: {X_baseline.shape[1]}, 时序增强: {X_temporal.shape[1]}')

# 3. 模型对比
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def make_rf():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=-1))
    ])

def make_xgb():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # 自动平衡
            random_state=42, n_jobs=-1, eval_metric='logloss'))
    ])

def evaluate(pipe, X, label):
    scores = {
        'f1': cross_val_score(pipe, X, y, cv=cv, scoring='f1', n_jobs=-1),
        'recall': cross_val_score(pipe, X, y, cv=cv, scoring='recall', n_jobs=-1),
        'precision': cross_val_score(pipe, X, y, cv=cv, scoring='precision', n_jobs=-1),
        'roc_auc': cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1),
    }
    print(f'\n--- {label}: {X.shape[1]} 维 ---')
    for k, v in scores.items():
        print(f'  {k}: {v.mean():.4f} ± {v.std():.4f}')
    return scores

results = {}

# A: RF + Baseline
results['A_RF_baseline'] = evaluate(make_rf(), X_baseline, 'A: RF + Baseline')

# B: RF + 时序
results['B_RF_temporal'] = evaluate(make_rf(), X_temporal, 'B: RF + 时序 (63 维)')

# C: XGBoost + Baseline
results['C_XGB_baseline'] = evaluate(make_xgb(), X_baseline, 'C: XGBoost + Baseline')

# D: XGBoost + 时序
results['D_XGB_temporal'] = evaluate(make_xgb(), X_temporal, 'D: XGBoost + 时序 (63 维)')

# 4. 对比表
print('\n' + '=' * 60)
print('4 模型对比')
print('=' * 60)
comp = pd.DataFrame({k: {m: v[m].mean() for m in ['precision', 'recall', 'f1', 'roc_auc']}
                      for k, v in results.items()}).T
comp = comp[['precision', 'recall', 'f1', 'roc_auc']]
print(comp.round(4).to_string())

# 5. 保存对比
comp.to_csv(os.path.join(OUT_DIR, 'xgb_vs_rf.csv'))
print(f'\n  → output/xgb_vs_rf.csv')

# 6. 选最优 XGBoost 模型
xgb_temporal_f1 = results['D_XGB_temporal']['f1'].mean()
xgb_baseline_f1 = results['C_XGB_baseline']['f1'].mean()

if xgb_temporal_f1 > xgb_baseline_f1:
    print(f'\nXGBoost 时序 F1 ({xgb_temporal_f1:.4f}) > XGBoost Baseline ({xgb_baseline_f1:.4f})')
    pipe_winner = make_xgb()
    X_winner = X_temporal
    winner_name = 'XGBoost_temporal'
else:
    print(f'\nXGBoost Baseline F1 ({xgb_baseline_f1:.4f}) ≥ XGBoost 时序 F1 ({xgb_temporal_f1:.4f})')
    pipe_winner = make_xgb()
    X_winner = X_baseline
    winner_name = 'XGBoost_baseline'

pipe_winner.fit(X_winner, y)

# 7. feature_importance
print('\n' + '=' * 60)
print('Feature Importance(XGBoost)')
print('=' * 60)
xgb_clf = pipe_winner.named_steps['clf']
imp = pd.DataFrame({
    'feature': X_winner.columns,
    'importance_gain': xgb_clf.feature_importances_,
}).sort_values('importance_gain', ascending=False)
imp['rank'] = range(1, len(imp) + 1)
imp['cum_importance'] = imp['importance_gain'].cumsum() / imp['importance_gain'].sum()

print(imp.head(15).to_string(index=False))

imp_path = os.path.join(OUT_DIR, 'xgb_feature_importance.csv')
imp.to_csv(imp_path, index=False)
print(f'\n  → {imp_path}')

# 8. 保存模型
model_path = os.path.join(MODEL_DIR, 'fraud_detection_xgb_temporal.pkl')
joblib.dump({
    'pipeline': pipe_winner,
    'feature_names': list(X_winner.columns),
    'winner': winner_name,
    'f1_xgb_temporal': xgb_temporal_f1,
    'f1_xgb_baseline': xgb_baseline_f1,
    'feature_importance': imp,
}, model_path)
print(f'  → {model_path}')

# 9. 全量预测概率
print('\n--- 全量预测 ---')
df_full = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_with_temporal.csv'))
X_full = df_full[X_winner.columns].copy()
p_ml_xgb = pipe_winner.predict_proba(X_full)[:, 1]
pd.DataFrame({
    'Symbol': df_full['Symbol'],
    'ShortName': df_full['ShortName'],
    'violation_year': df_full['violation_year'],
    'p_ml_xgb': p_ml_xgb,
}).to_csv(os.path.join(DATA_DIR, 'p_ml_xgb_full.csv'), index=False)
print(f'  → data/p_ml_xgb_full.csv')
print(f'  高概率(>0.5): {(p_ml_xgb > 0.5).sum()}')

# 10. 时序特征子集的重要性(只看时序特征)
imp_temporal_only = imp[imp['feature'].isin(temporal_cols)].head(20)
print('\n=== 时序特征重要性 TOP 20(若有时序特征被选入) ===')
if len(imp_temporal_only) > 0:
    print(imp_temporal_only.to_string(index=False))
else:
    print('时序特征未被 XGBoost 选中(importance 均为 0)')

print('\n✅ XGBoost + feature_importance 完成')
#!/usr/bin/env python3
"""
时序 v2 — 任务 4.5: XGBoost + 长窗口特征训练
============================================
对比:
  A. XGBoost + 7 单年(Baseline,最优:F1=0.8530)
  B. XGBoost + 7 单年 + 35 长窗口
  C. XGBoost + 7 单年 + 56 短时序 + 35 长窗口(综合)

输出:
  output/xgb_long_window_results.csv
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

RATIO_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio',
              'asset_turnover', 'net_margin', 'ocf_to_rev']

print('=' * 60)
print('XGBoost + 长窗口特征对比')
print('=' * 60)

# 1. 加载
df = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_with_long_window.csv'))
df_clean = df.dropna(subset=['roe', 'ann_related']).copy()
df_clean = df_clean[df_clean['ann_related'].isin([0, 1])]
target = 'ann_fin_flag'
df_clean = df_clean.dropna(subset=[target])
df_clean = df_clean[df_clean[target].isin([0, 1])]
y = df_clean[target].astype(int)
print(f'样本: {len(df_clean)}, 正样本: {y.sum()}')

# 2. 三组特征
X_baseline = df_clean[RATIO_COLS].copy()

# 长窗口特征
lw_cols = [c for c in df_clean.columns
           if any(x in c for x in ['cagr5y', 'cagr10y', 'loss_years_5y',
                                    'volatility_5y', 'slope_5y', 'total_loss',
                                    'mean_volatility', 'high_leverage_years',
                                    'liquidity_stress_years'])]
X_longwin = df_clean[RATIO_COLS + lw_cols].copy()
print(f'Baseline: {X_baseline.shape[1]} 维')
print(f'+ 长窗口: {X_longwin.shape[1]} 维 (新增 {len(lw_cols)} 维)')

# 短时序特征(从时序 v1)— 注意要先做同样的 dropna 过滤再 merge
df_temp_full = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_with_temporal.csv'))
df_temp_full['Symbol'] = df_temp_full['Symbol'].astype(str).str.zfill(6)
df_temp_clean = df_temp_full[df_temp_full['Symbol'].isin(df_clean['Symbol'])].copy()
short_temp_cols = [c for c in df_temp_clean.columns
                   if c not in df_clean.columns
                   and c not in ['Symbol', 'violation_year', 'feature_year',
                                  target, 'ann_related', 'third_party_flag',
                                  'industry', 'ShortName', 'area', 'list_date',
                                  'list_year', 'list_years']
                   and df_temp_clean[c].dtype in ['int64', 'float64']
                   and c not in RATIO_COLS]
# 一对多 merge 会膨胀,要先在 df_temp 上 drop_duplicates
df_temp_dedup = df_temp_clean[['Symbol', 'violation_year'] + short_temp_cols].drop_duplicates(
    subset=['Symbol', 'violation_year'])
df_combo = df_clean.merge(
    df_temp_dedup,
    on=['Symbol', 'violation_year'], how='left'
)
X_combo = df_combo[RATIO_COLS + lw_cols + short_temp_cols].copy()
y_combo = df_combo[target].astype(int)
print(f'+ 短+长: {X_combo.shape[1]} 维 (新增短 {len(short_temp_cols)} + 长 {len(lw_cols)})')
print(f'  y_combo 长度: {len(y_combo)}, 正样本: {y_combo.sum()}')

# 3. 模型训练
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def make_xgb():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
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
results['A_XGB_baseline'] = evaluate(make_xgb(), X_baseline, 'A: XGBoost + Baseline')
results['B_XGB_longwin'] = evaluate(make_xgb(), X_longwin, 'B: XGBoost + 长窗口')
# C 用 y_combo(因为 merge 后可能丢行)
print(f'\n--- C: XGBoost + 短+长 ({X_combo.shape[1]} 维, {len(y_combo)} 样本) ---')
cv_combo = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
def make_xgb_v2(y_data):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=(y_data == 0).sum() / max((y_data == 1).sum(), 1),
            random_state=42, n_jobs=-1, eval_metric='logloss'))
    ])
results['C_XGB_combo'] = {
    'f1': cross_val_score(make_xgb_v2(y_combo), X_combo, y_combo, cv=cv_combo, scoring='f1', n_jobs=-1),
    'recall': cross_val_score(make_xgb_v2(y_combo), X_combo, y_combo, cv=cv_combo, scoring='recall', n_jobs=-1),
    'precision': cross_val_score(make_xgb_v2(y_combo), X_combo, y_combo, cv=cv_combo, scoring='precision', n_jobs=-1),
    'roc_auc': cross_val_score(make_xgb_v2(y_combo), X_combo, y_combo, cv=cv_combo, scoring='roc_auc', n_jobs=-1),
}
print(f'  f1: {results["C_XGB_combo"]["f1"].mean():.4f} ± {results["C_XGB_combo"]["f1"].std():.4f}')
print(f'  recall: {results["C_XGB_combo"]["recall"].mean():.4f}')
print(f'  precision: {results["C_XGB_combo"]["precision"].mean():.4f}')
print(f'  roc_auc: {results["C_XGB_combo"]["roc_auc"].mean():.4f}')

# 4. 对比
print('\n' + '=' * 60)
print('3 模型对比')
print('=' * 60)
comp = pd.DataFrame({k: {m: v[m].mean() for m in ['precision', 'recall', 'f1', 'roc_auc']}
                      for k, v in results.items()}).T
comp = comp[['precision', 'recall', 'f1', 'roc_auc']]
print(comp.round(4).to_string())
comp.to_csv(os.path.join(OUT_DIR, 'xgb_long_window_results.csv'))
print(f'\n  → output/xgb_long_window_results.csv')

# 5. 选最优
best_f1 = comp['f1'].max()
best_name = comp['f1'].idxmax()
print(f'\n胜出: {best_name} (F1={best_f1:.4f})')

# 6. 训练最终 + 保存
X_winner = {'A_XGB_baseline': X_baseline,
            'B_XGB_longwin': X_longwin,
            'C_XGB_combo': X_combo}[best_name]
pipe_winner = make_xgb()
pipe_winner.fit(X_winner, y)

# Feature importance
xgb_clf = pipe_winner.named_steps['clf']
imp = pd.DataFrame({
    'feature': X_winner.columns,
    'importance': xgb_clf.feature_importances_,
}).sort_values('importance', ascending=False)
imp['rank'] = range(1, len(imp) + 1)
print('\n=== Feature Importance TOP 15 ===')
print(imp.head(15).to_string(index=False))

# 看长窗口特征有没有进入 TOP 20
imp_lw = imp[imp['feature'].isin(lw_cols)].head(20)
print(f'\n=== 长窗口特征被选入数 ===')
print(f'  TOP 20 中有 {len(imp.head(20)[imp.head(20)["feature"].isin(lw_cols)])} 个长窗口特征')
if len(imp_lw) > 0:
    print(imp_lw.to_string(index=False))

imp.to_csv(os.path.join(OUT_DIR, 'xgb_long_window_feature_importance.csv'), index=False)

# 保存最终模型
import os
os.makedirs(os.path.join(MODEL_DIR, 'final'), exist_ok=True)
joblib.dump({
    'pipeline': pipe_winner,
    'feature_names': list(X_winner.columns),
    'winner': best_name,
    'f1': best_f1,
}, os.path.join(MODEL_DIR, 'fraud_detection_xgb_long_window.pkl'))
print(f'\n  → models/fraud_detection_xgb_long_window.pkl')

# 全量预测
df_full = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_with_long_window.csv'))
X_full = df_full[X_winner.columns].copy()
p_ml = pipe_winner.predict_proba(X_full)[:, 1]
pd.DataFrame({
    'Symbol': df_full['Symbol'],
    'ShortName': df_full['ShortName'],
    'violation_year': df_full['violation_year'],
    'p_ml_longwin': p_ml,
}).to_csv(os.path.join(DATA_DIR, 'p_ml_longwin_full.csv'), index=False)
print(f'  → data/p_ml_longwin_full.csv')
print(f'  高概率(>0.5): {(p_ml > 0.5).sum()}')

print('\n✅ XGBoost + 长窗口训练完成')
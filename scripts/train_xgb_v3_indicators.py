#!/usr/bin/env python3
"""
扩展 A + C 训练: 用 fina_indicator(70+ 比率)+ pledge_stat 训练 XGBoost
=====================================================================
对比:
  A. v2.2 baseline(XGBoost + 7 比率,F1=0.8530)
  B. v2.3 = XGBoost + 7 + fina_indicator 拓展比率
  C. v2.3 + pledge_stat(完整扩展)
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
print('XGBoost v2.3 训练(fina_indicator + pledge_stat)')
print('=' * 60)

# 1. 加载主表
df = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_combined.csv'))
df['Symbol'] = df['Symbol'].astype(str).str.zfill(6)
df['violation_year'] = df['violation_year'].astype(int)

# 2. 合并 fina_indicator
print('\n[1] 合并 fina_indicator...')
if os.path.exists(os.path.join(DATA_DIR, 'fina_indicator_full.csv')):
    fina = pd.read_csv(os.path.join(DATA_DIR, 'fina_indicator_full.csv'))
    print(f'  fina_indicator_full: {fina.shape}')

    # Symbol 对齐:fina.ts_code → df.Symbol
    fina['ts_code'] = fina['ts_code'].astype(str)
    def symbol_from_tscode(tc):
        return tc.split('.')[0]
    fina['Symbol'] = fina['ts_code'].apply(symbol_from_tscode)

    # end_date 取年份
    fina['end_date'] = fina['end_date'].astype(str)
    fina['feature_year'] = fina['end_date'].str[:4].astype(int)

    # 去除与原 7 比率完全重复的列(保留原版,因为原版有 custom NaN 处理)
    fina_cols_drop = [c for c in ['roe', 'roa', 'debt_ratio', 'current_ratio']
                      if c in fina.columns]
    fina = fina.drop(columns=fina_cols_drop)

    # 合并
    df = df.merge(fina, on=['Symbol', 'feature_year'], how='left', suffixes=('', '_fina'))
    print(f'  合并后: {df.shape}')

    # 提取新比率列(原 7 中 asset_turnover/net_margin/ocf_to_rev 在 fina 中也叫同名,但被原版覆盖)
    fina_extra_cols = [c for c in fina.columns
                       if c not in ['Symbol', 'ts_code', 'end_date', 'feature_year', 'ann_date']
                       and c in df.columns]
    print(f'  新增 fina_indicator 比率: {len(fina_extra_cols)} 列')
    print(f'  示例: {fina_extra_cols[:10]}')
else:
    print('  ⚠️ fina_indicator_full.csv 不存在,跳过')
    fina_extra_cols = []

# 3. 合并 pledge_stat
print('\n[2] 合并 pledge_stat...')
if os.path.exists(os.path.join(DATA_DIR, 'pledge_stat_full.csv')):
    pled = pd.read_csv(os.path.join(DATA_DIR, 'pledge_stat_full.csv'))
    print(f'  pledge_stat_full: {pled.shape}')

    pled['ts_code'] = pled['ts_code'].astype(str)
    pled['Symbol'] = pled['ts_code'].apply(symbol_from_tscode)
    pled['end_date'] = pled['end_date'].astype(str)
    pled['feature_year'] = pled['end_date'].str[:4].astype(int)

    # 保留关键字段(避免 join 冲突)
    pled_keep = ['Symbol', 'feature_year', 'pledge_count', 'unrest_pledge',
                 'rest_pledge', 'total_share', 'pledge_ratio']
    pled_keep = [c for c in pled_keep if c in pled.columns]
    pled = pled[pled_keep].copy()

    # 去重(取每年最后一条)
    pled = pled.sort_values(['Symbol', 'feature_year']).drop_duplicates(
        subset=['Symbol', 'feature_year'], keep='last')

    df = df.merge(pled, on=['Symbol', 'feature_year'], how='left', suffixes=('', '_pled'))
    pled_extra_cols = [c for c in pled_keep if c not in ['Symbol', 'feature_year']]
    print(f'  新增 pledge 字段: {len(pled_extra_cols)} 列')
    print(f'  字段: {pled_extra_cols}')
else:
    print('  ⚠️ pledge_stat_full.csv 不存在,跳过')
    pled_extra_cols = []

# 4. 准备训练数据
df_clean = df.dropna(subset=['roe', 'ann_related']).copy()
df_clean = df_clean[df_clean['ann_related'].isin([0, 1])]
target = 'ann_fin_flag'
df_clean = df_clean.dropna(subset=[target])
df_clean = df_clean[df_clean[target].isin([0, 1])]
y = df_clean[target].astype(int)
print(f'\n样本: {len(df_clean)}, 正样本: {y.sum()} ({y.mean()*100:.1f}%)')

# 5. 三组特征
X_baseline = df_clean[ORIG_7].copy()
X_with_fina = df_clean[ORIG_7 + fina_extra_cols].copy() if fina_extra_cols else X_baseline.copy()
X_with_all = df_clean[ORIG_7 + fina_extra_cols + pled_extra_cols].copy() \
    if (fina_extra_cols and pled_extra_cols) else X_with_fina.copy()

print(f'  A. Baseline: {X_baseline.shape[1]} 列')
print(f'  B. + fina_indicator: {X_with_fina.shape[1]} 列')
print(f'  C. + fina + pledge: {X_with_all.shape[1]} 列')

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
print('v2.3 模型对比')
print('=' * 60)
comp = pd.DataFrame({k: {m: v[m].mean() for m in ['precision', 'recall', 'f1', 'roc_auc']}
                      for k, v in results.items()}).T
comp = comp[['precision', 'recall', 'f1', 'roc_auc']]
print(comp.round(4).to_string())
comp.to_csv(os.path.join(OUT_DIR, 'v23_model_comparison.csv'))
print(f'\n  → output/v23_model_comparison.csv')

# 8. 选最优
best_f1 = comp['f1'].max()
best_name = comp['f1'].idxmax()
print(f'\n胜出: {best_name} (F1={best_f1:.4f})')

# 9. 训练最终模型
X_winner = {'A_baseline': X_baseline,
            'B_with_fina': X_with_fina,
            'C_with_all': X_with_all}[best_name]
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

imp.to_csv(os.path.join(OUT_DIR, 'v23_feature_importance.csv'), index=False)

# 10. 保存新模型(覆盖 v2.2)
joblib.dump(pipe_winner, os.path.join(MODEL_DIR, 'fraud_detection_xgb_combined.pkl'))
print(f'\n  → models/fraud_detection_xgb_combined.pkl (覆盖 v2.2)')

# 同时存 metadata
metadata = {
    'pipeline_path': 'models/fraud_detection_xgb_combined.pkl',
    'model_type': 'XGBoost',
    'version': 'v2.3',
    'f1_cv': best_f1,
    'feature_names': list(X_winner.columns),
    'n_features': X_winner.shape[1],
}
joblib.dump(metadata, os.path.join(MODEL_DIR, 'fraud_detection_xgb_combined_metadata.pkl'))

# 11. 全量预测概率
p_ml = pipe_winner.predict_proba(X_winner)[:, 1]
df_full = df.copy()
df_full['p_ml_v23'] = np.nan
df_full.loc[df_clean.index, 'p_ml_v23'] = p_ml
out_p = os.path.join(DATA_DIR, 'p_ml_v23_full.csv')
df_full[['Symbol', 'ShortName', 'violation_year', 'p_ml_v23']].to_csv(out_p, index=False)
print(f'  → data/p_ml_v23_full.csv')

print('\n✅ v2.3 训练完成')
#!/usr/bin/env python3
"""
P3-6: XGBoost vs RF 对比 + Stacking 集成
==========================================
- 训练 XGBoost、LightGBM、RF 三个模型
- 对比性能 (CV F1)
- Stacking 集成 (Logistic Regression 作为 meta-learner)
- 替换/增强现有 RF 模型
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']


# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("P3-6.1: 加载数据")
print("=" * 60)

risk = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_industry_v2.csv"))
df_valid = risk.dropna(subset=FIN_COLS + ['ann_fin_flag'])
df_valid = df_valid[df_valid['ann_fin_flag'].isin([0, 1])].copy()
df_valid['ann_fin_flag'] = df_valid['ann_fin_flag'].astype(int)

X = df_valid[FIN_COLS]
y = df_valid['ann_fin_flag']
print(f"  训练样本: {len(X)}, 正例: {y.sum()}, 负例: {(y==0).sum()}")

# ============================================================
# 2. 训练 3 个模型
# ============================================================
print("\n" + "=" * 60)
print("P3-6.2: 训练 RF + XGBoost + LightGBM")
print("=" * 60)

models = {
    'Random Forest': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=8,
                                        min_samples_leaf=20,
                                        class_weight='balanced',
                                        random_state=42, n_jobs=-1)),
    ]),
    'XGBoost': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('clf', xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=(y==0).sum() / max((y==1).sum(), 1),  # 平衡
            random_state=42, n_jobs=-1, eval_metric='logloss',
            use_label_encoder=False,
        )),
    ]),
    'LightGBM': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('clf', lgb.LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.05,
            num_leaves=31, min_child_samples=20,
            class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1,
        )),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, pipe in models.items():
    print(f"\n  训练 {name}...")
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    cv_f1 = cross_val_score(pipe, X, y, cv=cv, scoring='f1', n_jobs=-1).mean()

    results[name] = {
        'pipe': pipe,
        'f1': f1, 'auc': auc, 'recall': recall, 'precision': precision, 'cv_f1': cv_f1,
    }
    print(f"    F1={f1:.3f} AUC={auc:.3f} Recall={recall:.3f} Precision={precision:.3f}")
    print(f"    CV F1={cv_f1:.3f}")

# ============================================================
# 3. Stacking 集成
# ============================================================
print("\n" + "=" * 60)
print("P3-6.3: Stacking 集成 (Logistic Regression meta-learner)")
print("=" * 60)

# 准备 stacking 训练数据
# 用 3 个模型的预测概率作为新特征
from sklearn.model_selection import KFold

stacking_X = np.zeros((len(X), 3))
for i, (name, model_info) in enumerate(results.items()):
    pipe = model_info['pipe']
    stacking_X[:, i] = pipe.predict_proba(X)[:, 1]

# Meta-learner
meta = LogisticRegression(random_state=42, max_iter=1000)
meta.fit(stacking_X, y)
meta_pred = meta.predict(stacking_X)
meta_proba = meta.predict_proba(stacking_X)[:, 1]
print(f"  Meta-learner 系数: {meta.coef_[0]}")
print(f"  Meta-learner 截距: {meta.intercept_[0]:.4f}")
print(f"  Stacking F1: {f1_score(y, meta_pred):.3f}")
print(f"  Stacking AUC: {roc_auc_score(y, meta_proba):.3f}")
print(f"  Stacking Recall: {recall_score(y, meta_pred):.3f}")

# 5-Fold CV 评估 stacking
from sklearn.model_selection import cross_val_predict
stacking_cv_proba = cross_val_predict(
    Pipeline([('meta', meta)]),  # 简化
    stacking_X, y, cv=5, method='predict_proba', n_jobs=-1
)[:, 1]
stacking_cv_f1 = cross_val_score(
    Pipeline([('meta', meta)]), stacking_X, y, cv=5, scoring='f1', n_jobs=-1
).mean()
print(f"  Stacking CV F1: {stacking_cv_f1:.3f}")

results['Stacking'] = {
    'pipe': None,
    'meta': meta,
    'models': {n: r['pipe'] for n, r in results.items()},
    'f1': f1_score(y, meta_pred),
    'auc': roc_auc_score(y, meta_proba),
    'recall': recall_score(y, meta_pred),
    'precision': precision_score(y, meta_pred),
    'cv_f1': stacking_cv_f1,
}

# ============================================================
# 4. 对比 + 选择最佳
# ============================================================
print("\n" + "=" * 60)
print("P3-6.4: 模型对比 + 选择最佳")
print("=" * 60)

print(f"  {'模型':<15} {'F1':>8} {'AUC':>8} {'CV F1':>8}")
print("  " + "-"*50)
for name, r in results.items():
    print(f"  {name:<15} {r['f1']:>8.3f} {r['auc']:>8.3f} {r['cv_f1']:>8.3f}")

# 选 CV F1 最高的
best_name = max(results, key=lambda k: results[k]['cv_f1'])
print(f"\n  最佳模型: {best_name} (CV F1 = {results[best_name]['cv_f1']:.3f})")

# ============================================================
# 5. 保存最佳模型 + 集成模型
# ============================================================
print("\n" + "=" * 60)
print("P3-6.5: 保存模型")
print("=" * 60)

# 保存集成模型
stacking_pipeline = {
    'rf': results['Random Forest']['pipe'],
    'xgb': results['XGBoost']['pipe'],
    'lgb': results['LightGBM']['pipe'],
    'meta': meta,
    'feature_names': FIN_COLS,
    'best_individual': best_name,
}
joblib.dump(stacking_pipeline, os.path.join(MODEL_DIR, "fraud_detection_stacking.pkl"))
print(f"  → models/fraud_detection_stacking.pkl (集成 4 个模型)")

# 单独保存 XGBoost(如果它最好)
if best_name == 'XGBoost' or 'XGBoost' in results:
    joblib.dump(results['XGBoost']['pipe'], os.path.join(MODEL_DIR, "fraud_detection_xgb.pkl"))
    print(f"  → models/fraud_detection_xgb.pkl")

if best_name == 'LightGBM' or 'LightGBM' in results:
    joblib.dump(results['LightGBM']['pipe'], os.path.join(MODEL_DIR, "fraud_detection_lgb.pkl"))
    print(f"  → models/fraud_detection_lgb.pkl")

# ============================================================
# 6. 总结
# ============================================================
print("\n" + "=" * 60)
print("✅ P3-6 完成: XGBoost / LightGBM / Stacking")
print("=" * 60)
print(f"  模型数量: 4 (RF + XGBoost + LightGBM + Stacking)")
print(f"  最佳模型: {best_name}")
print(f"  最佳 CV F1: {results[best_name]['cv_f1']:.3f}")

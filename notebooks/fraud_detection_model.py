#!/usr/bin/env python3
"""
阶段二：财务舞弊识别 ML 模型
============================
特征矩阵 → 清洗 → 训练 → 评估 → 特征重要性
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              precision_recall_curve, f1_score, recall_score, precision_score)
from sklearn.pipeline import Pipeline

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("1. 加载特征矩阵")
print("=" * 60)

df = pd.read_csv(f"{BASE}/data/fraud_features_combined.csv")
print(f"总记录: {len(df)}")


# ============================================================
# 2. 数据清洗 & 特征工程
# ============================================================
print("\n" + "=" * 60)
print("2. 数据清洗")
print("=" * 60)

fin_cols = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']

# 只保留有财务数据且标签有效的行
df_clean = df.dropna(subset=['roe', 'ann_related']).copy()
df_clean = df_clean[df_clean['ann_related'].isin([0, 1])]
# 预测目标：ann_fin_flag（是否影响财务信息）比 ann_related 更适合用财务指标预测
target = 'ann_fin_flag'
df_clean = df_clean.dropna(subset=[target])
df_clean = df_clean[df_clean[target].isin([0, 1])]
y = df_clean[target].astype(int)
X = df_clean[fin_cols].copy()

print(f"可用样本: {len(X)}")
print(f"{target}=1: {y.sum()} ({y.sum()/len(y)*100:.0f}%)")

# Winsorize 极端值（1% 和 99% 分位数）
for col in fin_cols:
    lo = X[col].quantile(0.01)
    hi = X[col].quantile(0.99)
    n_clipped = ((X[col] < lo) | (X[col] > hi)).sum()
    X[col] = X[col].clip(lo, hi)
    if n_clipped > 0:
        print(f"  {col}: 裁剪 {n_clipped} 个极端值 → [{lo:.3f}, {hi:.3f}]")

print(f"\n特征矩阵: {X.shape}")
print(f"标签分布: 0={len(y[y==0])}, 1={len(y[y==1])}")

# ============================================================
# 3. 训练模型
# ============================================================
print("\n" + "=" * 60)
print("3. 训练模型")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# 构建 Pipeline
models = {
    "Logistic Regression": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=8,
                                        min_samples_leaf=20, class_weight='balanced',
                                        random_state=42, n_jobs=-1)),
    ]),
    "Gradient Boosting": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                            learning_rate=0.05, random_state=42)),
    ]),
}

results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        'pipe': pipe,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': (y_pred == y_test).mean(),
        'f1': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
    }
    print(f"  {name}: Acc={results[name]['accuracy']:.3f}, F1={results[name]['f1']:.3f}, "
          f"Recall={results[name]['recall']:.3f}, AUC={results[name]['auc']:.3f}")

# ============================================================
# 4. 详细评估
# ============================================================
print("\n" + "=" * 60)
print("4. 详细评估 — Random Forest")
print("=" * 60)

best_model = models["Random Forest"]
y_pred = results["Random Forest"]["y_pred"]
print(classification_report(y_test, y_pred, target_names=['非财务信息影响', '财务信息影响']))

cm = confusion_matrix(y_test, y_pred)
print(f"混淆矩阵:\n  TN={cm[0][0]}, FP={cm[0][1]}\n  FN={cm[1][0]}, TP={cm[1][1]}")

# ============================================================
# 5. 特征重要性
# ============================================================
print("\n" + "=" * 60)
print("5. 特征重要性分析")
print("=" * 60)

rf = best_model.named_steps['clf']
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("特征重要性排名:")
for i in range(len(fin_cols)):
    print(f"  {i+1}. {fin_cols[indices[i]]}: {importances[indices[i]]:.4f}")

# ============================================================
# 6. 可视化
# ============================================================
print("\n" + "=" * 60)
print("6. 生成可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 6a. 特征重要性
ax = axes[0, 0]
ax.barh([fin_cols[i] for i in indices][::-1], importances[indices][::-1], color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('Random Forest 特征重要性', fontweight='bold')

# 6b. 模型对比
ax = axes[0, 1]
metrics_names = ['Accuracy', 'F1', 'Recall', 'Precision', 'AUC']
x = np.arange(len(metrics_names))
width = 0.25
for j, (name, res) in enumerate(results.items()):
    vals = [res['accuracy'], res['f1'], res['recall'], res['precision'], res['auc']]
    ax.bar(x + j * width, vals, width, label=name, alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.set_title('模型性能对比', fontweight='bold')

# 6c. PR曲线
ax = axes[1, 0]
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(y_test, res['y_proba'])
    ax.plot(recall, precision, label=f"{name} (AUC={res['auc']:.3f})")
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall 曲线', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 6d. 混淆矩阵热力图
ax = axes[1, 1]
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['非年报相关', '年报相关'])
ax.set_yticklabels(['非年报相关', '年报相关'])
ax.set_xlabel('预测')
ax.set_ylabel('实际')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i][j], ha='center', va='center', fontsize=16, fontweight='bold')
ax.set_title(f'混淆矩阵 (Random Forest)\nF1={results["Random Forest"]["f1"]:.3f}', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{BASE}/output/model_evaluation.png", dpi=150, bbox_inches='tight')
print(f"  → {BASE}/output/model_evaluation.png")

# ============================================================
# 7. 交叉验证
# ============================================================
print("\n" + "=" * 60)
print("7. 交叉验证 (Random Forest")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='f1')
print(f"5-Fold CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# ============================================================
# 8. 保存模型
# ============================================================
import joblib
joblib.dump(best_model, f"{BASE}/models/fraud_detection_rf_combined.pkl")
print(f"\n  → 模型已保存: {BASE}/models/fraud_detection_rf_combined.pkl")

print("\n" + "=" * 60)
print("🏁 建模完成")
print("=" * 60)
print(f"最佳模型: Random Forest")
print(f"  F1 Score:  {results['Random Forest']['f1']:.3f}")
print(f"  Recall:    {results['Random Forest']['recall']:.3f}")
print(f"  Precision: {results['Random Forest']['precision']:.3f}")
print(f"  AUC:       {results['Random Forest']['auc']:.3f}")
print(f"  CV F1:     {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
print(f"Top 3 特征: {', '.join([fin_cols[i] for i in indices[:3]])}")

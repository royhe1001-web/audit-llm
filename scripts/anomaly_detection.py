#!/usr/bin/env python3
"""
阶段三·步骤 5: 异常检测 (无监督)
============================
Isolation Forest + LOF 双算法投票,识别财务特征异常公司
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("步骤 5.1: 加载评分数据")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_full.csv"))
print(f"总记录: {len(df)}")

# ============================================================
# 2. 预处理
# ============================================================
print("\n" + "=" * 60)
print("步骤 5.2: 预处理 (中位数填补 + 标准化)")
print("=" * 60)

imp = SimpleImputer(strategy='median')
X = imp.fit_transform(df[FIN_COLS])
print(f"  缺失填补: {(df[FIN_COLS].isna().sum().sum())} 个缺失值 → 0")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  标准化: mean={X_scaled.mean():.3f}, std={X_scaled.std():.3f}")

# ============================================================
# 3. Isolation Forest
# ============================================================
print("\n" + "=" * 60)
print("步骤 5.3: Isolation Forest")
print("=" * 60)

iso = IsolationForest(n_estimators=200, contamination=0.05,
                       random_state=42, n_jobs=-1)
iso_pred = iso.fit_predict(X_scaled)
df['if_anomaly'] = (iso_pred == -1).astype(int)
n_if = df['if_anomaly'].sum()
print(f"  IF 标记异常: {n_if} ({n_if/len(df)*100:.1f}%)")

# ============================================================
# 4. LOF
# ============================================================
print("\n" + "=" * 60)
print("步骤 5.4: Local Outlier Factor")
print("=" * 60)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, n_jobs=-1)
lof_pred = lof.fit_predict(X_scaled)
df['lof_anomaly'] = (lof_pred == -1).astype(int)
n_lof = df['lof_anomaly'].sum()
print(f"  LOF 标记异常: {n_lof} ({n_lof/len(df)*100:.1f}%)")

# ============================================================
# 5. 合并 (任一标记)
# ============================================================
print("\n" + "=" * 60)
print("步骤 5.5: 合并 (任一标记为异常)")
print("=" * 60)

df['is_anomaly'] = ((df['if_anomaly'] == 1) | (df['lof_anomaly'] == 1)).astype(int)

def anomaly_method(r):
    if r['if_anomaly'] == 1 and r['lof_anomaly'] == 1:
        return 'IF+LOF'
    if r['if_anomaly'] == 1:
        return 'IF_only'
    if r['lof_anomaly'] == 1:
        return 'LOF_only'
    return 'None'

df['anomaly_method'] = df.apply(anomaly_method, axis=1)

print(f"  合并后异常总数: {df['is_anomaly'].sum()} ({df['is_anomaly'].sum()/len(df)*100:.1f}%)")
print(f"  IF+LOF 双标记: {((df['if_anomaly']==1) & (df['lof_anomaly']==1)).sum()}")
print(f"  仅 IF: {((df['if_anomaly']==1) & (df['lof_anomaly']==0)).sum()}")
print(f"  仅 LOF: {((df['if_anomaly']==0) & (df['lof_anomaly']==1)).sum()}")

# ============================================================
# 6. 命中率验证
# ============================================================
print("\n" + "=" * 60)
print("步骤 5.6: 命中率验证")
print("=" * 60)

actual_fraud_mask = df['ann_fin_flag'] == 1
actual_fraud_n = actual_fraud_mask.sum()
hit = df.loc[actual_fraud_mask, 'is_anomaly'].sum()
hit_rate = hit / actual_fraud_n * 100
baseline = df['is_anomaly'].mean() * 100
print(f"  已知违规 (ann_fin_flag=1): {actual_fraud_n}")
print(f"  其中被标记为异常: {hit} ({hit_rate:.1f}%)")
print(f"  基准异常率: {baseline:.1f}%")
print(f"  富集比: {hit_rate/baseline:.2f}x (>1 即优于随机)")

# ============================================================
# 7. 异常公司清单
# ============================================================
print("\n" + "=" * 60)
print("步骤 5.7: 异常公司清单")
print("=" * 60)

anomalies = df[df['is_anomaly'] == 1].copy()
anomalies.sort_values('risk_score', ascending=False, inplace=True)

# 去重: 同一公司只保留最高风险分
anom_dedup = anomalies.drop_duplicates(subset=['Symbol']).head(200)
anom_dedup.to_csv(os.path.join(DATA_DIR, "anomaly_companies.csv"), index=False)
print(f"  → data/anomaly_companies.csv ({len(anom_dedup)} 个唯一公司)")

# Top10 异常公司
print(f"\n  Top 10 异常公司 (按风险分):")
print(anom_dedup[['ShortName', 'industry', 'violation_year', 'risk_score',
                   'p_ml', 'rule_ids', 'anomaly_method']]
      .head(10).to_string(index=False))

# ============================================================
# 8. PCA 2D 投影可视化
# ============================================================
print("\n" + "=" * 60)
print("步骤 5.8: PCA 2D 投影可视化")
print("=" * 60)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
print(f"  PC1 解释方差: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PC2 解释方差: {pca.explained_variance_ratio_[1]*100:.1f}%")

fig, ax = plt.subplots(figsize=(12, 8))

# 正常点(采样,避免过密)
normal_mask = df['is_anomaly'] == 0
n_normal = normal_mask.sum()
sample_n = min(2000, n_normal)
normal_sample = df[normal_mask].sample(n=sample_n, random_state=42)
sample_idx = normal_sample.index
ax.scatter(coords[sample_idx, 0], coords[sample_idx, 1],
           c='steelblue', alpha=0.25, s=12, label='正常 (采样)')

# IF only
if_only = df[(df['if_anomaly'] == 1) & (df['lof_anomaly'] == 0)]
if len(if_only) > 0:
    if_idx = if_only.index
    ax.scatter(coords[if_idx, 0], coords[if_idx, 1],
               c='orange', alpha=0.7, s=25, label=f'IF only ({len(if_only)})', edgecolor='white')

# LOF only
lof_only = df[(df['if_anomaly'] == 0) & (df['lof_anomaly'] == 1)]
if len(lof_only) > 0:
    lof_idx = lof_only.index
    ax.scatter(coords[lof_idx, 0], coords[lof_idx, 1],
               c='green', alpha=0.7, s=25, label=f'LOF only ({len(lof_only)})', edgecolor='white')

# IF+LOF 双标记 (最高风险)
both = df[(df['if_anomaly'] == 1) & (df['lof_anomaly'] == 1)]
if len(both) > 0:
    both_idx = both.index
    ax.scatter(coords[both_idx, 0], coords[both_idx, 1],
               c='red', alpha=0.85, s=35, label=f'IF+LOF ({len(both)})', edgecolor='black', linewidth=0.5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title('异常检测 (PCA 2D 投影)\nIsolation Forest + LOF 投票', fontweight='bold', fontsize=14)
ax.legend(fontsize=11, loc='best')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "anomaly_detection_scatter.png"), dpi=150, bbox_inches='tight')
print(f"  → output/anomaly_detection_scatter.png")

# ============================================================
# 9. 异常公司行业分布
# ============================================================
print("\n" + "=" * 60)
print("步骤 5.9: 异常公司行业分布")
print("=" * 60)

anom_ind = anomalies['industry'].value_counts().head(10)
print(f"  Top 10 异常行业:")
print(anom_ind)

# ============================================================
# 10. 总结
# ============================================================
print("\n" + "=" * 60)
print("✅ 步骤 5 完成")
print("=" * 60)
print(f"  IF 标记: {n_if} ({n_if/len(df)*100:.1f}%)")
print(f"  LOF 标记: {n_lof} ({n_lof/len(df)*100:.1f}%)")
print(f"  合并异常: {df['is_anomaly'].sum()} ({df['is_anomaly'].sum()/len(df)*100:.1f}%)")
print(f"  ann_fin_flag=1 命中率: {hit_rate:.1f}% (基准 {baseline:.1f}%, 富集 {hit_rate/baseline:.2f}x)")
print(f"  产出: data/anomaly_companies.csv")
print(f"        output/anomaly_detection_scatter.png")

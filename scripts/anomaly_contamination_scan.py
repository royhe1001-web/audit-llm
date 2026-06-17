#!/usr/bin/env python3
"""
阶段四·优化 4: 异常检测 contamination 自适应扫描
=================================================
扫描 [0.03, 0.05, 0.08, 0.10, 0.15] 五个污染率,
对比 IsolationForest 在已知违规公司(ann_fin_flag=1)上的命中率,
选出最优 contamination。
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']

print('=' * 60)
print('异常检测 contamination 自适应扫描')
print('=' * 60)

# 加载
df = pd.read_csv(os.path.join(DATA_DIR, 'risk_scored_full.csv'))
print(f'样本: {len(df)}, ann_fin_flag=1: {(df["ann_fin_flag"] == 1).sum()}')

# 预处理
imp = SimpleImputer(strategy='median')
X = imp.fit_transform(df[FIN_COLS])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 扫描
contaminations = [0.03, 0.05, 0.08, 0.10, 0.15]
results = []

fraud_mask = (df['ann_fin_flag'] == 1).values

for c in contaminations:
    iso = IsolationForest(n_estimators=200, contamination=c, random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    pred = iso.predict(X_scaled)
    # -1 表示异常
    anomaly_pred = (pred == -1)
    n_anomaly = anomaly_pred.sum()

    # 在已知违规公司上的命中率
    if fraud_mask.sum() > 0:
        hit = (anomaly_pred & fraud_mask).sum()
        hit_rate = hit / fraud_mask.sum() * 100
    else:
        hit_rate = 0
    # 在已知正常公司上的命中率(误报率)
    nonfraud_mask = (df['ann_fin_flag'] == 0).values
    if nonfraud_mask.sum() > 0:
        fp = (anomaly_pred & nonfraud_mask).sum()
        fp_rate = fp / nonfraud_mask.sum() * 100
    else:
        fp_rate = 0

    results.append({
        'contamination': c,
        'n_anomaly': n_anomaly,
        'hit_fraud': hit if fraud_mask.sum() > 0 else 0,
        'hit_rate_fraud_pct': hit_rate,
        'fp_nonfraud': fp if nonfraud_mask.sum() > 0 else 0,
        'fp_rate_nonfraud_pct': fp_rate,
    })
    print(f'  contamination={c}: 异常={n_anomaly}, 命中违规={hit}, 命中率={hit_rate:.2f}%, 误报率={fp_rate:.2f}%')

# 对比表
df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(OUT_DIR, 'anomaly_contamination_scan.csv'), index=False)
print(f'\n  → output/anomaly_contamination_scan.csv')

# 选择最优:命中率最高且误报率合理
# 优先级:命中率优先,误报率 < 5% 为约束
best_idx = None
for i, row in df_res.iterrows():
    if row['fp_rate_nonfraud_pct'] < 10:
        if best_idx is None or row['hit_rate_fraud_pct'] > df_res.loc[best_idx, 'hit_rate_fraud_pct']:
            best_idx = i

if best_idx is None:
    best_idx = df_res['hit_rate_fraud_pct'].idxmax()

best = df_res.loc[best_idx]
print(f'\n  最优 contamination: {best["contamination"]}')
print(f'  命中率: {best["hit_rate_fraud_pct"]:.2f}%, 误报率: {best["fp_rate_nonfraud_pct"]:.2f}%')

# 可视化
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_res['contamination'], df_res['hit_rate_fraud_pct'], 'o-', color='red', label='违规命中率(越高越好)', linewidth=2)
ax.plot(df_res['contamination'], df_res['fp_rate_nonfraud_pct'], 's-', color='blue', label='正常公司误报率(越低越好)', linewidth=2)
ax.axvline(best['contamination'], color='green', linestyle='--', linewidth=2,
           label=f'最优={best["contamination"]}')
ax.set_xlabel('contamination 参数', fontsize=12)
ax.set_ylabel('百分比 (%)', fontsize=12)
ax.set_title('Isolation Forest contamination 自适应扫描', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'anomaly_contamination_scan.png'), dpi=150, bbox_inches='tight')
print(f'  → output/anomaly_contamination_scan.png')
plt.close()

# 用最优 contamination 重跑异常检测,生成最终异常公司列表
print('\n' + '=' * 60)
print(f'用最优 contamination={best["contamination"]} 重跑')
print('=' * 60)

iso_best = IsolationForest(n_estimators=200, contamination=best['contamination'], random_state=42, n_jobs=-1)
iso_best.fit(X_scaled)
df['if_anomaly_v2'] = (iso_best.predict(X_scaled) == -1).astype(int)
n_v2 = df['if_anomaly_v2'].sum()
print(f'  异常数: {n_v2} ({n_v2/len(df)*100:.2f}%)')

# 保存更新后的异常公司列表
anomaly_v2 = df[df['if_anomaly_v2'] == 1].copy()
anomaly_v2.to_csv(os.path.join(DATA_DIR, 'anomaly_companies_v2.csv'), index=False)
print(f'  → data/anomaly_companies_v2.csv ({len(anomaly_v2)} 家)')
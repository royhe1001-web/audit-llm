#!/usr/bin/env python3
"""
P3-2: 数据质量清洗
===================
1. 去重 (Symbol, feature_year, ann_fin_flag 三元组)
2. 异常值 winsorize (1%/99%)
3. 缺失率作为新特征
4. 极端值标记
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']


def assign_level(s):
    if s >= 0.55: return '高风险'
    if s >= 0.25: return '中风险'
    return '低风险'


# ============================================================
# 1. 加载最新评分
# ============================================================
print("=" * 60)
print("P3-2.1: 加载最新评分")
print("=" * 60)

risk = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_temporal_v2.csv"))
print(f"  原始: {risk.shape}")

# ============================================================
# 2. 去重策略
# ============================================================
print("\n" + "=" * 60)
print("P3-2.2: 去重策略")
print("=" * 60)

# 2.1 按 (Symbol, feature_year) 去重,保留 ann_fin_flag=1 优先
print(f"  去重前: {len(risk)}")
risk = risk.sort_values(['Symbol', 'feature_year', 'ann_fin_flag'],
                          ascending=[True, True, False])  # ann_fin_flag=1 优先
risk = risk.drop_duplicates(subset=['Symbol', 'feature_year'], keep='first')
print(f"  去重后: {len(risk)}")
print(f"  减少: {len(risk)} 行重复")

# 2.2 ann_fin_flag 冲突检查
# 同一 (Symbol, year) 可能在标注数据中既有 ann_fin_flag=1 又有 0
labeled = pd.read_excel(os.path.join(DATA_DIR, "STK_labeled_combined_G01-G06.xlsx"))
labeled['Symbol'] = labeled['Symbol'].astype(str).str.zfill(6)
labeled['ViolationYear'] = pd.to_numeric(labeled['ViolationYear'], errors='coerce')
labeled_clean = labeled.dropna(subset=['ViolationYear']).copy()
labeled_clean['ViolationYear'] = labeled_clean['ViolationYear'].astype(int)

# 同一 (Symbol, year) 多个 ann_fin_flag → 取众数
conflicts = labeled_clean.groupby(['Symbol', 'ViolationYear'])['ann_fin_flag'].nunique()
n_conflict = (conflicts > 1).sum()
print(f"  标注冲突(Symbol,year 有多个 ann_fin_flag): {n_conflict} 个 (公司,年)")

# ============================================================
# 3. 异常值 winsorize
# ============================================================
print("\n" + "=" * 60)
print("P3-2.3: 异常值 winsorize (1%/99%)")
print("=" * 60)

# 之前 fraud_detection_model.py 已经 winsorize 一次,这里再做更激进的处理
risk_w = risk.copy()
for col in FIN_COLS:
    s = risk_w[col].dropna()
    if len(s) < 10:
        continue
    p01 = s.quantile(0.01)
    p99 = s.quantile(0.99)
    n_clipped = ((risk_w[col] < p01) | (risk_w[col] > p99)).sum()
    risk_w[col] = risk_w[col].clip(p01, p99)
    if n_clipped > 0:
        print(f"  {col}: 裁剪 {n_clipped} 个极端值 → [{p01:.4f}, {p99:.4f}]")

# 添加 winsorize 标记列
for col in FIN_COLS:
    s = risk[col].dropna()
    if len(s) < 10:
        continue
    p01 = s.quantile(0.01)
    p99 = s.quantile(0.99)
    risk[f'{col}_is_extreme'] = ((risk[col] < p01) | (risk[col] > p99)).fillna(False).astype(int)
    risk[f'{col}_is_outlier'] = ((risk[col] < p01 * 2) | (risk[col] > p99 * 2)).fillna(False).astype(int)

extreme_cols = [f'{c}_is_extreme' for c in FIN_COLS]
outlier_cols = [f'{c}_is_outlier' for c in FIN_COLS]
risk['n_extreme_features'] = risk[extreme_cols].sum(axis=1)
risk['n_outlier_features'] = risk[outlier_cols].sum(axis=1)
print(f"  新增列: n_extreme_features, n_outlier_features")

# ============================================================
# 4. 缺失率作为新特征
# ============================================================
print("\n" + "=" * 60)
print("P3-2.4: 缺失率特征")
print("=" * 60)

risk['n_missing'] = risk[FIN_COLS].isna().sum(axis=1)
risk['missing_rate'] = risk['n_missing'] / len(FIN_COLS)
print(f"  缺失率分布:")
print(risk['missing_rate'].value_counts().sort_index().head())

# ============================================================
# 5. 应用硬规则
# ============================================================
print("\n" + "=" * 60)
print("P3-2.5: 数据质量硬规则")
print("=" * 60)

risk['risk_score_v5'] = risk['risk_score_v4'].copy()
adjustments = []

# R29: ≥3 个特征是极端值 (top 1% 之外)
mask = (risk['n_extreme_features'] >= 3)
n = mask.sum()
risk.loc[mask, 'risk_score_v5'] = risk.loc[mask, 'risk_score_v5'].clip(lower=0.5)
adjustments.append(('R29 ≥3 个极端值特征', n, 0.5))

# R30: ≥2 个特征是异常值 (top 0.5% 之外) → 强信号
mask = (risk['n_outlier_features'] >= 2)
n = mask.sum()
risk.loc[mask, 'risk_score_v5'] = risk.loc[mask, 'risk_score_v5'].clip(lower=0.55)
adjustments.append(('R30 ≥2 个异常值特征', n, 0.55))

# R31: 关键比率缺失(net_margin 缺失)→ 数据不完整,需复核
mask = (risk['net_margin'].isna())
n = mask.sum()
risk.loc[mask, 'risk_score_v5'] = risk.loc[mask, 'risk_score_v5'].clip(lower=0.4)
adjustments.append(('R31 净利率缺失(数据不全)', n, 0.4))

# 重新分配等级
risk['risk_level_v5'] = risk['risk_score_v5'].apply(assign_level)

for name, n, thresh in adjustments:
    print(f"  {name}: 触发 {n} 条, 风险分下限 {thresh}")

# ============================================================
# 6. 验证
# ============================================================
print("\n" + "=" * 60)
print("P3-2.6: 验证")
print("=" * 60)

actual_fraud = risk['ann_fin_flag'] == 1
for col, label in [('risk_level', 'v0'), ('risk_level_v3', 'v3'),
                    ('risk_level_v4', 'v4'), ('risk_level_v5', 'v5')]:
    hit = (risk.loc[actual_fraud, col] == '高风险').sum() / actual_fraud.sum() * 100
    print(f"  {label}: 高风险命中率 = {hit:.1f}%")

# ============================================================
# 7. 保存
# ============================================================
print("\n" + "=" * 60)
print("P3-2.7: 保存")
print("=" * 60)

risk.to_csv(os.path.join(DATA_DIR, "risk_scored_quality.csv"), index=False)
print(f"  → data/risk_scored_quality.csv ({os.path.getsize(os.path.join(DATA_DIR, 'risk_scored_quality.csv'))/1024:.0f} KB)")

print("\n" + "=" * 60)
print("✅ P3-2 完成: 数据质量清洗")
print("=" * 60)

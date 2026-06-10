#!/usr/bin/env python3
"""
P3-1: 多年时序特征 (CAGR + 波动率)
=====================================
基于 raw_financials.csv 计算:
- 收入 3 年 CAGR
- 净利润 3 年 CAGR
- 总资产 3 年 CAGR
- 收入波动率
- 净利润波动率
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")


def assign_level(s):
    if s >= 0.55: return '高风险'
    if s >= 0.25: return '中风险'
    return '低风险'


# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("P3-1.1: 加载原始财务数据")
print("=" * 60)

raw = pd.read_csv(os.path.join(DATA_DIR, "raw_financials.csv"))
raw['Symbol'] = raw['fetch_symbol'].astype(str).str.zfill(6)
raw = raw.dropna(subset=['Symbol', 'fetch_year'])
raw['fetch_year'] = raw['fetch_year'].astype(int)
print(f"  raw_financials: {raw.shape}")
print(f"  公司数: {raw['Symbol'].nunique()}")
print(f"  年份范围: {raw['fetch_year'].min()}-{raw['fetch_year'].max()}")

# 去重: 同一 (Symbol, year) 只保留一条
raw = raw.drop_duplicates(subset=['Symbol', 'fetch_year'], keep='first')
print(f"  去重后: {raw.shape}")

# ============================================================
# 2. 计算多年时序特征
# ============================================================
print("\n" + "=" * 60)
print("P3-1.2: 计算 CAGR 和波动率")
print("=" * 60)

# 按公司、年份排序
raw = raw.sort_values(['Symbol', 'fetch_year']).reset_index(drop=True)

# 对每家公司,计算 3 年 CAGR
def calc_features(grp):
    grp = grp.sort_values('fetch_year')
    if len(grp) < 2:
        return pd.Series({
            'revenue_cagr_3y': np.nan,
            'np_cagr_3y': np.nan,
            'asset_cagr_3y': np.nan,
            'revenue_volatility': np.nan,
            'np_volatility': np.nan,
            'n_years_data': len(grp),
        })

    n_years = len(grp)
    revenue = grp['revenue'].values if 'revenue' in grp else None
    net_profit_arr = grp['n_income'].values if 'n_income' in grp else None
    assets = grp['total_assets'].values if 'total_assets' in grp else None

    # 3 年 CAGR: 末值/初值 ^ (1/n) - 1
    def cagr(arr, n=3):
        if len(arr) < n + 1:
            return np.nan
        first = arr[-(n+1)] if len(arr) > n else arr[0]
        last = arr[-1]
        if first is None or last is None or first <= 0 or last is None:
            return np.nan
        if first < 0 and last > 0:  # 转亏为盈
            return 1.0  # 标记
        if first > 0 and last < 0:  # 转盈为亏
            return -1.0  # 标记
        try:
            return (last / first) ** (1.0 / n) - 1
        except:
            return np.nan

    # 波动率: std / mean (变异系数)
    def volatility(arr):
        if len(arr) < 2:
            return np.nan
        s = pd.Series(arr).dropna()
        if len(s) < 2 or s.mean() == 0:
            return np.nan
        # 只对正值计算(净利润可能为负)
        pos = s[s > 0]
        if len(pos) >= 2:
            return pos.std() / pos.mean()
        return np.nan

    return pd.Series({
        'revenue_cagr_3y': cagr(revenue, 3) if revenue is not None else np.nan,
        'np_cagr_3y': cagr(net_profit_arr, 3) if net_profit_arr is not None else np.nan,
        'asset_cagr_3y': cagr(assets, 3) if assets is not None else np.nan,
        'revenue_volatility': volatility(revenue) if revenue is not None else np.nan,
        'np_volatility': volatility(net_profit_arr) if net_profit_arr is not None else np.nan,
        'n_years_data': n_years,
    })

temporal_features = raw.groupby('Symbol').apply(calc_features, include_groups=False).reset_index()
print(f"  多年时序特征: {temporal_features.shape}")
print(f"  有效样本(>=3年): {temporal_features['n_years_data'].ge(3).sum()}")
print(f"  覆盖率: {temporal_features['n_years_data'].ge(3).sum() / len(temporal_features) * 100:.1f}%")

# ============================================================
# 3. 合并到风险评分
# ============================================================
print("\n" + "=" * 60)
print("P3-1.3: 合并到风险评分")
print("=" * 60)

risk = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_industry.csv"))
risk['Symbol'] = risk['Symbol'].astype(str).str.zfill(6)
print(f"  风险评分: {risk.shape}")

# 对每个 (Symbol, violation_year), 找该公司的多年时序特征
# 直接合并 - 因为风险评分有 6,167 行(去重后),每行是 (Symbol, year)
# temporal_features 也有 (Symbol), 同一公司所有行共享
# 需要为每个 (Symbol) 找该公司在 violation_year 前 3 年的数据

# 简单方案:每家公司一个多年时序特征,然后 merge
merged = risk.merge(temporal_features, on='Symbol', how='left')
print(f"  合并后: {merged.shape}")

# ============================================================
# 4. 应用硬规则
# ============================================================
print("\n" + "=" * 60)
print("P3-1.4: 应用多年时序硬规则")
print("=" * 60)

# 风险分 = max(原风险分, 时序信号下限)
merged['risk_score_v4'] = merged['risk_score_v3'].copy()
adjustments = []

# R23: 净利润 3 年 CAGR < -30% (持续恶化)
mask = (merged['np_cagr_3y'] < -0.3)
n = mask.sum()
merged.loc[mask, 'risk_score_v4'] = merged.loc[mask, 'risk_score_v4'].clip(lower=0.65)
adjustments.append(('R23 净利润 3 年 CAGR<-30%', n, 0.65))

# R24: 收入 3 年 CAGR < -20% (业务萎缩)
mask = (merged['revenue_cagr_3y'] < -0.2)
n = mask.sum()
merged.loc[mask, 'risk_score_v4'] = merged.loc[mask, 'risk_score_v4'].clip(lower=0.6)
adjustments.append(('R24 收入 3 年 CAGR<-20%', n, 0.6))

# R25: 净利润波动率 > 100% (业绩大起大落)
mask = (merged['np_volatility'] > 1.0)
n = mask.sum()
merged.loc[mask, 'risk_score_v4'] = merged.loc[mask, 'risk_score_v4'].clip(lower=0.55)
adjustments.append(('R25 净利润波动率>100%', n, 0.55))

# R26: 总资产 3 年 CAGR > 100% (并购/虚增,异常扩张)
mask = (merged['asset_cagr_3y'] > 1.0)
n = mask.sum()
merged.loc[mask, 'risk_score_v4'] = merged.loc[mask, 'risk_score_v4'].clip(lower=0.5)
adjustments.append(('R26 总资产 3 年 CAGR>100%', n, 0.5))

# R27: 转盈为亏(从盈利到亏损)
mask = (merged['np_cagr_3y'] == -1.0)
n = mask.sum()
merged.loc[mask, 'risk_score_v4'] = merged.loc[mask, 'risk_score_v4'].clip(lower=0.65)
adjustments.append(('R27 转盈为亏', n, 0.65))

# R28: 转亏为盈(可能是反转或一次性收益,需核查)
mask = (merged['np_cagr_3y'] == 1.0)
n = mask.sum()
merged.loc[mask, 'risk_score_v4'] = merged.loc[mask, 'risk_score_v4'].clip(lower=0.45)
adjustments.append(('R28 转亏为盈', n, 0.45))

for name, n, thresh in adjustments:
    print(f"  {name}: 触发 {n} 条, 风险分下限 {thresh}")

# 重新分配等级
merged['risk_level_v4'] = merged['risk_score_v4'].apply(assign_level)

# ============================================================
# 5. 验证
# ============================================================
print("\n" + "=" * 60)
print("P3-1.5: 验证 — 已知违规命中率")
print("=" * 60)

actual_fraud = merged['ann_fin_flag'] == 1
for col, label in [('risk_level', 'v0'), ('risk_level_v2', 'v2'),
                    ('risk_level_v3', 'v3'), ('risk_level_v4', 'v4')]:
    hit = (merged.loc[actual_fraud, col] == '高风险').sum() / actual_fraud.sum() * 100
    print(f"  {label}: 高风险命中率 = {hit:.1f}%")

# 分布对比
print("\n" + "=" * 60)
print("P3-1.6: 风险分布对比")
print("=" * 60)
for col, label in [('risk_level_v3', 'v3'), ('risk_level_v4', 'v4')]:
    counts = merged[col].value_counts()
    print(f"  {label}: {dict(counts)}")

# ============================================================
# 6. 保存
# ============================================================
print("\n" + "=" * 60)
print("P3-1.7: 保存")
print("=" * 60)

save_cols = list(merged.columns)
merged.to_csv(os.path.join(DATA_DIR, "risk_scored_temporal_v2.csv"), index=False)
print(f"  → data/risk_scored_temporal_v2.csv ({os.path.getsize(os.path.join(DATA_DIR, 'risk_scored_temporal_v2.csv'))/1024:.0f} KB)")

# 单独保存多年时序特征(可复用到其他数据集)
temporal_features.to_csv(os.path.join(DATA_DIR, "temporal_features_master.csv"), index=False)
print(f"  → data/temporal_features_master.csv ({os.path.getsize(os.path.join(DATA_DIR, 'temporal_features_master.csv'))/1024:.0f} KB)")

print("\n" + "=" * 60)
print("✅ P3-1 完成: 多年时序 CAGR + 波动率")
print("=" * 60)

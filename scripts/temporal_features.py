#!/usr/bin/env python3
"""
P0-2: 时序特征工程
===================
对每家公司,基于历史财务快照计算:
- 净利润 YoY 变化
- 营收 YoY 变化
- 连续亏损(2+ 年净利润为负)
- 现金流恶化(OCF 同比下降)
- 资产负债率恶化
- 3 年累计下滑标识
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")


def assign_level(s):
    if s >= 0.55: return '高风险'
    if s >= 0.25: return '中风险'
    return '低风险'


# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("P0-2.1: 加载数据")
print("=" * 60)

# 加载治理信号增强后的评分数据(risk 本身已包含 7 个财务特征)
risk = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_governance.csv"))
risk['Symbol'] = risk['Symbol'].astype(str)
risk['violation_year'] = risk['violation_year'].astype(int)
risk['feature_year'] = risk['feature_year'].astype(int)
print(f"  风险评分(含治理信号 + 7 特征): {risk.shape}")

# 去重 (Symbol, feature_year) 保留第一条
df = risk.drop_duplicates(subset=['Symbol', 'feature_year'], keep='first').copy()
print(f"  去重后: {df.shape}")

# ============================================================
# 2. 按公司排序,计算时序特征
# ============================================================
print("\n" + "=" * 60)
print("P0-2.2: 计算时序特征")
print("=" * 60)

# 按 (Symbol, feature_year) 排序
df = df.sort_values(['Symbol', 'feature_year']).reset_index(drop=True)

# 计算 YoY 变化(对每家公司)
def compute_yoy(grp):
    grp = grp.sort_values('feature_year')
    # 净利润 YoY (用 net_margin 作代理,因为是比率)
    grp['net_margin_yoy'] = grp['net_margin'].pct_change()
    grp['roe_yoy'] = grp['roe'].pct_change()
    grp['ocf_to_rev_yoy'] = grp['ocf_to_rev'].pct_change()
    grp['debt_ratio_yoy'] = grp['debt_ratio'].diff()  # 绝对变化
    # 连续亏损(2+ 年 net_margin < 0)
    grp['consecutive_loss'] = (grp['net_margin'] < 0).rolling(window=2, min_periods=2).sum()
    grp['consecutive_loss'] = (grp['consecutive_loss'] >= 2).astype(int)
    # 营收代理:用 asset_turnover 反推(实际营收难拿)
    grp['asset_turnover_yoy'] = grp['asset_turnover'].pct_change()
    # 现金流恶化:OCF 同比下降 > 50%
    grp['ocf_plunge'] = ((grp['ocf_to_rev_yoy'] < -0.5)).astype(int)
    # 负债恶化:debt_ratio 同比上升 > 5pp
    grp['leverage_deterioration'] = ((grp['debt_ratio_yoy'] > 0.05)).astype(int)
    # 3 年累计下滑 (用 rolling 算)
    grp['np_3y_decline'] = grp['net_margin'].rolling(window=3, min_periods=3).apply(
        lambda x: 1 if all(x.diff().dropna() < 0) else 0, raw=False)
    return grp

df = df.groupby('Symbol', group_keys=False).apply(compute_yoy)
print(f"  完成时序特征计算")

# ============================================================
# 3. 时序特征汇总
# ============================================================
print("\n" + "=" * 60)
print("P0-2.3: 时序特征分布")
print("=" * 60)

new_features = ['net_margin_yoy', 'roe_yoy', 'ocf_to_rev_yoy', 'debt_ratio_yoy',
                'consecutive_loss', 'ocf_plunge', 'leverage_deterioration', 'np_3y_decline']
for f in new_features:
    non_null = df[f].notna().sum()
    if df[f].dtype in [float, 'float64']:
        print(f"  {f:25s} 有效: {non_null:5d}  mean: {df[f].mean():+.3f}  std: {df[f].std():.3f}")
    else:
        print(f"  {f:25s} 有效: {non_null:5d}  触发率: {df[f].mean()*100:.1f}%")

# ============================================================
# 4. 硬规则:时序信号 → 风险分调整
# ============================================================
print("\n" + "=" * 60)
print("P0-2.4: 时序信号 → 风险分调整")
print("=" * 60)

df['risk_score_v2'] = df['risk_score'].copy()  # 已含 P0-1 治理信号
adjustments = []

# R11: 连续 2 年亏损
mask = (df['consecutive_loss'] == 1)
df.loc[mask, 'risk_score_v2'] = df.loc[mask, 'risk_score_v2'].clip(lower=0.65)
adjustments.append(('R11 连续亏损', mask.sum(), 0.65))

# R12: 净利润 YoY 暴跌(下降 > 50%)
mask = (df['net_margin_yoy'] < -0.5) & (df['net_margin'] < 0)
df.loc[mask, 'risk_score_v2'] = df.loc[mask, 'risk_score_v2'].clip(lower=0.6)
adjustments.append(('R12 净利润暴跌', mask.sum(), 0.6))

# R13: 现金流恶化(同比下降 > 50%)
mask = (df['ocf_plunge'] == 1)
df.loc[mask, 'risk_score_v2'] = df.loc[mask, 'risk_score_v2'].clip(lower=0.55)
adjustments.append(('R13 现金流恶化', mask.sum(), 0.55))

# R14: 负债恶化(debt_ratio 上升 > 5pp)
mask = (df['leverage_deterioration'] == 1)
df.loc[mask, 'risk_score_v2'] = df.loc[mask, 'risk_score_v2'].clip(lower=0.5)
adjustments.append(('R14 负债恶化', mask.sum(), 0.5))

# R15: 3 年累计下滑
mask = (df['np_3y_decline'] == 1)
df.loc[mask, 'risk_score_v2'] = df.loc[mask, 'risk_score_v2'].clip(lower=0.6)
adjustments.append(('R15 3年累计下滑', mask.sum(), 0.6))

for name, n, thresh in adjustments:
    print(f"  {name}: 触发 {n} 条, 风险分下限 {thresh}")

# 重新分配等级
df['risk_level_v2'] = df['risk_score_v2'].apply(assign_level)

# ============================================================
# 5. 验证 — 洲际油气回测
# ============================================================
print("\n" + "=" * 60)
print("P0-2.5: 验证 — 洲际油气 600759 回测")
print("=" * 60)

zj = df[df['Symbol'] == '600759'].sort_values('violation_year', ascending=False)
if len(zj) > 0:
    zj_latest = zj.iloc[0]
    print(f"  洲际油气 {zj_latest['violation_year']} 年:")
    print(f"    原风险分 (P0-1): {zj_latest['risk_score']:.4f} → {zj_latest['risk_level']}")
    print(f"    新风险分 (P0-2): {zj_latest['risk_score_v2']:.4f} → {zj_latest['risk_level_v2']}")
    print(f"    净利润 YoY: {zj_latest['net_margin_yoy']:+.3f}")
    print(f"    OCF YoY: {zj_latest['ocf_to_rev_yoy']:+.3f}")
    print(f"    连续亏损: {zj_latest['consecutive_loss']}")
    print(f"    现金流恶化: {zj_latest['ocf_plunge']}")
    print(f"    负债恶化: {zj_latest['leverage_deterioration']}")
    print(f"    3年累计下滑: {zj_latest['np_3y_decline']}")

print()
sj = df[df['Symbol'] == '600520'].sort_values('violation_year', ascending=False)
if len(sj) > 0:
    sj_latest = sj.iloc[0]
    print(f"  三佳科技 {sj_latest['violation_year']} 年:")
    print(f"    原风险分 (P0-1): {sj_latest['risk_score']:.4f} → {sj_latest['risk_level']}")
    print(f"    新风险分 (P0-2): {sj_latest['risk_score_v2']:.4f} → {sj_latest['risk_level_v2']}")
    print(f"    净利润 YoY: {sj_latest['net_margin_yoy']:+.3f}")

# ============================================================
# 6. 分布对比
# ============================================================
print("\n" + "=" * 60)
print("P0-2.6: 风险等级分布对比")
print("=" * 60)

new_levels = df['risk_level_v2'].value_counts()
v1_levels = df['risk_level'].value_counts()
print(f"  P0-1 分布: {dict(v1_levels)}")
print(f"  P0-2 分布: {dict(new_levels)}")

# 已知违规命中率
actual_fraud = df['ann_fin_flag'] == 1
hit_p01 = ((df.loc[actual_fraud, 'risk_level'] == '高风险').sum() / actual_fraud.sum() * 100)
hit_p02 = ((df.loc[actual_fraud, 'risk_level_v2'] == '高风险').sum() / actual_fraud.sum() * 100)
print(f"  已知违规命中率: P0-1 = {hit_p01:.1f}% → P0-2 = {hit_p02:.1f}%")

# ============================================================
# 7. 保存
# ============================================================
print("\n" + "=" * 60)
print("P0-2.7: 保存")
print("=" * 60)

# 选关键列保存
save_cols = ['Symbol', 'ShortName', 'industry', 'violation_year', 'feature_year',
             'ann_related', 'ann_fin_flag', 'third_party_flag',
             'roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover',
             'net_margin', 'ocf_to_rev',
             'p_ml', 'rule_score_norm', 'risk_score', 'risk_level',
             'is_st_current', 'is_strict_st', 'is_st', 'has_prior_violation',
             'n_violations_prior_1y', 'n_violations_prior_3y',
             'net_margin_yoy', 'roe_yoy', 'ocf_to_rev_yoy', 'debt_ratio_yoy',
             'consecutive_loss', 'ocf_plunge', 'leverage_deterioration', 'np_3y_decline',
             'risk_score_v2', 'risk_level_v2']
save_cols = [c for c in save_cols if c in df.columns]
df[save_cols].to_csv(os.path.join(DATA_DIR, "risk_scored_temporal.csv"), index=False)
print(f"  → data/risk_scored_temporal.csv ({os.path.getsize(os.path.join(DATA_DIR, 'risk_scored_temporal.csv'))/1024:.0f} KB)")

print("\n" + "=" * 60)
print("✅ P0-2 完成: 时序特征已加入 (R11-R15)")
print("=" * 60)

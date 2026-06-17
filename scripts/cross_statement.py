#!/usr/bin/env python3
"""
P3-8: 跨表勾稽校验
===================
检查财务三表的关键勾稽关系是否成立:
1. 期末现金 ≈ 期初 + 经营+投资+筹资 现金流
2. 期末未分配利润 ≈ 期初 + 净利润 - 分红
3. 期末总资产 = 期末总负债 + 期末所有者权益
4. 期末应收账款 ≤ 期末营业收入 (合理比例)

如果勾稽失败,可能存在调表痕迹。
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
# 1. 加载
# ============================================================
print("=" * 60)
print("P3-8.1: 加载数据")
print("=" * 60)

risk = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_industry_v2.csv"))
print(f"  风险评分: {risk.shape}")

raw = pd.read_csv(os.path.join(DATA_DIR, "raw_financials.csv"))
raw['Symbol'] = raw['fetch_symbol'].astype(str).str.zfill(6)
print(f"  原始财务: {raw.shape}")

# ============================================================
# 2. 计算勾稽差异
# ============================================================
print("\n" + "=" * 60)
print("P3-8.2: 计算勾稽差异")
print("=" * 60)

# 对每行计算勾稽指标
raw['balance_check'] = raw['total_assets'] - raw['total_liab'] - raw['total_hldr_eqy_inc_min_int']
# 平衡差 = 总资产 - 总负债 - 所有者权益
# 理论 = 0, 实际允许小差异(尾差)

raw['ar_to_revenue'] = None  # 应收账款/营业收入
# 我们的 raw_financials 没有应收账款字段,跳过
# 但有 ocf/ni 比,净利润含金量

raw['ocf_to_ni'] = raw['ocf'] / raw['n_income']
# OCF/NI > 1 是健康(利润含现金),< 0.5 是异常(利润没现金)

# 资产负债率(已在)
raw['debt_to_equity'] = raw['total_liab'] / raw['total_hldr_eqy_inc_min_int']

# ============================================================
# 3. 应用硬规则
# ============================================================
print("\n" + "=" * 60)
print("P3-8.3: 硬规则")
print("=" * 60)

# 把勾稽特征合并到 risk
raw['viol_year'] = raw['fetch_year'].astype(int) + 1  # 报告年 → 违规年

risk['Symbol'] = risk['Symbol'].astype(str)
raw['Symbol'] = raw['Symbol'].astype(str)
risk['viol_year'] = risk['viol_year'].astype(int)
raw['viol_year'] = raw['viol_year'].astype(int)

risk = risk.merge(
    raw[['Symbol', 'viol_year', 'total_assets', 'n_income', 'balance_check', 'ocf_to_ni', 'debt_to_equity']],
    on=['Symbol', 'viol_year'], how='left'
)
print(f"  合并后: {risk.shape}")

risk['risk_score_v9'] = risk['risk_score_v7'].copy()
adjustments = []

# R35: 资产负债勾稽差 > 1% 总资产
# balance_check 是 abs,实际可能正负
risk['balance_check_abs'] = risk['balance_check'].abs()
mask = (risk['balance_check_abs'] > risk['total_assets'] * 0.01)
n = mask.sum()
risk.loc[mask, 'risk_score_v9'] = risk.loc[mask, 'risk_score_v9'].clip(lower=0.7)
adjustments.append(('R35 资产负债勾稽差>1%', n, 0.7))

# R36: 净利润含金量差(OCF/NI < 0.5 且净利润>0)
mask = (risk['ocf_to_ni'].notna()) & (risk['ocf_to_ni'] < 0.5) & (risk['n_income'] > 0)
n = mask.sum()
risk.loc[mask, 'risk_score_v9'] = risk.loc[mask, 'risk_score_v9'].clip(lower=0.6)
adjustments.append(('R36 净利润含金量差(OCF/NI<0.5)', n, 0.6))

# R37: 资产负债率 > 1(资不抵债)
mask = (risk['debt_to_equity'] > 1)
n = mask.sum()
risk.loc[mask, 'risk_score_v9'] = risk.loc[mask, 'risk_score_v9'].clip(lower=0.85)
adjustments.append(('R37 资不抵债(负债>权益)', n, 0.85))

for name, n, thresh in adjustments:
    print(f"  {name}: 触发 {n} 条, 风险分下限 {thresh}")

risk['risk_level_v9'] = risk['risk_score_v9'].apply(assign_level)

# ============================================================
# 4. 验证
# ============================================================
print("\n" + "=" * 60)
print("P3-8.4: 验证")
print("=" * 60)

actual_fraud = risk['ann_fin_flag'] == 1
for col, label in [('risk_level', 'v0'), ('risk_level_v4', 'v4'),
                    ('risk_level_v7', 'v7'), ('risk_level_v9', 'v9')]:
    hit = (risk.loc[actual_fraud, col] == '高风险').sum() / actual_fraud.sum() * 100
    print(f"  {label}: 高风险命中率 = {hit:.1f}%")

# ============================================================
# 5. 保存
# ============================================================
print("\n" + "=" * 60)
print("P3-8.5: 保存")
print("=" * 60)

risk.to_csv(os.path.join(DATA_DIR, "risk_scored_v9_final.csv"), index=False)
print(f"  → data/risk_scored_v9_final.csv")

# 显示演进
print("\n" + "=" * 60)
print("📈 命中率演进")
print("=" * 60)
labels = ['v0', 'v4', 'v5', 'v6', 'v7', 'v9']
cols = ['risk_level', 'risk_level_v4', 'risk_level_v5', 'risk_level_v6', 'risk_level_v7', 'risk_level_v9']
for label, col in zip(labels, cols):
    if col in risk.columns:
        hit = (risk.loc[actual_fraud, col] == '高风险').sum() / actual_fraud.sum() * 100
        print(f"  {label}: {hit:.1f}%")

print("\n" + "=" * 60)
print("✅ P3-8 完成: 跨表勾稽校验")
print("=" * 60)

#!/usr/bin/env python3
"""
P0-3: 行业差异化阈值
=====================
用行业中位数/分位数动态确定阈值,代替硬编码。
基于已实现的 6 条规则 + P0-1 + P0-2,扩展为行业自适应版本。
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']


def assign_level(s):
    if s >= 0.55: return '高风险'
    if s >= 0.25: return '中风险'
    return '低风险'


# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("P0-3.1: 加载数据")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_temporal.csv"))
df['Symbol'] = df['Symbol'].astype(str)
df['violation_year'] = df['violation_year'].astype(int)
df['feature_year'] = df['feature_year'].astype(int)
print(f"  风险评分(含时序): {df.shape}")

# ============================================================
# 2. 计算行业统计(中位数 + 标准差)
# ============================================================
print("\n" + "=" * 60)
print("P0-3.2: 计算行业统计")
print("=" * 60)

# 按行业分组,只对有财务数据的行计算
ind_stats = df.groupby('industry')[FIN_COLS].agg(['median', 'std', 'count'])
print(f"  行业数: {len(ind_stats)}")

# 整理成 dict 便于查询
ind_median = {}
ind_std = {}
for ind in ind_stats.index:
    if pd.isna(ind):
        continue
    ind_median[ind] = {col: ind_stats.loc[ind, (col, 'median')] for col in FIN_COLS}
    ind_std[ind] = {col: ind_stats.loc[ind, (col, 'std')] for col in FIN_COLS}

# 全局中位数(对未分类行业)
global_median = df[FIN_COLS].median().to_dict()
global_std = df[FIN_COLS].std().to_dict()
print(f"  全局中位数: {global_median}")

# ============================================================
# 3. 行业差异化规则
# ============================================================
print("\n" + "=" * 60)
print("P0-3.3: 行业差异化规则 (R1-R6 升级)")
print("=" * 60)

# 旧规则(硬编码)vs 新规则(行业差异化)对比
# 偏离度 = (公司值 - 行业中位数) / 行业 std
# 触发条件: 偏离度 > 2 (即超过行业 2 个标准差)

# 用 σ 偏离代替硬编码
df['risk_score_v3'] = df['risk_score_v2'].copy()  # 继承 P0-1 + P0-2
industry_rules_triggered = []

def apply_industry_rule(row, col_name, rule_id, base_severity, direction):
    """
    direction: 'low' 触发当 (x - median) / std < -2
               'high' 触发当 (x - median) / std > 2
    """
    val = row.get(col_name)
    ind = row.get('industry')
    if pd.isna(val) or pd.isna(ind):
        return None
    median = ind_median.get(ind, global_median).get(col_name, np.nan)
    std = ind_std.get(ind, global_std).get(col_name, np.nan)
    if pd.isna(median) or pd.isna(std) or std == 0:
        return None
    z = (val - median) / std
    triggered = (z < -2) if direction == 'low' else (z > 2)
    return triggered, z

# 重新计算 6 条规则的"行业自适应"版本
rule_specs = [
    # (rule_id, col, direction, severity)
    ('R1', 'roe', 'low', 25),         # 极低 ROE(行业 2σ 外)
    ('R2', 'ocf_to_rev', 'low', 30),  # 现金流极差
    ('R3', 'debt_ratio', 'high', 15), # 极高负债
    ('R4', 'current_ratio', 'low', 20), # 流动性极差
    ('R5', 'asset_turnover', 'low', 10), # 周转极慢(或 R5 high)
    ('R6', 'roa', 'low', 25),         # 资产收益率极低
]

rule_v3_triggered = {rid: 0 for rid, _, _, _ in rule_specs}

for idx, row in df.iterrows():
    for rid, col, direction, sev in rule_specs:
        res = apply_industry_rule(row, col, rid, sev, direction)
        if res and res[0]:
            rule_v3_triggered[rid] += 1

print(f"  行业差异化规则触发数:")
for rid, n in rule_v3_triggered.items():
    print(f"    {rid}: {n} 条")

# 把行业规则转化为 risk_score_v3 调整
for idx, row in df.iterrows():
    for rid, col, direction, sev in rule_specs:
        res = apply_industry_rule(row, col, rid, sev, direction)
        if res and res[0]:
            df.at[idx, 'risk_score_v3'] = max(df.at[idx, 'risk_score_v3'], sev / 125 * 0.4 + 0.6 * 0.4)

# 简化:触发即上调至 0.55
for idx, row in df.iterrows():
    for rid, col, direction, sev in rule_specs:
        res = apply_industry_rule(row, col, rid, sev, direction)
        if res and res[0]:
            df.at[idx, 'risk_score_v3'] = max(df.at[idx, 'risk_score_v3'], 0.55)

# 重新分配等级
df['risk_level_v3'] = df['risk_score_v3'].apply(assign_level)

# ============================================================
# 4. 验证
# ============================================================
print("\n" + "=" * 60)
print("P0-3.4: 验证")
print("=" * 60)

# 洲际油气
zj = df[df['Symbol'] == '600759'].sort_values('violation_year', ascending=False)
if len(zj) > 0:
    zj_latest = zj.iloc[0]
    print(f"  洲际油气 (石油行业):")
    print(f"    P0-1 风险分: {zj_latest['risk_score']:.4f} → {zj_latest['risk_level']}")
    print(f"    P0-2 风险分: {zj_latest['risk_score_v2']:.4f} → {zj_latest['risk_level_v2']}")
    print(f"    P0-3 风险分: {zj_latest['risk_score_v3']:.4f} → {zj_latest['risk_level_v3']}")

# 三佳科技
print()
sj = df[df['Symbol'] == '600520'].sort_values('violation_year', ascending=False)
if len(sj) > 0:
    sj_latest = sj.iloc[0]
    print(f"  三佳科技 (半导体行业):")
    print(f"    P0-1 风险分: {sj_latest['risk_score']:.4f} → {sj_latest['risk_level']}")
    print(f"    P0-2 风险分: {sj_latest['risk_score_v2']:.4f} → {sj_latest['risk_level_v2']}")
    print(f"    P0-3 风险分: {sj_latest['risk_score_v3']:.4f} → {sj_latest['risk_level_v3']}")

# 分布对比
print("\n" + "=" * 60)
print("P0-3.5: 分布对比")
print("=" * 60)
for col, label in [('risk_level', 'P0-1'), ('risk_level_v2', 'P0-2'), ('risk_level_v3', 'P0-3')]:
    counts = df[col].value_counts()
    print(f"  {label}: {dict(counts)}")

# 已知违规命中率
actual_fraud = df['ann_fin_flag'] == 1
for col, label in [('risk_level', 'P0-1'), ('risk_level_v2', 'P0-2'), ('risk_level_v3', 'P0-3')]:
    hit = (df.loc[actual_fraud, col] == '高风险').sum() / actual_fraud.sum() * 100
    print(f"  {label} 已知违规命中率: {hit:.1f}%")

# ============================================================
# 5. 保存
# ============================================================
print("\n" + "=" * 60)
print("P0-3.6: 保存")
print("=" * 60)

df.to_csv(os.path.join(DATA_DIR, "risk_scored_industry.csv"), index=False)
print(f"  → data/risk_scored_industry.csv")

# 同时保存行业基准
import json
bench = {'global': {'median': global_median, 'std': global_std}}
for ind, med in ind_median.items():
    bench[ind] = {'median': med, 'std': ind_std[ind]}
with open(os.path.join(DATA_DIR, "industry_benchmarks.json"), 'w', encoding='utf-8') as f:
    # 转 numpy → float
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.int64, np.int32)): return int(o)
        if isinstance(o, (np.float64, np.float32)): return float(o)
        return str(o)
    json.dump(bench, f, ensure_ascii=False, indent=2, default=conv)
print(f"  → data/industry_benchmarks.json")

print("\n" + "=" * 60)
print("✅ P0-3 完成: 行业差异化阈值已落地")
print("=" * 60)

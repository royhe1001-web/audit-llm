#!/usr/bin/env python3
"""
阶段三·步骤 7: 审计中间表 (阶段三核心交付)
========================================
整合 features + industry + rules + scores + anomaly 五大数据源
主键: (Symbol, violation_year)
约 7,997 行 × 25+ 列
"""

import os
import pandas as pd
import numpy as np

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

# ============================================================
# 1. 加载所有数据源
# ============================================================
print("=" * 60)
print("步骤 7.1: 加载 5 大数据源")
print("=" * 60)

feat = pd.read_csv(os.path.join(DATA_DIR, "fraud_features_enriched.csv"))
print(f"  features: {feat.shape}, 列: {feat.columns.tolist()[:6]}...")

risk = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_full.csv"))
# risk_scored_full 有 1830 个重复 (Symbol, year),先去重
risk = risk.drop_duplicates(subset=['Symbol', 'violation_year'], keep='first')
print(f"  risk_scored: {risk.shape} (去重后)")

rules = pd.read_csv(os.path.join(OUT_DIR, "rule_trigger_aggregate.csv"))
# 计算 rule_score_norm
rules['rule_score_norm'] = (rules['rule_score_sum'] / 125).clip(0, 1)
print(f"  rules: {rules.shape}")

anom = pd.read_csv(os.path.join(DATA_DIR, "anomaly_companies.csv"))
print(f"  anomalies: {anom.shape}")

# ============================================================
# 2. 整合
# ============================================================
print("\n" + "=" * 60)
print("步骤 7.2: 整合宽表")
print("=" * 60)

# Step 1: 基础列(从 features)
base = feat[['Symbol', 'ShortName', 'industry', 'area', 'list_date',
              'violation_year', 'feature_year',
              'ann_related', 'ann_fin_flag', 'third_party_flag',
              'roe', 'roa', 'debt_ratio', 'current_ratio',
              'asset_turnover', 'net_margin', 'ocf_to_rev']].copy()
print(f"  Step 1: 基础列 = {base.shape}")

# Step 2: 合并 risk 评分
risk_cols = ['Symbol', 'violation_year', 'p_ml', 'risk_score', 'risk_level']
base = base.merge(risk[risk_cols], on=['Symbol', 'violation_year'], how='left')
print(f"  Step 2: + 风险评分 = {base.shape}")

# Step 3: 合并规则
rule_cols = ['Symbol', 'violation_year', 'n_rules_triggered', 'rule_ids',
              'rule_score_sum', 'rule_score_norm']
base = base.merge(rules[rule_cols], on=['Symbol', 'violation_year'], how='left')
base['n_rules_triggered'] = base['n_rules_triggered'].fillna(0).astype(int)
base['rule_ids'] = base['rule_ids'].fillna('')
base['rule_score_sum'] = base['rule_score_sum'].fillna(0)
base['rule_score_norm'] = base['rule_score_norm'].fillna(0)
print(f"  Step 3: + 规则得分 = {base.shape}")

# Step 4: 合并异常
anom_cols = ['Symbol', 'violation_year', 'is_anomaly', 'anomaly_method',
              'if_anomaly', 'lof_anomaly']
base = base.merge(anom[anom_cols], on=['Symbol', 'violation_year'], how='left')
for c in ['is_anomaly', 'if_anomaly', 'lof_anomaly']:
    base[c] = base[c].fillna(0).astype(int)
base['anomaly_method'] = base['anomaly_method'].fillna('None')
print(f"  Step 4: + 异常 = {base.shape}")

# ============================================================
# 3. 添加派生列
# ============================================================
print("\n" + "=" * 60)
print("步骤 7.3: 添加派生列")
print("=" * 60)

# 上市年限
base['list_date'] = base['list_date'].astype(str).str.replace('', '')
base['list_year'] = pd.to_numeric(base['list_date'], errors='coerce')
base['list_years'] = base['violation_year'] - base['list_year']
base['list_years'] = base['list_years'].where(base['list_years'] > 0, np.nan)

# 风险标签简化
base['is_high_risk'] = (base['risk_level'] == '高风险').astype(int)
base['is_known_fraud'] = (base['ann_fin_flag'] == 1).astype(int)

# 综合风险标识 (高风险 OR 异常)
base['priority_flag'] = ((base['is_high_risk'] == 1) | (base['is_anomaly'] == 1)).astype(int)

print(f"  新增列: list_year, list_years, is_high_risk, is_known_fraud, priority_flag")

# ============================================================
# 4. 主键唯一性检查
# ============================================================
print("\n" + "=" * 60)
print("步骤 7.4: 主键唯一性检查")
print("=" * 60)

n_total = len(base)
n_unique = base.groupby(['Symbol', 'violation_year']).ngroups
n_dup = n_total - n_unique
print(f"  总行数: {n_total}")
print(f"  唯一 (Symbol, year): {n_unique}")
print(f"  重复: {n_dup}")

# ============================================================
# 5. 列顺序
# ============================================================
print("\n" + "=" * 60)
print("步骤 7.5: 列顺序整理")
print("=" * 60)

col_order = [
    # 基础信息
    'Symbol', 'ShortName', 'industry', 'area', 'list_date', 'list_year', 'list_years',
    'violation_year', 'feature_year',
    # 违规标签
    'ann_related', 'ann_fin_flag', 'third_party_flag',
    # 财务特征
    'roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev',
    # ML 评分
    'p_ml',
    # 规则评分
    'n_rules_triggered', 'rule_ids', 'rule_score_sum', 'rule_score_norm',
    # 综合风险
    'risk_score', 'risk_level', 'is_high_risk', 'is_known_fraud',
    # 异常
    'is_anomaly', 'anomaly_method', 'if_anomaly', 'lof_anomaly',
    # 优先级
    'priority_flag',
]
base = base[col_order]
print(f"  列数: {len(col_order)}")
print(f"  顺序: {col_order[:6]} ... {col_order[-3:]}")

# ============================================================
# 6. 保存
# ============================================================
print("\n" + "=" * 60)
print("步骤 7.6: 保存审计中间表")
print("=" * 60)

out_path = os.path.join(DATA_DIR, "audit_intermediate_table.csv")
base.to_csv(out_path, index=False)
size_kb = os.path.getsize(out_path) / 1024
print(f"  → {out_path} ({size_kb:.0f} KB)")
print(f"  形状: {base.shape}")

# ============================================================
# 7. 总结统计
# ============================================================
print("\n" + "=" * 60)
print("步骤 7.7: 关键统计")
print("=" * 60)

print(f"  唯一公司数: {base['Symbol'].nunique()}")
print(f"  唯一行业数: {base['industry'].nunique()}")
print(f"  年份范围: {int(base['violation_year'].min())}-{int(base['violation_year'].max())}")
print(f"  已知违规 (ann_fin_flag=1): {(base['ann_fin_flag']==1).sum()}")
print(f"  高风险数: {(base['risk_level']=='高风险').sum()}")
print(f"  异常数: {(base['is_anomaly']==1).sum()}")
print(f"  优先级 (高风险 OR 异常): {(base['priority_flag']==1).sum()}")
print(f"  财务完整行: {base['roe'].notna().sum()} ({base['roe'].notna().mean()*100:.1f}%)")
print(f"  平均风险分: {base['risk_score'].mean():.3f}")

print("\n" + "=" * 60)
print("✅ 步骤 7 完成 — 审计中间表构建完成")
print("=" * 60)

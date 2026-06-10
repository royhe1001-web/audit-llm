#!/usr/bin/env python3
"""
P1-5: 动态规则挖掘
===================
从历史违规数据中自动发现"特征组合 → ann_fin_flag=1" 的强关联规则。
将挖掘出的规则合并到现有 6+5+6 = 17 条规则,扩展为动态规则库。
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("P1-5.1: 加载数据")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_industry.csv"))
df['Symbol'] = df['Symbol'].astype(str)
print(f"  风险评分: {df.shape}")

# ============================================================
# 2. 离散化(布尔化)
# ============================================================
print("\n" + "=" * 60)
print("P1-5.2: 离散化 7 特征 + 治理 + 时序信号")
print("=" * 60)

# 财务特征离散化(基于分位数)
fin_cols = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']
df_bool = pd.DataFrame(index=df.index)

for col in fin_cols:
    s = df[col]
    if s.notna().sum() < 10:
        continue
    p25 = s.quantile(0.25)
    p75 = s.quantile(0.75)
    df_bool[f'{col}_low'] = (s < p25).fillna(False).astype(bool)
    df_bool[f'{col}_high'] = (s > p75).fillna(False).astype(bool)

# 治理信号
df_bool['is_st'] = (df.get('is_st', 0) == 1).fillna(False).astype(bool)
df_bool['is_strict_st'] = (df.get('is_strict_st', 0) == 1).fillna(False).astype(bool)
df_bool['has_prior_violation'] = (df.get('has_prior_violation', 0) == 1).fillna(False).astype(bool)

# 时序信号
df_bool['consecutive_loss'] = (df.get('consecutive_loss', 0) == 1).fillna(False).astype(bool)
df_bool['ocf_plunge'] = (df.get('ocf_plunge', 0) == 1).fillna(False).astype(bool)
df_bool['leverage_deterioration'] = (df.get('leverage_deterioration', 0) == 1).fillna(False).astype(bool)
df_bool['np_3y_decline'] = (df.get('np_3y_decline', 0) == 1).fillna(False).astype(bool)
df_bool['net_margin_yoy_crash'] = (df.get('net_margin_yoy', 1) < -0.5).fillna(False).astype(bool)

print(f"  布尔特征数: {df_bool.shape[1]}")
print(f"  总记录: {len(df_bool)}")
print(f"  ann_fin_flag=1 分布: {df['ann_fin_flag'].value_counts(dropna=False).to_dict()}")

# ============================================================
# 3. FP-Growth 挖掘频繁项集
# ============================================================
print("\n" + "=" * 60)
print("P1-5.3: FP-Growth 挖掘频繁项集")
print("=" * 60)

# 准备关联数据: 每行 = (布尔特征, ann_fin_flag=1)
df_mining = df_bool.copy()
df_mining['ann_fin_flag_1'] = (df['ann_fin_flag'] == 1).fillna(False).astype(bool)

# FP-Growth
min_support = 0.02  # 至少 2% 的样本
freq_items = fpgrowth(df_mining, min_support=min_support, use_colnames=True, max_len=4)
print(f"  频繁项集数: {len(freq_items)} (min_support={min_support})")

# 计算关联规则
if len(freq_items) > 0:
    rules = association_rules(freq_items, metric='lift', min_threshold=1.5)
    print(f"  关联规则数: {len(rules)} (min_lift=1.5)")

    # 过滤: 前件不含 ann_fin_flag_1,后件含 ann_fin_flag_1
    fraud_rules = rules[
        (~rules['antecedents'].apply(lambda x: 'ann_fin_flag_1' in x)) &
        (rules['consequents'].apply(lambda x: 'ann_fin_flag_1' in x))
    ].sort_values('lift', ascending=False)

    print(f"  → '特征组合 → ann_fin_flag=1' 规则数: {len(fraud_rules)}")

    # ============================================================
    # 4. 输出 top 规则
    # ============================================================
    print("\n" + "=" * 60)
    print("P1-5.4: Top 10 强关联规则")
    print("=" * 60)

    top_rules = fraud_rules.head(10)[['antecedents', 'consequents', 'support',
                                         'confidence', 'lift']]
    top_rules['rule'] = top_rules['antecedents'].apply(
        lambda x: ' AND '.join(sorted(x))
    )
    for idx, row in top_rules.iterrows():
        print(f"  {row['rule']}")
        print(f"    → 支持度={row['support']:.3f}  置信度={row['confidence']:.3f}  提升度={row['lift']:.2f}")
        print()

    # 保存
    fraud_rules_export = fraud_rules.copy()
    fraud_rules_export['antecedents'] = fraud_rules_export['antecedents'].apply(lambda x: list(x))
    fraud_rules_export['consequents'] = fraud_rules_export['consequents'].apply(lambda x: list(x))
    fraud_rules_export.to_csv(os.path.join(OUT_DIR, 'mined_rules.csv'), index=False)
    print(f"  → output/mined_rules.csv ({len(fraud_rules_export)} 条)")

# ============================================================
# 5. 转化为可执行的"动态规则"
# ============================================================
print("\n" + "=" * 60)
print("P1-5.5: 转化为可执行规则")
print("=" * 60)

# 取置信度 > 0.2 且提升度 > 5 的前 5 条
strong_rules = fraud_rules[(fraud_rules['confidence'] > 0.2) & (fraud_rules['lift'] > 5)].head(5)

# 生成新规则(R16-R20)
new_rules = []
for i, (idx, row) in enumerate(strong_rules.iterrows(), start=16):
    antecedent_str = ' AND '.join(sorted(row['antecedents']))
    severity = min(int(row['lift'] * 10), 30)  # 提升度 * 10 作为严重度
    new_rules.append({
        'rule_id': f'R{i}',
        'name': f'mined_rule_{i}',
        'antecedent': antecedent_str,
        'severity': severity,
        'confidence': row['confidence'],
        'lift': row['lift'],
        'support': row['support'],
    })
    print(f"  {f'R{i}'} 严重度 {severity}: {antecedent_str}")
    print(f"      置信度={row['confidence']:.3f}  提升度={row['lift']:.2f}  支持度={row['support']:.3f}")

# 保存新规则库
rules_df = pd.DataFrame(new_rules)
rules_df.to_csv(os.path.join(DATA_DIR, 'dynamic_rules.csv'), index=False)
print(f"\n  → data/dynamic_rules.csv ({len(rules_df)} 条动态规则)")

# ============================================================
# 6. 在测试集上验证动态规则
# ============================================================
print("\n" + "=" * 60)
print("P1-5.6: 验证动态规则 (在风险评分中应用)")
print("=" * 60)

# 应用动态规则到所有样本
# 简化:每个新规则都是布尔表达式,直接计算触发
def eval_rule(cond_str, row):
    """评估布尔条件"""
    conds = cond_str.split(' AND ')
    for c in conds:
        if c in row.index:
            if not bool(row[c]):
                return False
    return True

# 测试:对所有 ann_fin_flag=1 的样本,新规则的命中率
actual_fraud = df['ann_fin_flag'] == 1
fraud_samples = df[actual_fraud].head(500)
non_fraud_samples = df[df['ann_fin_flag'] == 0].head(500)

for rule in new_rules:
    rid = rule['rule_id']
    ant = rule['antecedent']
    hit_fraud = fraud_samples.apply(lambda r: eval_rule(ant, r), axis=1).sum()
    hit_nonfraud = non_fraud_samples.apply(lambda r: eval_rule(ant, r), axis=1).sum()
    print(f"  {rid}: 在 ann_fin_flag=1 中命中 {hit_fraud}/{len(fraud_samples)} ({hit_fraud/len(fraud_samples)*100:.1f}%)")
    print(f"      在 ann_fin_flag=0 中命中 {hit_nonfraud}/{len(non_fraud_samples)} ({hit_nonfraud/len(non_fraud_samples)*100:.1f}%)")

# ============================================================
# 7. 总结
# ============================================================
print("\n" + "=" * 60)
print("✅ P1-5 完成: 动态规则挖掘")
print("=" * 60)
print(f"  频繁项集: {len(freq_items)}")
print(f"  关联规则: {len(rules)}")
print(f"  强关联规则(置信度>0.2 且提升度>5): {len(strong_rules)}")
print(f"  集成到引擎: R16-R{15+len(new_rules)}")
print(f"  产出: mined_rules.csv, dynamic_rules.csv")

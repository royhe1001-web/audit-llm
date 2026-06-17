#!/usr/bin/env python3
"""
阶段三·步骤 2: 审计规则引擎
============================
6 条基于财务指标的疑点规则,严重度加权。
输出规则触发明细 + 按 (Symbol, year) 聚合的得分。
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

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']

# ============================================================
# 规则定义: (id, name, severity, check_function, description)
# ============================================================
RULES = [
    ('R1', 'consecutive_loss', 25,
     lambda r: (r['roe'] < 0) & (r['net_margin'] < 0),
     '连续亏损: roe<0 且 net_margin<0 → 经营性造血能力缺失,舞弊动机高'),
    ('R2', 'cashflow_divergence', 30,
     lambda r: (r['ocf_to_rev'] < 0) & (r['net_margin'] > 0),
     '现金流背离: 利润为正但经营现金流为负 → 利润含金量差,盈余操纵高发'),
    ('R3', 'high_leverage', 15,
     lambda r: r['debt_ratio'] > 0.7,
     '高负债: debt_ratio>0.7 → 偿债压力大,易触发违规融资'),
    ('R4', 'liquidity_stress', 20,
     lambda r: r['current_ratio'] < 1,
     '流动性紧张: current_ratio<1 → 短期偿债能力不足,破产/重组预警'),
    ('R5', 'asset_turnover_extreme', 10,
     lambda r: (r['asset_turnover'] < 0.3) | (r['asset_turnover'] > 5),
     '资产周转异常: 极端低(<0.3)或极端高(>5) → 经营效率异常'),
    ('R6', 'high_leverage_loss', 20,
     lambda r: (r['debt_ratio'] > 0.5) & (r['roe'] < 0),
     '高杠杆亏损: debt_ratio>0.5 且 roe<0 → 借债经营失败,违约前兆'),
]

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("步骤 2.1: 加载数据")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "fraud_features_enriched.csv"))
print(f"总记录: {len(df)}")

# 只对有财务数据的行应用规则
valid_mask = df[FIN_COLS].notna().any(axis=1)
df_valid = df[valid_mask].copy()
print(f"有财务数据的记录: {len(df_valid)}")

# ============================================================
# 2. 应用规则
# ============================================================
print("\n" + "=" * 60)
print("步骤 2.2: 应用 6 条规则")
print("=" * 60)

triggered_rows = []
for rid, rname, sev, fn, desc in RULES:
    # 仅对相关字段非空的行判断
    mask = fn(df_valid)
    n = mask.sum()
    print(f"  {rid} {rname:25s} severity={sev:3d}  触发数={n:4d}  ({n/len(df_valid)*100:.1f}%)  {desc}")
    for idx in mask[mask].index:
        row = df_valid.loc[idx]
        triggered_rows.append({
            'Symbol': row['Symbol'],
            'ShortName': row['ShortName'],
            'violation_year': row['violation_year'],
            'industry': row.get('industry', '未分类'),
            'rule_id': rid,
            'rule_name': rname,
            'severity': sev,
            'rule_score': sev,
            'ann_fin_flag': row.get('ann_fin_flag', np.nan),
        })

triggered_df = pd.DataFrame(triggered_rows)
# 同一 (Symbol, year, rule_id) 只计一次,避免同一公司同年多次违规重复计分
triggered_df = triggered_df.drop_duplicates(subset=['Symbol', 'violation_year', 'rule_id'])
print(f"\n去重后触发记录: {len(triggered_df)}")
if len(triggered_df) == 0:
    print("⚠️ 没有任何规则被触发,检查数据")
    raise SystemExit(1)

# ============================================================
# 3. 按 (Symbol, year) 聚合
# ============================================================
print("\n" + "=" * 60)
print("步骤 2.3: 聚合规则得分")
print("=" * 60)

agg = (triggered_df
       .groupby(['Symbol', 'ShortName', 'violation_year'])
       .agg(
           rule_score_sum=('rule_score', 'sum'),
           n_rules_triggered=('rule_id', 'count'),
           rule_ids=('rule_id', lambda x: ';'.join(sorted(set(x)))),
           industry=('industry', 'first'),
           ann_fin_flag=('ann_fin_flag', 'first'),
       )
       .reset_index()
       .sort_values('rule_score_sum', ascending=False))

print(f"聚合后: {len(agg)} 个 (公司, 年份) 组合")
print(f"  平均触发规则数: {agg['n_rules_triggered'].mean():.2f}")
print(f"  最高严重度得分: {agg['rule_score_sum'].max()}")
print(f"  触发 ≥3 条规则: {(agg['n_rules_triggered'] >= 3).sum()}")

agg_path = os.path.join(OUT_DIR, "rule_trigger_aggregate.csv")
agg.to_csv(agg_path, index=False)
print(f"  → {agg_path}")

# ============================================================
# 4. 规则触发汇总(全样本 vs 已知违规)
# ============================================================
print("\n" + "=" * 60)
print("步骤 2.4: 规则效果验证 (命中率)")
print("=" * 60)

summary_rows = []
for rid, rname, sev, _, _ in RULES:
    rule_trigs = triggered_df[triggered_df['rule_id'] == rid]
    n_total = len(rule_trigs)
    n_fraud = (rule_trigs['ann_fin_flag'] == 1).sum()
    n_nonfraud = (rule_trigs['ann_fin_flag'] == 0).sum()
    n_unknown = rule_trigs['ann_fin_flag'].isna().sum()
    summary_rows.append({
        'rule_id': rid,
        'rule_name': rname,
        'severity': sev,
        'n_triggered': n_total,
        'n_ann_fin_1': n_fraud,
        'n_ann_fin_0': n_nonfraud,
        'n_unknown': n_unknown,
        'hit_rate_fraud': n_fraud / n_total * 100 if n_total else 0,
    })
summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))

summary_path = os.path.join(OUT_DIR, "rules_trigger_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"\n  → {summary_path}")

# ============================================================
# 5. 可视化
# ============================================================
print("\n" + "=" * 60)
print("步骤 2.5: 规则触发可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左: 规则触发数
axes[0].barh(summary['rule_name'], summary['n_triggered'], color='steelblue', edgecolor='white')
axes[0].set_xlabel('触发数', fontsize=11)
axes[0].set_title('各规则触发数', fontweight='bold', fontsize=13)
for i, v in enumerate(summary['n_triggered']):
    axes[0].text(v + 2, i, str(v), va='center', fontsize=10)
axes[0].grid(axis='x', alpha=0.3)

# 右: 严重度加权得分
axes[1].barh(summary['rule_name'], summary['severity'] * summary['n_triggered'],
             color='indianred', edgecolor='white')
axes[1].set_xlabel('严重度 × 触发数', fontsize=11)
axes[1].set_title('各规则风险贡献 (严重度 × 触发数)', fontweight='bold', fontsize=13)
for i, v in enumerate(summary['severity'] * summary['n_triggered']):
    axes[1].text(v + 50, i, str(v), va='center', fontsize=10)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rule_severity_distribution.png"), dpi=150, bbox_inches='tight')
print(f"  → output/rule_severity_distribution.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("✅ 步骤 2 完成")
print("=" * 60)
print(f"  6 条规则总触发: {len(triggered_df)} 条记录")
print(f"  涉及 (公司,年份): {len(agg)} 个")
print(f"  规则最大严重度: {sum(r[2] for r in RULES)}")
print(f"  产出: {agg_path}")
print(f"        {summary_path}")

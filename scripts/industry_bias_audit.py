#!/usr/bin/env python3
"""
阶段四·优化 8: 行业偏见检测
=============================
按行业分组,对比各组的:
- 正样本 Recall(ann_fin_flag=1 的命中率)
- 误报率(ann_fin_flag=0 的误报率)
- 平均风险分偏差
- 平均 p_ml 偏差

识别"模型在哪些行业系统性偏强或偏弱"。
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

print('=' * 60)
print('行业偏见检测')
print('=' * 60)

df = pd.read_csv(os.path.join(DATA_DIR, 'audit_intermediate_table.csv'))
print(f'总样本: {len(df)}, 行业数: {df["industry"].nunique()}')

# 整体基线
overall_fraud_rate = (df['ann_fin_flag'] == 1).mean()
overall_risk_mean = df['risk_score'].mean()
overall_p_ml_mean = df['p_ml'].mean()
print(f'\n整体基线:')
print(f'  违规率: {overall_fraud_rate:.2%}')
print(f'  平均 risk_score: {overall_risk_mean:.3f}')
print(f'  平均 p_ml: {overall_p_ml_mean:.3f}')

# 按行业分组
print('\n' + '=' * 60)
print('行业偏见检测(仅看 ann_fin_flag ∈ {0, 1} 的样本)')
print('=' * 60)

# 过滤有标签的样本
labeled = df[df['ann_fin_flag'].isin([0, 1])].copy()

bias_rows = []
for industry, grp in labeled.groupby('industry'):
    n_total = len(grp)
    n_fraud = (grp['ann_fin_flag'] == 1).sum()
    n_nonfraud = (grp['ann_fin_flag'] == 0).sum()

    if n_fraud < 5 or n_nonfraud < 5:
        continue  # 样本太少跳过

    # 模型预测高风险
    pred_high = (grp['risk_score'] >= 0.55).astype(int)
    fraud_high = pred_high[grp['ann_fin_flag'] == 1].sum()
    nonfraud_high = pred_high[grp['ann_fin_flag'] == 0].sum()

    # Recall(TPR) = 真实违规中被标为高风险的比例
    recall_fraud = fraud_high / n_fraud
    # FPR = 真实非违规中被标为高风险的比例(误报率)
    fpr = nonfraud_high / n_nonfraud

    # 与整体基线对比
    risk_mean = grp['risk_score'].mean()
    risk_bias = risk_mean - overall_risk_mean

    p_ml_mean = grp['p_ml'].mean()
    p_ml_bias = p_ml_mean - overall_p_ml_mean

    fraud_rate = n_fraud / n_total
    fraud_rate_bias = fraud_rate - overall_fraud_rate

    bias_rows.append({
        'industry': industry,
        'n_total': n_total,
        'n_fraud': n_fraud,
        'n_nonfraud': n_nonfraud,
        'actual_fraud_rate': fraud_rate,
        'actual_fraud_rate_bias': fraud_rate_bias,
        'recall_high_risk': recall_fraud,
        'fpr_high_risk': fpr,
        'mean_risk_score': risk_mean,
        'risk_score_bias': risk_bias,
        'mean_p_ml': p_ml_mean,
        'p_ml_bias': p_ml_bias,
    })

bias_df = pd.DataFrame(bias_rows).sort_values('risk_score_bias', ascending=False)
bias_df.to_csv(os.path.join(OUT_DIR, 'industry_bias_audit.csv'), index=False)
print(f'\n  → output/industry_bias_audit.csv (共 {len(bias_df)} 个行业)')

# 关键输出
print('\n=== 模型系统性偏强的行业(risk_score 偏高,可能误报多) ===')
top_bias_high = bias_df.head(10)
print(top_bias_high[['industry', 'n_total', 'actual_fraud_rate', 'recall_high_risk',
                     'fpr_high_risk', 'mean_risk_score', 'risk_score_bias']].to_string(index=False))

print('\n=== 模型系统性偏弱的行业(risk_score 偏低,可能漏报) ===')
top_bias_low = bias_df.tail(10)
print(top_bias_low[['industry', 'n_total', 'actual_fraud_rate', 'recall_high_risk',
                    'fpr_high_risk', 'mean_risk_score', 'risk_score_bias']].to_string(index=False))

# 偏差可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 左: risk_score 偏差 vs 实际违规率偏差
ax = axes[0]
colors = ['red' if b > 0.05 else ('blue' if b < -0.05 else 'gray')
          for b in bias_df['risk_score_bias']]
ax.scatter(bias_df['actual_fraud_rate_bias'], bias_df['risk_score_bias'],
           c=colors, alpha=0.7, s=50, edgecolors='white')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('实际违规率偏差(行业-整体)', fontsize=11)
ax.set_ylabel('平均风险分偏差(行业-整体)', fontsize=11)
ax.set_title('行业偏差散点图(红=风险高估,蓝=风险低估)', fontsize=13, fontweight='bold')

# 标注 TOP 偏差行业
for _, row in bias_df.nlargest(3, 'risk_score_bias').iterrows():
    ax.annotate(row['industry'], (row['actual_fraud_rate_bias'], row['risk_score_bias']),
                fontsize=9, alpha=0.8)
for _, row in bias_df.nsmallest(3, 'risk_score_bias').iterrows():
    ax.annotate(row['industry'], (row['actual_fraud_rate_bias'], row['risk_score_bias']),
                fontsize=9, alpha=0.8)

ax.grid(True, alpha=0.3)

# 右: 各行业 Recall(违规命中率)柱状图
ax = axes[1]
sorted_by_recall = bias_df.sort_values('recall_high_risk', ascending=False)
ax.barh(sorted_by_recall['industry'][:15], sorted_by_recall['recall_high_risk'][:15],
        color='steelblue', edgecolor='white')
ax.set_xlabel('Recall(违规公司被标为高风险的比例)', fontsize=11)
ax.set_title('各行业 Recall(TOP 15)', fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'industry_bias_chart.png'), dpi=150, bbox_inches='tight')
print(f'\n  → output/industry_bias_chart.png')

# 总结
print('\n' + '=' * 60)
print('行业偏见检测结论')
print('=' * 60)

n_overestimated = (bias_df['risk_score_bias'] > 0.05).sum()
n_underestimated = (bias_df['risk_score_bias'] < -0.05).sum()
print(f'行业总数: {len(bias_df)}')
print(f'系统性高估(risk_score_bias > 0.05): {n_overestimated} 个行业')
print(f'系统性低估(risk_score_bias < -0.05): {n_underestimated} 个行业')
print(f'偏差较小(|bias| ≤ 0.05): {len(bias_df) - n_overestimated - n_underestimated} 个行业')

# 找最严重的偏见行业
worst_over = bias_df.loc[bias_df['risk_score_bias'].idxmax()]
worst_under = bias_df.loc[bias_df['risk_score_bias'].idxmin()]
print(f'\n最大高估行业: {worst_over["industry"]} (bias=+{worst_over["risk_score_bias"]:.3f})')
print(f'最大低估行业: {worst_under["industry"]} (bias={worst_under["risk_score_bias"]:.3f})')

print('\n✅ 行业偏见检测完成')
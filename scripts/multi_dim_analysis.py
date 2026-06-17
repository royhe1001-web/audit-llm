#!/usr/bin/env python3
"""
阶段三·步骤 4: 多维审计分析
============================
"系统分析 → 类别分析 → 个体分析" 三层递进
产出: 系统仪表盘 PNG + 类别层 Excel + Top200 个体层
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("步骤 4.1: 加载评分数据")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_full.csv"))
print(f"总记录: {len(df)}")
print(f"列: {df.columns.tolist()}")

# ============================================================
# 2. 系统层
# ============================================================
print("\n" + "=" * 60)
print("步骤 4.2: 系统层 - 全样本统计")
print("=" * 60)

sys_stats = {
    '总样本数': len(df),
    '唯一公司数': df['Symbol'].nunique(),
    '唯一行业数': df['industry'].nunique(),
    '年份范围': f"{int(df['feature_year'].min())}-{int(df['feature_year'].max())}",
    '高风险': (df['risk_level'] == '高风险').sum(),
    '中风险': (df['risk_level'] == '中风险').sum(),
    '低风险': (df['risk_level'] == '低风险').sum(),
    '已知违规(ann_fin_flag=1)': (df['ann_fin_flag'] == 1).sum(),
    '平均ML概率': f"{df['p_ml'].mean():.3f}",
    '平均风险分': f"{df['risk_score'].mean():.3f}",
    '规则触发比例': f"{(df['n_rules_triggered'] > 0).mean()*100:.1f}%",
}
print("系统层统计:")
for k, v in sys_stats.items():
    print(f"  {k}: {v}")

# 写报告片段
sys_md = ["## 系统层统计\n",
          "| 指标 | 值 |",
          "|---|---|"]
for k, v in sys_stats.items():
    sys_md.append(f"| {k} | {v} |")
with open(os.path.join(OUT_DIR, "system_layer_stats.md"), 'w') as f:
    f.write('\n'.join(sys_md))
print(f"  → output/system_layer_stats.md")

# ============================================================
# 3. 系统层仪表盘 (4 子图)
# ============================================================
print("\n" + "=" * 60)
print("步骤 4.3: 系统层仪表盘")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 风险等级饼图
counts = df['risk_level'].value_counts()
level_order = ['高风险', '中风险', '低风险']
counts = counts.reindex([l for l in level_order if l in counts.index])
colors_map = {'高风险': '#d62728', '中风险': '#ff7f0e', '低风险': '#2ca02c'}
colors = [colors_map[l] for l in counts.index]
axes[0, 0].pie(counts, labels=counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90,
                textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[0, 0].set_title('风险等级分布', fontweight='bold', fontsize=14)

# 2. 风险分数直方图 + 阈值
axes[0, 1].hist(df['risk_score'], bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0, 1].axvline(0.55, color='red', linestyle='--', linewidth=2, label='高风险阈值=0.55')
axes[0, 1].axvline(0.25, color='orange', linestyle='--', linewidth=2, label='中风险阈值=0.25')
axes[0, 1].set_xlabel('风险分数', fontsize=12)
axes[0, 1].set_ylabel('频数', fontsize=12)
axes[0, 1].set_title('风险分数分布', fontweight='bold', fontsize=14)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. 财务特征相关性
corr = df[FIN_COLS].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0],
             vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.8})
axes[1, 0].set_title('财务特征相关性矩阵', fontweight='bold', fontsize=14)

# 4. 行业风险排名 Top10
ind_risk = df.groupby('industry')['risk_score'].agg(['mean', 'count']).reset_index()
ind_risk = ind_risk[ind_risk['count'] >= 20]  # 至少 20 个样本
top_ind = ind_risk.sort_values('mean', ascending=False).head(10)
axes[1, 1].barh(top_ind['industry'][::-1], top_ind['mean'][::-1], color='indianred', edgecolor='white')
axes[1, 1].set_xlabel('平均风险分', fontsize=12)
axes[1, 1].set_title('行业平均风险排名 (Top 10, n≥20)', fontweight='bold', fontsize=14)
for i, v in enumerate(top_ind['mean'][::-1]):
    axes[1, 1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
axes[1, 1].grid(axis='x', alpha=0.3)

plt.suptitle('审计风险系统层仪表盘', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "system_dashboard.png"), dpi=150, bbox_inches='tight')
print(f"  → output/system_dashboard.png")

# ============================================================
# 4. 类别层 - 行业/年份/地区 切面
# ============================================================
print("\n" + "=" * 60)
print("步骤 4.4: 类别层 - 行业/年份/地区 切面")
print("=" * 60)

# 行业统计
ind_stats = df.groupby('industry').agg(
    n=('Symbol', 'count'),
    n_high=('risk_level', lambda x: (x == '高风险').sum()),
    n_mid=('risk_level', lambda x: (x == '中风险').sum()),
    n_low=('risk_level', lambda x: (x == '低风险').sum()),
    mean_risk=('risk_score', 'mean'),
    mean_ml=('p_ml', 'mean'),
    mean_rules=('rule_score_norm', 'mean'),
    ann_fin_actual=('ann_fin_flag', lambda x: (x == 1).sum()),
).reset_index()
ind_stats['high_rate'] = ind_stats['n_high'] / ind_stats['n']
ind_stats = ind_stats.sort_values('mean_risk', ascending=False)
print(f"  行业数: {len(ind_stats)}")
print(f"  Top 5 高风险行业:\n{ind_stats.head(5)[['industry', 'n', 'mean_risk', 'high_rate']].to_string(index=False)}")

# 年份统计
yr_stats = df.groupby('violation_year').agg(
    n=('Symbol', 'count'),
    mean_risk=('risk_score', 'mean'),
    mean_ml=('p_ml', 'mean'),
    n_high=('risk_level', lambda x: (x == '高风险').sum()),
    ann_fin_actual=('ann_fin_flag', lambda x: (x == 1).sum()),
).reset_index()
yr_stats = yr_stats.sort_values('violation_year')

# 地区统计
area_stats = df.groupby('area').agg(
    n=('Symbol', 'count'),
    mean_risk=('risk_score', 'mean'),
    n_high=('risk_level', lambda x: (x == '高风险').sum()),
).reset_index().sort_values('mean_risk', ascending=False)

# 保存 Excel
xl_path = os.path.join(OUT_DIR, "category_layer_stats.xlsx")
with pd.ExcelWriter(xl_path, engine='openpyxl') as writer:
    ind_stats.to_excel(writer, sheet_name='行业', index=False)
    yr_stats.to_excel(writer, sheet_name='年份', index=False)
    area_stats.to_excel(writer, sheet_name='地区', index=False)
print(f"  → output/category_layer_stats.xlsx (3 Sheet)")

# 单独可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 风险分时间趋势
yr_recent = yr_stats[yr_stats['violation_year'] >= yr_stats['violation_year'].max() - 12]
axes[0].plot(yr_recent['violation_year'], yr_recent['mean_risk'], marker='o', color='steelblue', linewidth=2)
axes[0].fill_between(yr_recent['violation_year'], yr_recent['mean_risk'], alpha=0.3, color='steelblue')
axes[0].set_xlabel('违规年份', fontsize=12)
axes[0].set_ylabel('平均风险分', fontsize=12)
axes[0].set_title('平均风险分时间趋势 (近 12 年)', fontweight='bold', fontsize=14)
axes[0].grid(alpha=0.3)

# 各年高风险/实际违规对比
axes[1].bar(yr_recent['violation_year'] - 0.2, yr_recent['n_high'], width=0.4,
             label='高风险预测数', color='indianred', alpha=0.85)
axes[1].bar(yr_recent['violation_year'] + 0.2, yr_recent['ann_fin_actual'], width=0.4,
             label='实际违规数(ann_fin_flag=1)', color='steelblue', alpha=0.85)
axes[1].set_xlabel('违规年份', fontsize=12)
axes[1].set_ylabel('公司数', fontsize=12)
axes[1].set_title('高风险预测 vs 实际违规', fontweight='bold', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "risk_by_year.png"), dpi=150, bbox_inches='tight')
print(f"  → output/risk_by_year.png")

# ============================================================
# 5. 个体层 - Top200
# ============================================================
print("\n" + "=" * 60)
print("步骤 4.5: 个体层 - Top200 高风险公司")
print("=" * 60)

top = df.sort_values('risk_score', ascending=False).head(500)
top_dedup = top.drop_duplicates(subset=['Symbol']).head(200)
top_dedup.to_csv(os.path.join(OUT_DIR, "top_risk_companies.csv"), index=False)
print(f"  → output/top_risk_companies.csv (200 行,去重后)")
print(f"  Top10:")
print(top_dedup[['ShortName', 'industry', 'violation_year', 'p_ml', 'rule_ids', 'risk_score']]
      .head(10).to_string(index=False))

# ============================================================
# 6. 总结
# ============================================================
print("\n" + "=" * 60)
print("✅ 步骤 4 完成")
print("=" * 60)
print(f"  系统层: 11 项指标 + 4 子图仪表盘")
print(f"  类别层: 行业 {len(ind_stats)} 个, 年份 {len(yr_stats)} 个, 地区 {len(area_stats)} 个")
print(f"  个体层: Top200 高风险公司")
print(f"  产出: output/system_dashboard.png")
print(f"        output/category_layer_stats.xlsx")
print(f"        output/top_risk_companies.csv")
print(f"        output/risk_by_year.png")

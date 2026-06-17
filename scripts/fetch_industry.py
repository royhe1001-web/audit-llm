#!/usr/bin/env python3
"""
阶段三·步骤 1: 补全行业数据
============================
用 tushare stock_basic 单次拉取全市场 industry/area/list_date,
合并到 fraud_features_combined.csv,产出富特征矩阵 + 行业分布图。
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

TOKEN = "f3bf8d32b09bb2cfa3f3632b5521caa8143576c3dff550a742f5f4cc"
ts.set_token(TOKEN)
pro = ts.pro_api()

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

# ============================================================
# 1. 拉取全市场行业/地区
# ============================================================
print("=" * 60)
print("步骤 1.1: 拉取全市场 stock_basic")
print("=" * 60)

basic_l = pro.stock_basic(
    list_status='L',
    fields='ts_code,symbol,name,industry,area,list_date'
)
basic_d = pro.stock_basic(
    list_status='D',
    fields='ts_code,symbol,name,industry,area,list_date'
)
basic_p = pro.stock_basic(
    list_status='P',
    fields='ts_code,symbol,name,industry,area,list_date'
)
basic = pd.concat([basic_l, basic_d, basic_p], ignore_index=True).drop_duplicates(subset='symbol')
print(f"在市: {len(basic_l)}, 退市: {len(basic_d)}, 暂停: {len(basic_p)}")
print(f"去重后: {len(basic)} 家")
print(f"行业字段缺失: {basic['industry'].isna().sum()}")
print(f"地区字段缺失: {basic['area'].isna().sum()}")

basic['Symbol'] = basic['symbol'].astype(str).str.zfill(6)
basic = basic[['Symbol', 'ts_code', 'name', 'industry', 'area', 'list_date']]
basic = basic.drop_duplicates(subset='Symbol', keep='first')
basic.to_csv(os.path.join(DATA_DIR, "industry_mapping.csv"), index=False)
print(f"  → industry_mapping.csv ({os.path.getsize(os.path.join(DATA_DIR, 'industry_mapping.csv'))/1024:.0f} KB)")

# ============================================================
# 2. 合并到特征矩阵
# ============================================================
print("\n" + "=" * 60)
print("步骤 1.2: 合并到 fraud_features_combined.csv")
print("=" * 60)

feat = pd.read_csv(os.path.join(DATA_DIR, "fraud_features_combined.csv"))
feat['Symbol'] = feat['Symbol'].astype(str)
# 补齐 6 位 (e.g. '966' → '000966')
feat['Symbol'] = feat['Symbol'].str.zfill(6)
print(f"原特征矩阵: {feat.shape}")

# Symbol 是 6 位代码 (e.g. "000001"), tushare symbol 也是 6 位, 直接 merge
enriched = feat.merge(basic, on='Symbol', how='left')
print(f"合并后: {enriched.shape}")

# 缺失填充
enriched['industry'] = enriched['industry'].fillna('未分类')
enriched['area'] = enriched['area'].fillna('未知')
enriched['list_date'] = enriched['list_date'].fillna('')

coverage = (enriched['industry'] != '未分类').mean() * 100
print(f"行业覆盖率: {coverage:.1f}%")
print(f"  唯一行业数: {enriched['industry'].nunique()}")
print(f"  唯一地区数: {enriched['area'].nunique()}")

enriched.to_csv(os.path.join(DATA_DIR, "fraud_features_enriched.csv"), index=False)
print(f"  → fraud_features_enriched.csv ({os.path.getsize(os.path.join(DATA_DIR, 'fraud_features_enriched.csv'))/1024:.0f} KB)")

# ============================================================
# 3. 行业分布可视化
# ============================================================
print("\n" + "=" * 60)
print("步骤 1.3: 行业分布可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 左: 全样本行业 Top15
top_all = enriched['industry'].value_counts().head(15)
axes[0].barh(top_all.index[::-1], top_all.values[::-1], color='steelblue', edgecolor='white')
axes[0].set_xlabel('公司数', fontsize=11)
axes[0].set_title('全样本行业分布 (Top 15)', fontweight='bold', fontsize=13)
axes[0].grid(axis='x', alpha=0.3)
for i, v in enumerate(top_all.values[::-1]):
    axes[0].text(v + 5, i, str(v), va='center', fontsize=9)

# 右: ann_fin_flag=1 子集
sub = enriched[enriched['ann_fin_flag'] == 1]
top_fraud = sub['industry'].value_counts().head(15)
axes[1].barh(top_fraud.index[::-1], top_fraud.values[::-1], color='indianred', edgecolor='white')
axes[1].set_xlabel('违规公司数', fontsize=11)
axes[1].set_title('财务违规公司行业分布 (Top 15)', fontweight='bold', fontsize=13)
axes[1].grid(axis='x', alpha=0.3)
for i, v in enumerate(top_fraud.values[::-1]):
    axes[1].text(v + 0.5, i, str(v), va='center', fontsize=9)

plt.suptitle(f'行业分布对比 (行业覆盖率 {coverage:.1f}%)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "industry_distribution.png"), dpi=150, bbox_inches='tight')
print(f"  → output/industry_distribution.png")

# 行业-年份 热力图(违规事件)
yr_ind = sub.groupby(['violation_year', 'industry']).size().unstack(fill_value=0)
top10_ind = sub['industry'].value_counts().head(10).index
yr_top = yr_ind[top10_ind] if all(c in yr_ind.columns for c in top10_ind) else yr_ind[yr_ind.sum().nlargest(10).index]
# 取近 15 年
yr_recent = yr_top.loc[yr_top.index >= (yr_top.index.max() - 15)]

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(yr_recent.T.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(yr_recent.index)))
ax.set_xticklabels(yr_recent.index, rotation=45, ha='right')
ax.set_yticks(range(len(yr_recent.columns)))
ax.set_yticklabels(yr_recent.columns)
ax.set_xlabel('违规年份', fontsize=11)
ax.set_title('行业-年份 财务违规事件热力图 (近15年)', fontweight='bold', fontsize=13)
plt.colorbar(im, ax=ax, label='事件数')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "industry_year_heatmap.png"), dpi=150, bbox_inches='tight')
print(f"  → output/industry_year_heatmap.png")

# ============================================================
# 4. 总结
# ============================================================
print("\n" + "=" * 60)
print("✅ 步骤 1 完成")
print("=" * 60)
print(f"  行业覆盖率: {coverage:.1f}%")
print(f"  富特征矩阵: {enriched.shape[0]} 行 × {enriched.shape[1]} 列")
print(f"  新增列: industry, area, list_date (+ ts_code, name)")
print(f"  Top 5 行业: {', '.join(top_all.head(5).index.tolist())}")

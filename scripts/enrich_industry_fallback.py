#!/usr/bin/env python3
"""
阶段四·优化 2: 行业覆盖率补全(板块级兜底)
============================================
未分类公司分两类:
  A. 在 tushare stock_basic 中但 industry 字段为空(258 家)
  B. Symbol 都不在 industry_mapping 中(994 家)
这两类共同占 1,252/7,997 = 15.7%

解决方案:用证券代码前缀做板块级兜底分类
  60xxxx / 601xxx / 603xxx / 605xxx → 沪市主板
  600xxx / 601xxx → 沪市主板(同上)
  000xxx / 001xxx / 002xxx / 003xxx → 深市主板/中小板
  300xxx / 301xxx → 创业板
  688xxx / 689xxx → 科创板
  4xxxxx / 8xxxxx → 北交所
  其他 → 兜底"其他板块"
"""

import os
import pandas as pd

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")


def fallback_industry(symbol: str) -> str:
    """根据证券代码前缀推断板块级行业"""
    s = str(symbol).zfill(6)
    if s.startswith(('600', '601', '603', '605')):
        return '沪市主板'
    if s.startswith(('000', '001', '002', '003')):
        return '深市主板'
    if s.startswith(('300', '301')):
        return '创业板'
    if s.startswith(('688', '689')):
        return '科创板'
    if s.startswith(('4', '83', '87', '88')):
        return '北交所'
    if s.startswith(('5', '9')):
        return '其他板块'
    return '其他板块'


print('=' * 60)
print('行业覆盖率补全:板块级兜底')
print('=' * 60)

# 加载中间表
df = pd.read_csv(os.path.join(DATA_DIR, 'audit_intermediate_table.csv'))
n_before = (df['industry'] == '未分类').sum()
print(f'补全前:未分类 = {n_before} / {len(df)} = {n_before/len(df)*100:.1f}%')

# 应用兜底分类
mask = (df['industry'] == '未分类') | (df['industry'].isna())
df.loc[mask, 'industry'] = df.loc[mask, 'Symbol'].astype(str).str.zfill(6).apply(fallback_industry)

# 验证
n_after_unclassified = (df['industry'] == '未分类').sum()
n_other = (df['industry'] == '其他板块').sum()
print(f'补全后:未分类 = {n_after_unclassified}, 其他板块 = {n_other}')
print(f'有效行业覆盖率: {(len(df) - n_after_unclassified - n_other) / len(df) * 100:.2f}%')

# 板块级分布
print('\n=== 板块级行业分布 ===')
print(df['industry'].value_counts().head(15))

# 保存
out_path = os.path.join(DATA_DIR, 'audit_intermediate_table.csv')
df.to_csv(out_path, index=False)
print(f'\n  → {out_path}')

# 同步更新行业风险排名表
print('\n=== 重新计算行业风险排名 ===')
if 'risk_score' in df.columns:
    industry_stats = df.groupby('industry').agg(
        n_companies=('Symbol', 'count'),
        mean_risk=('risk_score', 'mean'),
        n_high_risk=('is_high_risk', 'sum'),
        n_anomaly=('is_anomaly', 'sum'),
        n_fraud=('ann_fin_flag', lambda x: (x == 1).sum()),
    ).sort_values('mean_risk', ascending=False)
    industry_stats.to_csv(os.path.join(OUT_DIR, 'industry_risk_ranking_v2.csv'))
    print(f'  → output/industry_risk_ranking_v2.csv (含板块兜底)')
    print(f'\n=== 含板块兜底的行业风险 TOP 15 ===')
    print(industry_stats.head(15).to_string())
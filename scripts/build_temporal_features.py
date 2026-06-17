#!/usr/bin/env python3
"""
时序特征工程 — 任务 1: 构建 (Symbol, year) 财务面板
=====================================================
输入: data/raw_financials.csv
输出: data/temporal_panel.csv (Symbol × Year × 7 比率,含 is_imputed 标记)

策略:不用 pivot,直接用 melt + groupby 对每个 Symbol 的 7 比率做线性插值
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")

RATIO_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio',
              'asset_turnover', 'net_margin', 'ocf_to_rev']

print('=' * 60)
print('时序面板构建 (线性插值版)')
print('=' * 60)

# 1. 加载并 melt
raw = pd.read_csv(os.path.join(DATA_DIR, 'raw_financials.csv'))
raw['Symbol'] = raw['fetch_symbol'].astype(str).str.zfill(6)
raw['year'] = raw['fetch_year'].astype(int)
print(f'原始: {len(raw)} 行, {raw["Symbol"].nunique()} Symbol, {raw["year"].nunique()} 年份')

# 2. melt 到长格式
long = raw.melt(
    id_vars=['Symbol', 'year'],
    value_vars=RATIO_COLS,
    var_name='ratio',
    value_name='raw_value'
)
print(f'melt 后: {len(long)} 行')
print(f'原始 NaN: {long["raw_value"].isna().sum()} / {len(long)} ({long["raw_value"].isna().mean()*100:.1f}%)')

# 3. 对每个 (Symbol, ratio) 做线性插值
def interp_group(g):
    g = g.sort_values('year').copy()
    g['value'] = g['raw_value'].interpolate(method='linear', limit_direction='both')
    g['is_imputed'] = g['raw_value'].isna() & g['value'].notna()
    return g

print('\n开始插值(对每个 Symbol × Ratio)...')
interp = long.groupby(['Symbol', 'ratio'], group_keys=False).apply(interp_group)
print(f'插值完成')

# 4. 统计
print(f'\n=== 插值效果 ===')
print(f'原始 NaN: {interp["raw_value"].isna().sum()}')
print(f'插值后 NaN: {interp["value"].isna().sum()}')
print(f'插值填补数: {interp["is_imputed"].sum()}')
print(f'填补率: {interp["is_imputed"].sum() / long["raw_value"].isna().sum() * 100:.1f}%')

# 5. 保存
out_path = os.path.join(DATA_DIR, 'temporal_panel.csv')
interp[['Symbol', 'year', 'ratio', 'value', 'is_imputed']].to_csv(out_path, index=False)
print(f'\n  → {out_path}')
print(f'  行数: {len(interp)}, Symbol × Year × Ratio 组合')

# 6. 同时保存"宽表"版(便于后续脚本快速查找)
panel_wide = interp.pivot_table(
    index=['Symbol', 'year'], columns='ratio', values='value'
).reset_index()
panel_wide.columns.name = None
wide_path = os.path.join(DATA_DIR, 'temporal_panel_wide.csv')
panel_wide.to_csv(wide_path, index=False)
print(f'  → {wide_path} ({len(panel_wide)} 行 × {len(panel_wide.columns)} 列)')

# 7. 诊断:每年有多少 Symbol 有完整 7 比率数据
panel_wide['n_ratios'] = panel_wide[RATIO_COLS].notna().sum(axis=1)
print('\n=== 各年份 Symbol 完整度 ===')
year_quality = panel_wide.groupby('year').agg(
    n_symbols=('Symbol', 'count'),
    n_complete=('n_ratios', lambda x: (x == 7).sum()),
    n_partial=('n_ratios', lambda x: ((x > 0) & (x < 7)).sum()),
)
print(year_quality.head(25))

print('\n✅ 时序面板构建完成')
#!/usr/bin/env python3
"""
时序特征工程 — 任务 4: 留痕表 + 仪表盘数据更新
================================================
- 把时序特征(关键 5 维)加入 audit_score_traceability.csv,产出 v2 版
- 保留 v1 版作对照
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

print('=' * 60)
print('留痕表 v2 — 加入时序关键特征')
print('=' * 60)

# 1. 加载 v1 留痕表
trace_v1 = pd.read_csv(os.path.join(OUT_DIR, 'audit_score_traceability.csv'))
print(f'v1 留痕表: {trace_v1.shape}')

# 2. 加载时序关键特征(选最有代表性的 5 维)
# 优先选 t-1 值 + 3 年 CAGR(两者覆盖度最高)+ yoy
temporal = pd.read_csv(os.path.join(DATA_DIR, 'temporal_features_v2.csv'))
print(f'时序特征: {temporal.shape}')

# 3. 选择关键时序列(7 比率 × t-1 值 + 7 比率 × 3y CAGR + 7 比率 × yoy)
key_temporal_cols = (
    [f'{r}_t1' for r in ['roe', 'roa', 'debt_ratio']] +
    [f'{r}_yoy' for r in ['roe', 'roa', 'debt_ratio', 'ocf_to_rev']] +
    [f'{r}_cagr3y' for r in ['roe', 'net_margin']]
)
key_temporal = temporal[['Symbol', 'violation_year'] + key_temporal_cols].copy()

# 4. 合并到 v1
trace_v2 = trace_v1.merge(key_temporal, on=['Symbol', 'violation_year'], how='left')

# 5. 也把"是否时序增强"(至少有 1 个时序特征非空)加入
trace_v2['has_temporal_data'] = trace_v2[key_temporal_cols].notna().any(axis=1)

# 6. 保存
out_path = os.path.join(OUT_DIR, 'audit_score_traceability_v2.csv')
trace_v2.to_csv(out_path, index=False)
print(f'\n  → {out_path}')
print(f'  列数: {trace_v1.shape[1]} → {trace_v2.shape[1]} (+{trace_v2.shape[1]-trace_v1.shape[1]} 列)')

# 7. TOP100 v2
top100_v2 = trace_v2.head(100).copy()
top100_v2_path = os.path.join(OUT_DIR, 'audit_top100_traceability_v2.csv')
top100_v2.to_csv(top100_v2_path, index=False)
print(f'  → {top100_v2_path}')

# 8. 统计:TOP100 中有时序数据的公司比例
n_with_temporal = top100_v2['has_temporal_data'].sum()
print(f'\n  TOP100 中有时序数据的公司: {n_with_temporal} ({n_with_temporal/100*100:.0f}%)')

# 9. 同时把时序特征添加到中间表(便于仪表盘读取)
df_inter = pd.read_csv(os.path.join(DATA_DIR, 'audit_intermediate_table.csv'))
df_inter_v2 = df_inter.merge(key_temporal, on=['Symbol', 'violation_year'], how='left')
df_inter_v2['has_temporal_data'] = df_inter_v2[key_temporal_cols].notna().any(axis=1)
inter_path = os.path.join(DATA_DIR, 'audit_intermediate_table_v2.csv')
df_inter_v2.to_csv(inter_path, index=False)
print(f'\n  → {inter_path} ({df_inter_v2.shape[1]} 列)')

print('\n✅ 留痕表 v2 完成')
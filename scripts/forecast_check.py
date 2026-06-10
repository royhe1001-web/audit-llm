#!/usr/bin/env python3
"""
P3-3: 业绩预告 vs 实际对比
=============================
从 tushare forecast 接口拉业绩预告,对比实际财报:
- 预告 "增" 但实际"降" → 变脸 (强信号)
- 预告 "盈利" 但实际"亏损" → 强信号
- 预告 "亏损" 但实际"盈利" → 也异常
- 预告方向 vs 实际方向相反
"""

import os, time, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")

TOKEN = "f3bf8d32b09bb2cfa3f3632b5521caa8143576c3dff550a742f5f4cc"
ts.set_token(TOKEN)
pro = ts.pro_api()


def assign_level(s):
    if s >= 0.55: return '高风险'
    if s >= 0.25: return '中风险'
    return '低风险'


# ============================================================
# 1. 加载风险评分
# ============================================================
print("=" * 60)
print("P3-3.1: 加载风险评分")
print("=" * 60)

risk = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_quality.csv"))
risk['Symbol'] = risk['Symbol'].astype(str).str.zfill(6)
print(f"  风险评分: {risk.shape}")

# ============================================================
# 2. 拉取所有公司业绩预告
# ============================================================
print("\n" + "=" * 60)
print("P3-3.2: 拉取业绩预告 (2024-2025)")
print("=" * 60)

# 取 unique Symbol,去重
symbols = risk['Symbol'].unique()
print(f"  待拉公司数: {len(symbols)}")

forecasts = []
err_count = 0
for i, sym in enumerate(symbols):
    if i % 100 == 0:
        print(f"    进度: {i}/{len(symbols)}")
    try:
        ts_code = f"{sym}.{'SH' if sym.startswith('6') else 'SZ'}"
        f = pro.forecast(ts_code=ts_code, start_date='20240101', end_date='20260606',
                         fields='ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max')
        if f is not None and not f.empty:
            f['Symbol'] = sym
            forecasts.append(f)
        time.sleep(0.05)
    except Exception as e:
        err_count += 1
        if err_count > 50:
            print(f"  ⚠️ 错误过多,停止")
            break

if forecasts:
    fc = pd.concat(forecasts, ignore_index=True)
    print(f"  拉取到 {len(fc)} 条业绩预告")
    fc.to_csv(os.path.join(DATA_DIR, 'forecast_raw.csv'), index=False)
    print(f"  → data/forecast_raw.csv")
else:
    fc = pd.DataFrame()
    print("  无业绩预告数据")

# ============================================================
# 3. 对比业绩预告 vs 实际
# ============================================================
print("\n" + "=" * 60)
print("P3-3.3: 对比预告 vs 实际")
print("=" * 60)

if len(fc) == 0:
    print("  无数据,跳过")
else:
    # 把预告 p_change 中位数和 type 转成"方向"
    # type: 预增/略增/续盈/扭亏/预减/略减/续亏/首亏/不确定
    pos_types = ['预增', '略增', '续盈', '扭亏']
    neg_types = ['预减', '略减', '续亏', '首亏']

    def fc_direction(row):
        p_mid = (row.get('p_change_min', 0) + row.get('p_change_max', 0)) / 2 if pd.notna(row.get('p_change_min')) and pd.notna(row.get('p_change_max')) else 0
        t = str(row.get('type', ''))
        if t in pos_types: return 'positive'
        if t in neg_types: return 'negative'
        if p_mid > 0: return 'positive'
        if p_mid < 0: return 'negative'
        return 'unknown'

    fc['fc_direction'] = fc.apply(fc_direction, axis=1)
    fc['end_year'] = pd.to_datetime(fc['end_date']).dt.year

    # 对每家公司每年的实际业绩方向
    risk['actual_direction'] = risk.apply(
        lambda r: 'positive' if pd.notna(r.get('net_margin')) and r['net_margin'] > 0 else
                  ('negative' if pd.notna(r.get('net_margin')) and r['net_margin'] < 0 else 'unknown'),
        axis=1
    )
    risk['viol_year'] = risk['violation_year'].astype(int)

    # 合并
    fc['viol_year'] = fc['end_year'].astype(int)
    merged = risk.merge(
        fc[['Symbol', 'viol_year', 'type', 'p_change_min', 'p_change_max', 'fc_direction']],
        on=['Symbol', 'viol_year'], how='left'
    )

    # 变脸检测
    def detect_face_change(row):
        fc_dir = row.get('fc_direction')
        act_dir = row.get('actual_direction')
        if pd.isna(fc_dir) or fc_dir == 'unknown': return None
        if pd.isna(act_dir) or act_dir == 'unknown': return None
        if fc_dir != act_dir:
            return f'变脸: 预告 {fc_dir} → 实际 {act_dir}'
        return None

    merged['face_change'] = merged.apply(detect_face_change, axis=1)
    face_change_count = merged['face_change'].notna().sum()
    print(f"  业绩变脸记录: {face_change_count} 条")

    # 应用硬规则
    risk['risk_score_v6'] = risk['risk_score_v5'].copy()
    n_change = 0
    if face_change_count > 0:
        mask = merged['face_change'].notna()
        # 找到这些行的 risk_score 索引
        change_indices = merged[mask].index
        risk.loc[change_indices, 'risk_score_v6'] = risk.loc[change_indices, 'risk_score_v6'].clip(lower=0.6)
        n_change = len(change_indices)
        print(f"  R32 业绩变脸: 触发 {n_change} 条, 风险分下限 0.6")

    # 重新分配等级
    risk['risk_level_v6'] = risk['risk_score_v6'].apply(assign_level)

    # ============================================================
    # 4. 验证
    # ============================================================
    print("\n" + "=" * 60)
    print("P3-3.4: 验证")
    print("=" * 60)

    actual_fraud = risk['ann_fin_flag'] == 1
    for col, label in [('risk_level', 'v0'), ('risk_level_v4', 'v4'),
                        ('risk_level_v5', 'v5'), ('risk_level_v6', 'v6')]:
        hit = (risk.loc[actual_fraud, col] == '高风险').sum() / actual_fraud.sum() * 100
        print(f"  {label}: 高风险命中率 = {hit:.1f}%")

# ============================================================
# 5. 保存
# ============================================================
print("\n" + "=" * 60)
print("P3-3.5: 保存")
print("=" * 60)

risk.to_csv(os.path.join(DATA_DIR, "risk_scored_forecast.csv"), index=False)
print(f"  → data/risk_scored_forecast.csv")

print("\n" + "=" * 60)
print("✅ P3-3 完成: 业绩预告 vs 实际")
print("=" * 60)

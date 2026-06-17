#!/usr/bin/env python3
"""
P3-7: 财报披露延迟检测
=======================
正常 A 股年报披露截止日 4 月 30 日
- 4 月 30 日后披露 = 信号(财务有问题/需要时间)
- 5/6/7 月披露 = 强信号
- 配合"审计师变更"或"前期差错更正"= 红旗

数据源:
- tushare 不直接提供"披露日期"
- 但可以从年报第一页"披露日期"提取
- 简化方案:用报告期结束日 + 审计师签字日 估算
"""

import os, warnings, re
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pdfplumber
import tushare as ts
from datetime import datetime, timedelta

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
# 1. 拉取年报实际披露日期
# ============================================================
print("=" * 60)
print("P3-7.1: 拉取年报披露日期")
print("=" * 60)

# 简化:从 tushare 公告接口拉取 "年度报告" 公告的发布日期
# 取最近 2 年(2023 + 2024)的年报公告
# 这是真实数据,不是 PDF 提取

disclosure = []
err_count = 0
risk = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_industry_v2.csv"))
risk['Symbol'] = risk['Symbol'].astype(str).str.zfill(6)
symbols = risk['Symbol'].unique()[:500]  # 测试 500 家避免限流

for i, sym in enumerate(symbols):
    if i % 100 == 0:
        print(f"    进度: {i}/{len(symbols)}")
    try:
        ts_code = f"{sym}.{'SH' if sym.startswith('6') else 'SZ'}"
        # 找年报披露日
        ann = pro.anns(ts_code=ts_code,
                       start_date='20240101', end_date='20251231',
                       fields='ts_code,ann_date,title')
        if ann is None or ann.empty:
            continue
        # 找年度报告
        mask = ann['title'].str.contains('年度报告|年报', na=False) & \
               ~ann['title'].str.contains('摘要|英文|取消|修订|更新|备查|法律意见', na=False)
        if mask.any():
            yr_anns = ann[mask]
            for _, row in yr_anns.iterrows():
                ann_date = pd.to_datetime(row['ann_date'])
                # 推断报告期
                if ann_date.month <= 6:
                    year = ann_date.year - 1
                else:
                    year = ann_date.year
                disclosure.append({
                    'Symbol': sym,
                    'ann_date': row['ann_date'],
                    'report_year': year,
                    'days_after_april30': (ann_date - pd.Timestamp(f'{year}-04-30')).days,
                })
    except Exception as e:
        err_count += 1
        if err_count > 100:
            print(f"  ⚠️ 错误过多")
            break

if disclosure:
    disc = pd.DataFrame(disclosure)
    print(f"  拉取到 {len(disc)} 条年报披露记录")
    disc.to_csv(os.path.join(DATA_DIR, "disclosure_dates.csv"), index=False)

    # ============================================================
    # 2. 延迟检测
    # ============================================================
    print("\n" + "=" * 60)
    print("P3-7.2: 延迟检测")
    print("=" * 60)

    # 正常披露: ≤ 4/30
    # 轻微延迟: 5/1 - 5/31
    # 重大延迟: ≥ 6/1
    disc['delay_level'] = pd.cut(
        disc['days_after_april30'],
        bins=[-100, 0, 30, 60, 1000],
        labels=['正常', '轻微延迟', '重大延迟', '异常延迟']
    )
    print("  披露延迟分布:")
    print(disc['delay_level'].value_counts())

    # ============================================================
    # 3. 应用硬规则
    # ============================================================
    print("\n" + "=" * 60)
    print("P3-7.3: 硬规则")
    print("=" * 60)

    # 把 delay 合并到风险评分(只对 (Symbol, report_year) 唯一)
    disc_latest = disc.sort_values('ann_date').drop_duplicates(subset=['Symbol', 'report_year'], keep='last')

    # 合并到 risk
    risk['viol_year'] = risk['violation_year'].astype(int)
    disc_latest['viol_year'] = disc_latest['report_year'].astype(int)

    # 合并,报告年应该是 violation_year - 1 或同年
    disc_to_risk = disc_latest.copy()
    disc_to_risk['viol_year'] = disc_to_risk['report_year'] + 1  # 报告年 → 违规年映射

    risk = risk.merge(
        disc_to_risk[['Symbol', 'viol_year', 'ann_date', 'days_after_april30', 'delay_level']],
        on=['Symbol', 'viol_year'], how='left'
    )
    print(f"  合并后: {risk.shape}")

    risk['risk_score_v8'] = risk['risk_score_v7'].copy()
    adjustments = []

    # R33: 重大延迟(>6月1日) → 风险分下限 0.6
    mask = (risk['delay_level'] == '重大延迟')
    n = mask.sum()
    risk.loc[mask, 'risk_score_v8'] = risk.loc[mask, 'risk_score_v8'].clip(lower=0.6)
    adjustments.append(('R33 重大延迟披露(>6/1)', n, 0.6))

    # R34: 异常延迟(>7月) → 风险分下限 0.7
    mask = (risk['delay_level'] == '异常延迟')
    n = mask.sum()
    risk.loc[mask, 'risk_score_v8'] = risk.loc[mask, 'risk_score_v8'].clip(lower=0.7)
    adjustments.append(('R34 异常延迟披露(>7/1)', n, 0.7))

    for name, n, thresh in adjustments:
        print(f"  {name}: 触发 {n} 条, 风险分下限 {thresh}")

    risk['risk_level_v8'] = risk['risk_score_v8'].apply(assign_level)

    # ============================================================
    # 4. 验证
    # ============================================================
    print("\n" + "=" * 60)
    print("P3-7.4: 验证")
    print("=" * 60)

    actual_fraud = risk['ann_fin_flag'] == 1
    for col, label in [('risk_level', 'v0'), ('risk_level_v4', 'v4'),
                        ('risk_level_v7', 'v7'), ('risk_level_v8', 'v8')]:
        hit = (risk.loc[actual_fraud, col] == '高风险').sum() / actual_fraud.sum() * 100
        print(f"  {label}: 高风险命中率 = {hit:.1f}%")

    # ============================================================
    # 5. 保存
    # ============================================================
    print("\n" + "=" * 60)
    print("P3-7.5: 保存")
    print("=" * 60)

    risk.to_csv(os.path.join(DATA_DIR, "risk_scored_disclosure.csv"), index=False)
    print(f"  → data/risk_scored_disclosure.csv")
else:
    print("  无数据")

print("\n" + "=" * 60)
print("✅ P3-7 完成: 财报披露延迟")
print("=" * 60)

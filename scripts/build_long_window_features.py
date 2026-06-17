#!/usr/bin/env python3
"""
时序 v2 — 任务 4: 长窗口累计信号(5-10 年)
=========================================
新增特征:
  - 5 年 CAGR、10 年 CAGR(7 比率 × 2 = 14 维)
  - 累计亏损年数(7 比率 × 1 = 7 维)
  - 累计异常波动率(7 比率 × 1 = 7 维)
  - 累计触发规则次数(综合指标)
  - 趋势长期持续性(5 年窗口线性回归斜率 × 7)

输出: data/long_window_features.csv
然后用 XGBoost 重训,与短窗口对比
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

RATIO_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio',
              'asset_turnover', 'net_margin', 'ocf_to_rev']

print('=' * 60)
print('长窗口累计信号(5-10 年)')
print('=' * 60)

# 1. 加载
panel = pd.read_csv(os.path.join(DATA_DIR, 'temporal_panel_wide.csv'))
panel['Symbol'] = panel['Symbol'].astype(str).str.zfill(6)
panel['year'] = panel['year'].astype(int)
print(f'面板: {len(panel)} 行 ({panel["Symbol"].nunique()} Symbol × {panel["year"].nunique()} 年)')

feat = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_combined.csv'))
feat['Symbol'] = feat['Symbol'].astype(str).str.zfill(6)
feat['violation_year'] = feat['violation_year'].astype(int)
print(f'基准样本: {len(feat)}')

# 2. 索引
panel_idx = panel.set_index(['Symbol', 'year'])

def get_window(symbol, end_year, n_years):
    """获取 (symbol, end_year-n_years+1..end_year) 的 7 比率窗口"""
    seq = []
    for i in range(n_years):
        y = end_year - n_years + 1 + i
        row = []
        for r in RATIO_COLS:
            try:
                v = panel_idx.loc[(symbol, y), r]
                row.append(v if pd.notna(v) else np.nan)
            except KeyError:
                row.append(np.nan)
        seq.append(row)
    return np.array(seq)  # (n_years, 7)

# 3. 计算长窗口特征
print('\n计算长窗口特征(5 年/10 年)...')

rows = []
for _, row in feat.iterrows():
    symbol = row['Symbol']
    vy = row['violation_year']
    feat_row = {'Symbol': symbol, 'violation_year': vy}

    # === 5 年窗口 ===
    win5 = get_window(symbol, vy - 1, 5)
    # === 10 年窗口 ===
    win10 = get_window(symbol, vy - 1, 10)

    for r_idx, r in enumerate(RATIO_COLS):
        # 5 年 CAGR
        seq5 = win5[:, r_idx]
        valid5 = seq5[~np.isnan(seq5)]
        if len(valid5) >= 3 and valid5[0] * valid5[-1] > 0 and abs(valid5[0]) > 1e-6:
            n = len(valid5) - 1
            try:
                feat_row[f'{r}_cagr5y'] = ((abs(valid5[-1]) / abs(valid5[0])) ** (1 / n) - 1) * np.sign(valid5[0])
            except (ValueError, ZeroDivisionError):
                feat_row[f'{r}_cagr5y'] = np.nan
        else:
            feat_row[f'{r}_cagr5y'] = np.nan

        # 10 年 CAGR
        seq10 = win10[:, r_idx]
        valid10 = seq10[~np.isnan(seq10)]
        if len(valid10) >= 4 and valid10[0] * valid10[-1] > 0 and abs(valid10[0]) > 1e-6:
            n = len(valid10) - 1
            try:
                feat_row[f'{r}_cagr10y'] = ((abs(valid10[-1]) / abs(valid10[0])) ** (1 / n) - 1) * np.sign(valid10[0])
            except (ValueError, ZeroDivisionError):
                feat_row[f'{r}_cagr10y'] = np.nan
        else:
            feat_row[f'{r}_cagr10y'] = np.nan

        # 累计亏损年数(net_margin < 0 或 roe < 0)
        neg_count = (valid5 < 0).sum() if len(valid5) > 0 else 0
        if r in ('roe', 'net_margin', 'roa'):
            feat_row[f'{r}_loss_years_5y'] = neg_count

        # 累计波动率(5 年标准差)
        if len(valid5) >= 2:
            feat_row[f'{r}_volatility_5y'] = np.nanstd(valid5)
        else:
            feat_row[f'{r}_volatility_5y'] = np.nan

        # 趋势长期持续性(5 年线性回归斜率)
        if len(valid5) >= 3:
            xs = np.arange(len(valid5))
            slope = np.polyfit(xs, valid5, 1)[0]
            feat_row[f'{r}_slope_5y'] = slope
        else:
            feat_row[f'{r}_slope_5y'] = np.nan

    # 综合累计指标
    feat_row['total_loss_years_5y'] = sum(feat_row.get(f'{r}_loss_years_5y', 0) for r in ['roe', 'net_margin', 'roa'])
    feat_row['mean_volatility_5y'] = np.nanmean([feat_row[f'{r}_volatility_5y'] for r in RATIO_COLS])

    # 累计高负债年数(debt_ratio > 0.7)
    debt5 = [win5[i, RATIO_COLS.index('debt_ratio')] for i in range(5)]
    debt_valid = [v for v in debt5 if not np.isnan(v)]
    feat_row['high_leverage_years_5y'] = sum(1 for v in debt_valid if v > 0.7)

    # 累计流动性紧张年数(current_ratio < 1)
    curr5 = [win5[i, RATIO_COLS.index('current_ratio')] for i in range(5)]
    curr_valid = [v for v in curr5 if not np.isnan(v)]
    feat_row['liquidity_stress_years_5y'] = sum(1 for v in curr_valid if v < 1)

    rows.append(feat_row)

lw_df = pd.DataFrame(rows)
print(f'生成 {len(lw_df)} 条长窗口特征 × {len(lw_df.columns)} 列')

# 4. 覆盖度诊断
non_id_cols = [c for c in lw_df.columns if c not in ['Symbol', 'violation_year']]
coverage = lw_df[non_id_cols].notna().mean().sort_values(ascending=False)
print(f'\n=== 长窗口特征覆盖度 TOP 10 ===')
print(coverage.head(10).to_string())
print(f'\n=== 长窗口特征覆盖度 BOTTOM 10 ===')
print(coverage.tail(10).to_string())

# 5. 保存
out_path = os.path.join(DATA_DIR, 'long_window_features.csv')
lw_df.to_csv(out_path, index=False)
print(f'\n  → {out_path}')

# 6. 合并到主表
combined = feat.merge(lw_df, on=['Symbol', 'violation_year'], how='left')
combined_path = os.path.join(DATA_DIR, 'fraud_features_with_long_window.csv')
combined.to_csv(combined_path, index=False)
print(f'  → {combined_path} ({len(combined.columns)} 列)')

new_cols = [c for c in combined.columns if c not in feat.columns]
print(f'\n新增长窗口特征: {len(new_cols)} 列')
print(f'示例: {new_cols[:5]}...{new_cols[-5:]}')

print('\n✅ 长窗口累计信号构建完成')
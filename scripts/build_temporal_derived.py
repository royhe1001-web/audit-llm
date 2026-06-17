#!/usr/bin/env python3
"""
时序特征工程 — 任务 2: 时序衍生特征构建
==========================================
输入: data/temporal_panel_wide.csv + data/fraud_features_combined.csv
输出: data/temporal_features_v2.csv

为每个 (Symbol, violation_year) 提取:
1. t-1, t-2, t-3 年的 7 比率值(共 21 维)
2. yoy 增长率(当年 vs 上一年,共 7 维)
3. 3 年 CAGR(共 7 维)
4. 3 年波动率(标准差,共 7 维)
5. 趋势方向(连续上升/下降,共 7 维)
6. 趋势加速度(yoy 的 yoy,共 7 维)

合计约 56 维时序特征
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
print('时序衍生特征构建')
print('=' * 60)

# 1. 加载面板(宽表)
panel = pd.read_csv(os.path.join(DATA_DIR, 'temporal_panel_wide.csv'))
panel['Symbol'] = panel['Symbol'].astype(str).str.zfill(6)
panel['year'] = panel['year'].astype(int)
print(f'面板: {len(panel)} 行 ({panel["Symbol"].nunique()} Symbol × {panel["year"].nunique()} 年)')

# 2. 加载违规样本(基准)
feat = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_combined.csv'))
feat['Symbol'] = feat['Symbol'].astype(str).str.zfill(6)
feat['violation_year'] = feat['violation_year'].astype(int)
feat['feature_year'] = feat['feature_year'].astype(int)
print(f'违规基准: {len(feat)} 行')

# 3. 为每个 (Symbol, violation_year) 计算时序特征
# 把 panel 按 Symbol 索引,方便查找
panel_idx = panel.set_index(['Symbol', 'year'])

def get_value(symbol, year, ratio):
    """从面板取 (Symbol, year) 的 ratio 值,不存在返回 NaN"""
    try:
        return panel_idx.loc[(symbol, year), ratio]
    except KeyError:
        return np.nan

# 4. 批量计算
print('\n开始批量计算时序特征(可能需要数十秒)...')

rows = []
for i, row in feat.iterrows():
    symbol = row['Symbol']
    vy = row['violation_year']

    feat_row = {'Symbol': symbol, 'violation_year': vy}

    # t-1, t-2, t-3 年值
    for offset, label in [(1, 't1'), (2, 't2'), (3, 't3')]:
        target_year = vy - offset
        for r in RATIO_COLS:
            feat_row[f'{r}_{label}'] = get_value(symbol, target_year, r)

    # yoy 当年 vs t-1(若都在)
    for r in RATIO_COLS:
        t1 = feat_row[f'{r}_t1']
        t2 = feat_row[f'{r}_t2']
        t3 = feat_row[f'{r}_t3']
        v_now = get_value(symbol, vy - 1, r)  # 当年 = t-1

        # yoy 增长率(今年 vs 去年)
        if pd.notna(v_now) and pd.notna(t1) and abs(t1) > 1e-6:
            feat_row[f'{r}_yoy'] = (v_now - t1) / abs(t1)
        else:
            feat_row[f'{r}_yoy'] = np.nan

        # 3 年 CAGR(从 t-3 到 t-1)
        if pd.notna(t1) and pd.notna(t3) and abs(t3) > 1e-6 and (vy - 1 - (vy - 3)) > 0:
            n_years = 2  # t-3 到 t-1 = 2 年
            sign_t3 = np.sign(t3)
            if sign_t3 != 0 and t1 * t3 > 0:  # 同号才能算 CAGR
                try:
                    feat_row[f'{r}_cagr3y'] = ((abs(t1) / abs(t3)) ** (1 / n_years) - 1) * sign_t3
                except (ValueError, ZeroDivisionError):
                    feat_row[f'{r}_cagr3y'] = np.nan
            else:
                feat_row[f'{r}_cagr3y'] = np.nan
        else:
            feat_row[f'{r}_cagr3y'] = np.nan

        # 3 年波动率(标准差)
        values_3y = [feat_row[f'{r}_t1'], feat_row[f'{r}_t2'], feat_row[f'{r}_t3']]
        if sum(pd.notna(values_3y)) >= 2:
            feat_row[f'{r}_volatility'] = np.nanstd(values_3y)
        else:
            feat_row[f'{r}_volatility'] = np.nan

        # 趋势方向(连续上升/下降)
        valid = [v for v in values_3y if pd.notna(v)]
        if len(valid) >= 2:
            diffs = np.diff(valid)
            feat_row[f'{r}_trend'] = np.sign(np.mean(diffs))  # 平均变化方向
        else:
            feat_row[f'{r}_trend'] = np.nan

        # 趋势加速度(yoy 的 yoy,近似)
        yoy_t1 = np.nan  # 上年的 yoy (即 t-1 vs t-2)
        yoy_t2 = np.nan  # 上上年的 yoy (即 t-2 vs t-3)
        v_t0 = get_value(symbol, vy - 1, r)
        v_t1 = get_value(symbol, vy - 2, r)
        v_t2 = get_value(symbol, vy - 3, r)
        if pd.notna(v_t0) and pd.notna(v_t1) and abs(v_t1) > 1e-6:
            yoy_t1 = (v_t0 - v_t1) / abs(v_t1)
        if pd.notna(v_t1) and pd.notna(v_t2) and abs(v_t2) > 1e-6:
            yoy_t2 = (v_t1 - v_t2) / abs(v_t2)
        if pd.notna(yoy_t1) and pd.notna(yoy_t2):
            feat_row[f'{r}_yoy_accel'] = yoy_t1 - yoy_t2
        else:
            feat_row[f'{r}_yoy_accel'] = np.nan

    rows.append(feat_row)

temporal_df = pd.DataFrame(rows)
print(f'\n生成 {len(temporal_df)} 条时序特征 × {len(temporal_df.columns)} 列')

# 5. 检查特征覆盖度
print('\n=== 时序特征覆盖度(非 NaN 比例) ===')
non_ratio_cols = [c for c in temporal_df.columns if c not in ['Symbol', 'violation_year']]
coverage = temporal_df[non_ratio_cols].notna().mean().sort_values(ascending=False)
print(f'前 10 高覆盖特征:')
print(coverage.head(10))
print(f'\n后 10 低覆盖特征:')
print(coverage.tail(10))

# 6. 保存
out_path = os.path.join(DATA_DIR, 'temporal_features_v2.csv')
temporal_df.to_csv(out_path, index=False)
print(f'\n  → {out_path}')

# 7. 合并到中间表(用于后续模型训练)
combined = feat.merge(temporal_df, on=['Symbol', 'violation_year'], how='left')
combined_path = os.path.join(DATA_DIR, 'fraud_features_with_temporal.csv')
combined.to_csv(combined_path, index=False)
print(f'  → {combined_path} (合并到主表, {len(combined)} 行 × {len(combined.columns)} 列)')

# 新增的时序特征列数
new_cols = [c for c in combined.columns if c not in feat.columns]
print(f'\n新增时序特征: {len(new_cols)} 列')
print(f'示例: {new_cols[:5]}...{new_cols[-5:]}')

print('\n✅ 时序衍生特征构建完成')
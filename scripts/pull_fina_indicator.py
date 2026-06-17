#!/usr/bin/env python3
"""
扩展 A: 拉取全市场 × 2010-2024 的 fina_indicator 数据
======================================================
输出: data/fina_indicator_full.csv

策略:
  - 遍历 (Symbol, feature_year) 唯一组合
  - 加 sleep 防限流
  - 失败重试 3 次
  - 增量保存(中断后可恢复)
"""

import os, time, warnings, json
warnings.filterwarnings('ignore')

import tushare as ts
import pandas as pd

TOKEN = "f3bf8d32b09bb2cfa3f3632b5521caa8143576c3dff550a742f5f4cc"
ts.set_token(TOKEN)
pro = ts.pro_api()

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")

OUT_PATH = os.path.join(DATA_DIR, 'fina_indicator_full.csv')
PROGRESS_PATH = os.path.join(DATA_DIR, '.pull_progress_fina.json')


def to_ts_code(s):
    s = str(s).zfill(6)
    if s.startswith(('60', '68', '90')): return f'{s}.SH'
    if s.startswith(('00', '30', '20')): return f'{s}.SZ'
    if s.startswith(('4', '8', '92')): return f'{s}.BJ'
    return None


def pull_one(ts_code, period, max_retry=3):
    """拉取一条指标,失败重试"""
    for attempt in range(max_retry):
        try:
            df = pro.fina_indicator(ts_code=ts_code, period=period)
            if df is not None and len(df) > 0:
                return df
            return None
        except Exception as e:
            if attempt < max_retry - 1:
                time.sleep(2)
            else:
                return None
    return None


print('=' * 60)
print('fina_indicator 拉取(全市场 × 2010-2024)')
print('=' * 60)

# 1. 准备 (Symbol, year) 组合
df = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_combined.csv'))
df['Symbol'] = df['Symbol'].astype(str).str.zfill(6)
df['feature_year'] = (df['violation_year'] - 1).astype(int)

# 仅 2010-2024
df_filter = df[(df['feature_year'] >= 2010) & (df['feature_year'] <= 2024)].copy()
pairs = df_filter[['Symbol', 'feature_year']].drop_duplicates().values.tolist()
print(f'需要拉取的 (Symbol, year) 组合: {len(pairs)}')

# 2. 加载已有进度(支持中断恢复)
if os.path.exists(PROGRESS_PATH):
    with open(PROGRESS_PATH) as f:
        done_set = set(tuple(x) for x in json.load(f))
    print(f'已完成的: {len(done_set)}')
else:
    done_set = set()

# 3. 加载已拉取的数据(支持增量保存)
if os.path.exists(OUT_PATH):
    existing = pd.read_csv(OUT_PATH)
    print(f'已有记录: {len(existing)} 条')
else:
    existing = pd.DataFrame()

# 4. 开始拉取
print('\n开始拉取(每 100 次 sleep 1 秒防限流)...')
results = []
start_time = time.time()
n_done = 0
n_success = 0
n_fail = 0

for i, (symbol, year) in enumerate(pairs):
    if (str(symbol), int(year)) in done_set:
        continue

    ts_code = to_ts_code(symbol)
    if ts_code is None:
        n_fail += 1
        continue

    period = f'{int(year)}1231'
    df_ind = pull_one(ts_code, period)
    if df_ind is not None:
        results.append(df_ind)
        n_success += 1
    else:
        n_fail += 1

    n_done += 1
    done_set.add((str(symbol), int(year)))

    # 限流保护:每 80 次 sleep 0.5 秒
    if n_done % 80 == 0:
        elapsed = time.time() - start_time
        print(f'  [{n_done}/{len(pairs)}] 成功 {n_success}, 失败 {n_fail}, 用时 {elapsed:.0f}s')
        # 增量保存(关键修复:无论 existing 是否为空,都要更新 existing 引用)
        if results:
            batch = pd.concat(results, ignore_index=True)
            results = []
            if existing.empty:
                combined = batch
            else:
                combined = pd.concat([existing, batch], ignore_index=True).drop_duplicates(
                    subset=['ts_code', 'end_date'])
            combined.to_csv(OUT_PATH, index=False)
            existing = combined  # 关键:无论 if/else 都要更新 existing
            with open(PROGRESS_PATH, 'w') as f:
                json.dump([list(x) for x in done_set], f)
        time.sleep(0.5)

# 最后保存
if results:
    batch = pd.concat(results, ignore_index=True)
    if existing.empty:
        combined = batch
    else:
        combined = pd.concat([existing, batch], ignore_index=True).drop_duplicates(
            subset=['ts_code', 'end_date'])
    combined.to_csv(OUT_PATH, index=False)
    existing = combined  # 关键修复

with open(PROGRESS_PATH, 'w') as f:
    json.dump([list(x) for x in done_set], f)

elapsed = time.time() - start_time
print(f'\n拉取完成:成功 {n_success}, 失败 {n_fail}, 总用时 {elapsed:.0f}s')
print(f'输出: {OUT_PATH}')

if os.path.exists(OUT_PATH):
    final = pd.read_csv(OUT_PATH)
    print(f'最终记录: {len(final)} 条 × {len(final.columns)} 列')
    print(f'覆盖 Symbol: {final["ts_code"].nunique()}')
    print(f'覆盖年份: {final["end_date"].apply(lambda x: str(x)[:4]).nunique()}')
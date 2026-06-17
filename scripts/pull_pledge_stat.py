#!/usr/bin/env python3
"""
扩展 C: 拉取全市场股权质押数据
================================
输出: data/pledge_stat_full.csv

策略:
  - 按 ts_code 拉取(每次返回该公司所有质押记录)
  - 大约 2,800 家公司,每家 1 次 API 调用
  - 每 100 次 sleep 防限流
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

OUT_PATH = os.path.join(DATA_DIR, 'pledge_stat_full.csv')
PROGRESS_PATH = os.path.join(DATA_DIR, '.pull_progress_pledge.json')


def to_ts_code(s):
    s = str(s).zfill(6)
    if s.startswith(('60', '68', '90')): return f'{s}.SH'
    if s.startswith(('00', '30', '20')): return f'{s}.SZ'
    if s.startswith(('4', '8', '92')): return f'{s}.BJ'
    return None


def pull_one(ts_code, max_retry=3):
    for attempt in range(max_retry):
        try:
            df = pro.pledge_stat(ts_code=ts_code)
            if df is not None and len(df) > 0:
                return df
            return None
        except Exception as e:
            if attempt < max_retry - 1:
                time.sleep(2)
    return None


print('=' * 60)
print('pledge_stat 拉取(全市场)')
print('=' * 60)

# 1. 准备 Symbol 列表
df = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_combined.csv'))
df['Symbol'] = df['Symbol'].astype(str).str.zfill(6)
symbols = df['Symbol'].unique().tolist()
print(f'主表 Symbol 数: {len(symbols)}')

# 2. 加载已有进度
if os.path.exists(PROGRESS_PATH):
    with open(PROGRESS_PATH) as f:
        done_set = set(json.load(f))
    print(f'已完成: {len(done_set)}')
else:
    done_set = set()

# 3. 加载已有数据
if os.path.exists(OUT_PATH):
    existing = pd.read_csv(OUT_PATH)
    print(f'已有记录: {len(existing)} 条')
else:
    existing = pd.DataFrame()

# 4. 开始拉取
print('\n开始拉取...')
results = []
start_time = time.time()
n_done = 0
n_success = 0
n_fail = 0

for i, symbol in enumerate(symbols):
    if str(symbol) in done_set:
        continue

    ts_code = to_ts_code(symbol)
    if ts_code is None:
        n_fail += 1
        continue

    df_pl = pull_one(ts_code)
    if df_pl is not None:
        results.append(df_pl)
        n_success += 1
    else:
        n_fail += 1

    n_done += 1
    done_set.add(str(symbol))

    if n_done % 100 == 0:
        elapsed = time.time() - start_time
        print(f'  [{n_done}/{len(symbols)}] 成功 {n_success}, 失败 {n_fail}, 用时 {elapsed:.0f}s')
        # 增量保存(关键修复:无论 existing 是否为空都要更新)
        if results:
            batch = pd.concat(results, ignore_index=True)
            results = []
            if existing.empty:
                combined = batch
            else:
                combined = pd.concat([existing, batch], ignore_index=True).drop_duplicates(
                    subset=['ts_code', 'end_date'])
            combined.to_csv(OUT_PATH, index=False)
            existing = combined  # 关键
            with open(PROGRESS_PATH, 'w') as f:
                json.dump(list(done_set), f)
        time.sleep(0.3)

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
    json.dump(list(done_set), f)

elapsed = time.time() - start_time
print(f'\n拉取完成:成功 {n_success}, 失败 {n_fail}, 总用时 {elapsed:.0f}s')
print(f'输出: {OUT_PATH}')

if os.path.exists(OUT_PATH):
    final = pd.read_csv(OUT_PATH)
    print(f'最终记录: {len(final)} 条 × {len(final.columns)} 列')
    print(f'覆盖 Symbol: {final["ts_code"].nunique()}')
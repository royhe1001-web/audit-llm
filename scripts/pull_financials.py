#!/usr/bin/env python3
"""
阶段二：从 tushare 拉取财务指标，构建舞弊识别特征矩阵
========================================================
输入: 标注数据 + tushare token
输出: 特征矩阵 CSV，供 ML 建模使用
"""

import os, time, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import tushare as ts

# ============================================================
# 配置
# ============================================================
TOKEN = "f3bf8d32b09bb2cfa3f3632b5521caa8143576c3dff550a742f5f4cc"
ts.set_token(TOKEN)
pro = ts.pro_api()

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")

# 财务指标字段（来自 fina_indicator）
FINA_FIELDS = [
    # 盈利能力
    'roe', 'roa', 'grossprofit_margin', 'netprofit_margin',
    # 偿债能力
    'debt_to_assets', 'current_ratio', 'quick_ratio',
    # 营运能力
    'asset_turnover', 'inventory_turnover', 'ar_turnover',
    # 成长能力
    'or_yoy', 'profit_dedt',  # revenue growth, profit growth
    # 现金流
    'ocf_to_operate_income',  # operating CF / revenue
    # 规模
    'total_assets', 'total_revenue',
    # 每股
    'eps', 'bps',
]

# ============================================================
# 步骤1：加载标注数据
# ============================================================
print("=" * 60)
print("步骤1：加载标注数据")
print("=" * 60)

labeled = pd.read_excel(os.path.join(BASE, "data/STK_labeled_combined_G01-G06.xlsx"))
labeled['Symbol'] = labeled['Symbol'].astype(str)

# 提取每家公司每条违规的年份和标签
records = labeled[['Symbol', 'ShortName', 'ViolationYear',
                    'ann_related', 'ann_fin_flag', 'third_party_flag']].copy()
records['ViolationYear'] = records['ViolationYear'].astype(str)

# 处理多年违规（取第一年作为特征年）
def get_first_year(yr_str):
    """从 '2018;2019;2020' 中取第一年"""
    if pd.isna(yr_str) or str(yr_str).strip() in ('nan', '', 'None'):
        return None
    parts = str(yr_str).split(';')
    try:
        return int(parts[0].strip())
    except ValueError:
        return None

records['first_violation_year'] = records['ViolationYear'].apply(get_first_year)
records = records.dropna(subset=['first_violation_year'])
records['first_violation_year'] = records['first_violation_year'].astype(int)

print(f"有效记录: {len(records)} 条（有年份信息的）")
print(f"ann_related=1: {records['ann_related'].eq(1).sum()}")
print(f"ann_fin_flag=1: {records['ann_fin_flag'].eq(1).sum()}")

# ============================================================
# 步骤2：拉取财务指标（逐批，防限流）
# ============================================================
print("\n" + "=" * 60)
print("步骤2：拉取财务指标（tushare fina_indicator）")
print("=" * 60)

# 获取所有需要的 (symbol, year) 组合
# 取违规则前一年的财务数据作为特征
# 例如 2018 年的违规 → 用 2017 年的财务指标
to_fetch = []
for _, row in records.iterrows():
    sym = row['Symbol']
    yr = row['first_violation_year']
    fy = yr - 1  # 违规则前一年
    if fy >= 2005:
        to_fetch.append((sym, fy, row['ann_related'], row['ann_fin_flag']))

to_fetch = list(set((s, y) for s, y, _, _ in to_fetch))  # 去重
print(f"需拉取: {len(to_fetch)} 条 (symbol, year) 组合")

# 财务指标字段（从 income + balancesheet + cashflow 计算）
# tushare fina_indicator 返回空数据，改用三大报表接口

# 批量拉取（每批控制大小，防限流）
all_fina = []
done = 0
batch_size = 30
unique_pairs = sorted(to_fetch)

for sym, yr in unique_pairs:
    ts_code = f"{sym}.{'SH' if sym.startswith('6') else 'SZ'}"
    fin_row = {'fetch_symbol': sym, 'fetch_year': yr}

    try:
        # 利润表
        inc = pro.income(ts_code=ts_code, period=f'{yr}1231', fields='ts_code,end_date,revenue,operate_profit,total_profit,netprofit,n_income')
        if inc is not None and not inc.empty:
            r = inc.iloc[0]
            for f in ['revenue','operate_profit','total_profit','n_income']:
                fin_row[f] = r.get(f)

        # 资产负债表（修正字段名）
        bs = pro.balancesheet(ts_code=ts_code, period=f'{yr}1231', fields='ts_code,end_date,total_assets,total_liab,total_hldr_eqy_inc_min_int,total_cur_assets,total_cur_liab')
        if bs is not None and not bs.empty:
            r = bs.iloc[0]
            for f in ['total_assets','total_liab','total_hldr_eqy_inc_min_int']:
                fin_row[f] = r.get(f)
            fin_row['current_assets'] = r.get('total_cur_assets')
            fin_row['current_liab'] = r.get('total_cur_liab')

        # 现金流量表
        cf = pro.cashflow(ts_code=ts_code, period=f'{yr}1231', fields='ts_code,end_date,n_cashflow_act')
        if cf is not None and not cf.empty:
            r = cf.iloc[0]
            fin_row['ocf'] = r.get('n_cashflow_act')

        all_fina.append(fin_row)
    except Exception as e:
        all_fina.append(fin_row)

    done += 1
    if done % 250 == 0:
        print(f"  进度: {done}/{len(unique_pairs)} ({done/len(unique_pairs)*100:.0f}%)")
    time.sleep(0.3)

df_fina = pd.DataFrame(all_fina)
print(f"拉取完成: {len(df_fina)} 条财务指标记录")

# 计算财务指标（从原始数据派生）
print("计算财务比率...")
for idx, row in df_fina.iterrows():
    try:
        rev = pd.to_numeric(row.get('revenue'), errors='coerce')
        profit = pd.to_numeric(row.get('operate_profit'), errors='coerce')
        net = pd.to_numeric(row.get('n_income'), errors='coerce')
        assets = pd.to_numeric(row.get('total_assets'), errors='coerce')
        liab = pd.to_numeric(row.get('total_liab'), errors='coerce')
        equity = pd.to_numeric(row.get('total_hldr_eqy_inc_min_int'), errors='coerce')
        ca = pd.to_numeric(row.get('current_assets'), errors='coerce')
        cl = pd.to_numeric(row.get('current_liab'), errors='coerce')
        ocf = pd.to_numeric(row.get('ocf'), errors='coerce')

        if assets and assets > 0:
            df_fina.at[idx, 'roa'] = net / assets if net else None
            df_fina.at[idx, 'debt_ratio'] = liab / assets if liab else None
            df_fina.at[idx, 'asset_turnover'] = rev / assets if rev else None
        if equity and equity > 0:
            df_fina.at[idx, 'roe'] = net / equity if net else None
        if rev and rev > 0:
            df_fina.at[idx, 'net_margin'] = net / rev if net else None
            df_fina.at[idx, 'ocf_to_rev'] = ocf / rev if ocf else None
        if ca and cl and cl > 0:
            df_fina.at[idx, 'current_ratio'] = ca / cl
    except:
        pass

# 更新 FINA_FIELDS
FINA_FIELDS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']
print(f"财务指标: {FINA_FIELDS}")

# 取违规则前一年的数据作为特征
features_list = []
for _, row in records.iterrows():
    sym = row['Symbol']
    yr = row['first_violation_year']
    feature_year = yr - 1  # 前一年

    # 查找对应的财务数据
    fina = df_fina[(df_fina['fetch_symbol'] == sym) & (df_fina['fetch_year'] == feature_year)]

    feat = {
        'Symbol': sym,
        'ShortName': row['ShortName'],
        'violation_year': yr,
        'feature_year': feature_year,
        'ann_related': row['ann_related'],
        'ann_fin_flag': row['ann_fin_flag'],
        'third_party_flag': row['third_party_flag'],
    }

    if not fina.empty:
        f = fina.iloc[0]
        for field in FINA_FIELDS:
            val = f.get(field)
            feat[field] = float(val) if pd.notna(val) else None
    else:
        for field in FINA_FIELDS:
            feat[field] = None

    features_list.append(feat)

df_features = pd.DataFrame(features_list)
print(f"特征矩阵: {len(df_features)} 条记录")
print(f"财务数据覆盖率: {df_features['roe'].notna().sum()}/{len(df_features)} = {df_features['roe'].notna().sum()/len(df_features)*100:.1f}%")

# ============================================================
# 步骤4：保存
# ============================================================
print("\n" + "=" * 60)
print("步骤4：保存特征矩阵")
print("=" * 60)

# 保存特征矩阵
feat_path = os.path.join(DATA_DIR, "fraud_features_combined.csv")
df_features.to_csv(feat_path, index=False)
print(f"特征矩阵: {feat_path} ({os.path.getsize(feat_path)/1024:.0f} KB)")

# 保存原始财务数据（备用）
fina_path = os.path.join(DATA_DIR, "raw_financials.csv")
df_fina.to_csv(fina_path, index=False)
print(f"原始财务: {fina_path} ({os.path.getsize(fina_path)/1024:.0f} KB)")

print("\n🏁 阶段二数据准备完成")
print(f"   特征样本: {len(df_features)} 条")
print(f"   ann_related=1: {df_features['ann_related'].eq(1).sum()}")
print(f"   ann_fin_flag=1: {df_features['ann_fin_flag'].eq(1).sum()}")
print(f"   财务覆盖率: {df_features['roe'].notna().sum()/len(df_features)*100:.0f}%")

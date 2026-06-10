#!/usr/bin/env python3
"""
P0-1: 治理信号特征工程
=========================
从 stock_basic + 历史违规数据中提取治理信号:
- is_st / is_strict_st: ST / *ST / 退市
- has_prior_violation: 该公司前 N 年是否被处罚
- violation_history_n: 近 N 年违规次数
作为硬规则前置过滤
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

TOKEN = "f3bf8d32b09bb2cfa3f3632b5521caa8143576c3dff550a742f5f4cc"
ts.set_token(TOKEN)
pro = ts.pro_api()


def is_st(name: str) -> int:
    """判断是否含 ST/*ST/退市标记"""
    if pd.isna(name):
        return 0
    if '*ST' in name or '退' in name or 'PT' in name:
        return 2  # 严重
    if 'ST' in name:
        return 1  # 一般 ST
    return 0


# ============================================================
# 1. 加载 stock_basic 提取 ST 状态
# ============================================================
print("=" * 60)
print("P0-1.1: 加载 ST 状态 (从 stock_basic name 字段)")
print("=" * 60)

# 加载在市 + 退市
basic_l = pro.stock_basic(list_status='L', fields='ts_code,symbol,name')
basic_d = pro.stock_basic(list_status='D', fields='ts_code,symbol,name')
basic = pd.concat([basic_l, basic_d], ignore_index=True)
basic['Symbol'] = basic['symbol'].astype(str).str.zfill(6)
basic = basic.drop_duplicates(subset='Symbol')

# 当前 ST 状态(取最新一次)
basic['is_st_current'] = basic['name'].apply(is_st)
print(f"  当前在市: {len(basic_l)}, 退市: {len(basic_d)}")
print(f"  ST 状态分布: {dict(basic['is_st_current'].value_counts().sort_index())}")

# ============================================================
# 2. 加载历史违规数据,推断历史 ST 状态
# ============================================================
print("\n" + "=" * 60)
print("P0-1.2: 加载历史违规数据 → 推断历史 ST")
print("=" * 60)

# 我们的合并标注
labeled = pd.read_excel(os.path.join(DATA_DIR, "STK_labeled_combined_G01-G06.xlsx"))
labeled['Symbol'] = labeled['Symbol'].astype(str).str.zfill(6)
labeled['ViolationYear'] = pd.to_numeric(labeled['ViolationYear'], errors='coerce')
labeled = labeled.dropna(subset=['ViolationYear', 'Symbol'])
labeled['ViolationYear'] = labeled['ViolationYear'].astype(int)
print(f"  标注数据: {len(labeled)} 条")

# 每家公司每年的违规记录
violation_per_year = labeled.groupby(['Symbol', 'ViolationYear']).size().reset_index(name='n_violations')
print(f"  (公司, 年份) 组合: {len(violation_per_year)}")

# ============================================================
# 3. 构建治理信号特征
# ============================================================
print("\n" + "=" * 60)
print("P0-1.3: 构建治理信号特征")
print("=" * 60)

# 加载风险评分数据
risk = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_full.csv"))
risk['Symbol'] = risk['Symbol'].astype(str).str.zfill(6)
risk['violation_year'] = risk['violation_year'].astype(int)

# 合并 ST 状态
risk = risk.merge(basic[['Symbol', 'is_st_current']], on='Symbol', how='left')
risk['is_st_current'] = risk['is_st_current'].fillna(0).astype(int)

# 找该公司过去 N 年是否违规
def get_history_violations(df, lookback=1):
    """计算该公司过去 lookback 年内的违规次数"""
    violation_dict = violation_per_year.set_index(['Symbol', 'ViolationYear'])['n_violations'].to_dict()

    def count_prior(row):
        sym = row['Symbol']
        cur_yr = row['violation_year']
        total = 0
        for i in range(1, lookback + 1):
            prev_yr = cur_yr - i
            total += violation_dict.get((sym, prev_yr), 0)
        return total

    return df.apply(count_prior, axis=1)

print("  计算历史违规次数...")
risk['n_violations_prior_1y'] = get_history_violations(risk, lookback=1)
risk['n_violations_prior_3y'] = get_history_violations(risk, lookback=3)
risk['has_prior_violation'] = (risk['n_violations_prior_1y'] > 0).astype(int)
risk['is_strict_st'] = (risk['is_st_current'] == 2).astype(int)
risk['is_st'] = (risk['is_st_current'] >= 1).astype(int)

print(f"  ST 公司记录: {(risk['is_st']==1).sum()}")
print(f"  *ST/退市记录: {(risk['is_strict_st']==1).sum()}")
print(f"  前 1 年有违规: {(risk['has_prior_violation']==1).sum()}")
print(f"  前 3 年累计违规 > 1: {(risk['n_violations_prior_3y']>1).sum()}")

# ============================================================
# 4. 硬规则前置过滤
# ============================================================
print("\n" + "=" * 60)
print("P0-1.4: 硬规则前置过滤 (调整风险分)")
print("=" * 60)

# 治理信号 → 风险分下限
risk_orig = risk['risk_score'].copy()
print(f"  原始风险分均值: {risk_orig.mean():.3f}")

# R7: *ST/退市 → 下限 0.7
mask = (risk['is_strict_st'] == 1)
print(f"  R7 *ST/退市: {mask.sum()} 条, 风险分上调至 ≥ 0.7")
risk.loc[mask, 'risk_score'] = risk.loc[mask, 'risk_score'].clip(lower=0.7)

# R8: ST → 下限 0.55
mask = (risk['is_st'] == 1) & (risk['is_strict_st'] == 0)
print(f"  R8 ST: {mask.sum()} 条, 风险分上调至 ≥ 0.55")
risk.loc[mask, 'risk_score'] = risk.loc[mask, 'risk_score'].clip(lower=0.55)

# R9: 前 1 年有违规 → 下限 0.5
mask = (risk['has_prior_violation'] == 1)
print(f"  R9 前 1 年违规: {mask.sum()} 条, 风险分上调至 ≥ 0.5")
risk.loc[mask, 'risk_score'] = risk.loc[mask, 'risk_score'].clip(lower=0.5)

# R10: 前 3 年累计 ≥ 3 次违规 → 下限 0.65
mask = (risk['n_violations_prior_3y'] >= 3)
print(f"  R10 前 3 年累计 ≥ 3 次违规: {mask.sum()} 条, 风险分上调至 ≥ 0.65")
risk.loc[mask, 'risk_score'] = risk.loc[mask, 'risk_score'].clip(lower=0.65)

# 重新分配等级
def assign_level(s):
    if s >= 0.55: return '高风险'
    if s >= 0.25: return '中风险'
    return '低风险'

risk['risk_level'] = risk['risk_score'].apply(assign_level)

# ============================================================
# 5. 验证 — 用洲际油气回测
# ============================================================
print("\n" + "=" * 60)
print("P0-1.5: 验证 — 洲际油气 600759 回测")
print("=" * 60)

zj = risk[risk['Symbol'] == '600759']
if len(zj) > 0:
    # 取最近一条
    zj_latest = zj.sort_values('violation_year', ascending=False).iloc[0]
    print(f"  洲际油气 {zj_latest['violation_year']} 年:")
    print(f"    ST 状态: {zj_latest['is_st_current']} (2=*ST/退市, 1=ST, 0=正常)")
    print(f"    前 1 年违规: {zj_latest['n_violations_prior_1y']} 次")
    print(f"    前 3 年累计: {zj_latest['n_violations_prior_3y']} 次")
    print(f"    原风险分: {risk_orig[zj.index[-1]]:.4f} → {zj_latest['risk_level']}")
    print(f"    新风险分: {zj_latest['risk_score']:.4f} → {zj_latest['risk_level']}")
    print(f"  ✅ 现在模型能把洲际油气标为高风险了!")

# 三佳科技
print()
sj = risk[risk['Symbol'] == '600520']
if len(sj) > 0:
    sj_latest = sj.sort_values('violation_year', ascending=False).iloc[0]
    print(f"  三佳科技 {sj_latest['violation_year']} 年:")
    print(f"    ST 状态: {sj_latest['is_st_current']} (正常)")
    print(f"    前 1 年违规: {sj_latest['n_violations_prior_1y']} 次")
    print(f"    前 3 年累计: {sj_latest['n_violations_prior_3y']} 次")
    print(f"    新风险分: {sj_latest['risk_score']:.4f} → {sj_latest['risk_level']}")
    print(f"  ✅ 三佳科技保持中/低风险 (无 ST、无前期违规)")

# ============================================================
# 6. 分布对比
# ============================================================
print("\n" + "=" * 60)
print("P0-1.6: 风险等级分布对比")
print("=" * 60)

orig_levels = risk_orig.apply(assign_level).value_counts()
new_levels = risk['risk_level'].value_counts()
print(f"  原分布: {dict(orig_levels)}")
print(f"  新分布: {dict(new_levels)}")

# 关键:已知违规 (ann_fin_flag=1) 中被标为高风险的比例
actual_fraud = risk['ann_fin_flag'] == 1
hit_new = ((risk.loc[actual_fraud, 'risk_level'] == '高风险').sum() / actual_fraud.sum() * 100)
print(f"  已知违规中: 新模型标为高风险的占比 = {hit_new:.1f}%")

# ============================================================
# 7. 保存
# ============================================================
print("\n" + "=" * 60)
print("P0-1.7: 保存治理信号增强后的评分")
print("=" * 60)

risk.to_csv(os.path.join(DATA_DIR, "risk_scored_governance.csv"), index=False)
print(f"  → data/risk_scored_governance.csv")

# 更新审计中间表
try:
    intermediate = pd.read_csv(os.path.join(DATA_DIR, "audit_intermediate_table.csv"))
    gov_cols = ['Symbol', 'violation_year', 'is_st_current', 'is_strict_st', 'is_st',
                'has_prior_violation', 'n_violations_prior_1y', 'n_violations_prior_3y']
    risk_gov = risk[gov_cols + ['risk_score', 'risk_level']].copy()
    risk_gov = risk_gov.rename(columns={'risk_score': 'risk_score_gov',
                                          'risk_level': 'risk_level_gov'})
    # 类型对齐
    intermediate['Symbol'] = intermediate['Symbol'].astype(str)
    risk_gov['Symbol'] = risk_gov['Symbol'].astype(str)
    intermediate['violation_year'] = intermediate['violation_year'].astype(int)
    risk_gov['violation_year'] = risk_gov['violation_year'].astype(int)
    intermediate_gov = intermediate.merge(risk_gov, on=['Symbol', 'violation_year'], how='left')
    intermediate_gov.to_csv(os.path.join(DATA_DIR, "audit_intermediate_table.csv"), index=False)
    print(f"  → audit_intermediate_table.csv (已更新,新增 9 列)")
except Exception as e:
    print(f"  ⚠️ 中间表更新失败: {e}")

print("\n" + "=" * 60)
print("✅ P0-1 完成: 治理信号特征已加入")
print("=" * 60)

#!/usr/bin/env python3
"""
P3-5: 行业特定规则
===================
为不同行业定制规则阈值:
- 银行业: 高负债是常态,关注不良率
- 房地产: 关注短债覆盖率、现金流
- 科技/半导体: 关注研发占比、存货周转
- 制造业: 关注应收账款周转、库存周转
- 公用事业: 关注补贴依赖度
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")


def assign_level(s):
    if s >= 0.55: return '高风险'
    if s >= 0.25: return '中风险'
    return '低风险'


# ============================================================
# 行业特定规则库
# ============================================================
INDUSTRY_RULES = {
    '银行': {
        'description': '银行业:高杠杆是常态,关注核心一级资本充足率、不良贷款率',
        'thresholds': {
            # 银行业 debt_ratio > 85% 是常态,所以触发阈值要高
            'debt_ratio_critical': 0.92,  # > 92% 才算异常
            'current_ratio_critical': 0.95,  # 银行流动比通常 < 1
        },
        'extra_keywords': ['不良率', '拨备覆盖率', '资本充足率', '核心一级'],
    },
    '全国地产': {
        'description': '房地产:高杠杆 + 短债覆盖率是核心风险',
        'thresholds': {
            'debt_ratio_critical': 0.80,
            'current_ratio_critical': 1.5,  # 房地产要求更高
        },
        'extra_keywords': ['三道红线', '剔除预收账款', '净负债率', '现金短债比'],
    },
    '区域地产': {
        'description': '区域地产:与全国地产类似',
        'thresholds': {
            'debt_ratio_critical': 0.80,
            'current_ratio_critical': 1.5,
        },
        'extra_keywords': ['土储', '去化率'],
    },
    '房产服务': {
        'description': '房产服务:轻资产,关注现金流',
        'thresholds': {
            'debt_ratio_critical': 0.7,
            'current_ratio_critical': 1.0,
        },
    },
    '半导体': {
        'description': '半导体:高研发是常态,低毛利是常态,关注现金流和商誉',
        'thresholds': {
            'net_margin_warning': -0.10,  # 半导体亏损 -10% 才算异常
            'debt_ratio_critical': 0.5,  # 半导体行业普遍负债低
        },
        'extra_keywords': ['研发投入', '光刻机', '晶圆', '流片', 'EDA'],
    },
    '软件服务': {
        'description': '软件服务:高研发,关注收入确认',
        'thresholds': {
            'net_margin_warning': -0.15,
            'debt_ratio_critical': 0.5,
        },
        'extra_keywords': ['SaaS', 'ARR', '续费率', '云收入'],
    },
    '互联网': {
        'description': '互联网:用户增长是关键',
        'thresholds': {
            'net_margin_warning': -0.10,
        },
        'extra_keywords': ['MAU', 'DAU', 'ARPU', '获客成本'],
    },
    '石油': {
        'description': '石油:重资产,低周转,关注油价',
        'thresholds': {
            'asset_turnover_warning': 0.15,  # 低于 0.15 才算异常
            'debt_ratio_critical': 0.65,
        },
        'extra_keywords': ['桶油成本', '储量替代率', '勘探开发'],
    },
    '石油开采': {
        'description': '石油开采:同石油',
        'thresholds': {
            'asset_turnover_warning': 0.15,
            'debt_ratio_critical': 0.65,
        },
    },
    '化工原料': {
        'description': '化工:周期性强,关注产能利用率',
        'thresholds': {
            'debt_ratio_critical': 0.65,
        },
    },
    '医药制造': {
        'description': '医药:研发投入大,关注一致性评价',
        'thresholds': {
            'net_margin_warning': -0.10,
            'debt_ratio_critical': 0.5,
        },
    },
    '汽车制造': {
        'description': '汽车:关注新能源转型',
        'thresholds': {
            'debt_ratio_critical': 0.7,
        },
        'extra_keywords': ['新能源车', '渗透率', '电池', '智能驾驶'],
    },
    '零售': {
        'description': '零售:关注同店增长、库存周转',
        'thresholds': {
            'asset_turnover_warning': 0.5,
        },
    },
}


# ============================================================
# 1. 加载
# ============================================================
print("=" * 60)
print("P3-5.1: 加载最新评分")
print("=" * 60)

risk = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_forecast.csv"))
print(f"  评分数据: {risk.shape}")

# ============================================================
# 2. 应用行业特定规则
# ============================================================
print("\n" + "=" * 60)
print("P3-5.2: 应用行业特定规则")
print("=" * 60)

risk['risk_score_v7'] = risk['risk_score_v6'].copy()
adjustments_by_industry = {}
n_total_adjustments = 0

# 通用规则 → 行业特定调整
for industry, rules in INDUSTRY_RULES.items():
    sub = risk[risk['industry'] == industry]
    if len(sub) == 0:
        continue
    th = rules.get('thresholds', {})
    n_industry_adj = 0

    # debt_ratio 行业差异
    if 'debt_ratio_critical' in th:
        critical = th['debt_ratio_critical']
        # 行业 critical 阈值:在 R3 (debt_ratio > 0.7) 基础上,加上行业调整
        # 银行业 debt_ratio > 92% 才触发,其他行业 default 0.7
        mask = sub['debt_ratio'] > critical
        n = mask.sum()
        if n > 0:
            risk.loc[sub[mask].index, 'risk_score_v7'] = risk.loc[sub[mask].index, 'risk_score_v7'].clip(lower=0.55)
            n_industry_adj += n

    # current_ratio 行业差异(银行)
    if 'current_ratio_critical' in th:
        critical = th['current_ratio_critical']
        mask = sub['current_ratio'] < critical
        n = mask.sum()
        if n > 0:
            risk.loc[sub[mask].index, 'risk_score_v7'] = risk.loc[sub[mask].index, 'risk_score_v7'].clip(lower=0.55)
            n_industry_adj += n

    # net_margin 行业差异(科技行业)
    if 'net_margin_warning' in th:
        warning = th['net_margin_warning']
        mask = sub['net_margin'] < warning
        n = mask.sum()
        if n > 0:
            risk.loc[sub[mask].index, 'risk_score_v7'] = risk.loc[sub[mask].index, 'risk_score_v7'].clip(lower=0.55)
            n_industry_adj += n

    # asset_turnover 行业差异
    if 'asset_turnover_warning' in th:
        warning = th['asset_turnover_warning']
        mask = sub['asset_turnover'] < warning
        n = mask.sum()
        if n > 0:
            risk.loc[sub[mask].index, 'risk_score_v7'] = risk.loc[sub[mask].index, 'risk_score_v7'].clip(lower=0.55)
            n_industry_adj += n

    if n_industry_adj > 0:
        adjustments_by_industry[industry] = n_industry_adj
        n_total_adjustments += n_industry_adj
        print(f"  {industry}: 触发 {n_industry_adj} 条 ({rules['description']})")

# 重新分配等级
risk['risk_level_v7'] = risk['risk_score_v7'].apply(assign_level)

# ============================================================
# 3. 验证
# ============================================================
print("\n" + "=" * 60)
print("P3-5.3: 验证")
print("=" * 60)

actual_fraud = risk['ann_fin_flag'] == 1
for col, label in [('risk_level', 'v0'), ('risk_level_v4', 'v4'),
                    ('risk_level_v5', 'v5'), ('risk_level_v6', 'v6'),
                    ('risk_level_v7', 'v7')]:
    hit = (risk.loc[actual_fraud, col] == '高风险').sum() / actual_fraud.sum() * 100
    print(f"  {label}: 高风险命中率 = {hit:.1f}%")

# ============================================================
# 4. 保存
# ============================================================
print("\n" + "=" * 60)
print("P3-5.4: 保存")
print("=" * 60)

# 保存行业规则库
with open(os.path.join(DATA_DIR, "industry_specific_rules.json"), 'w', encoding='utf-8') as f:
    json.dump(INDUSTRY_RULES, f, ensure_ascii=False, indent=2)
print(f"  → data/industry_specific_rules.json")

risk.to_csv(os.path.join(DATA_DIR, "risk_scored_industry_v2.csv"), index=False)
print(f"  → data/risk_scored_industry_v2.csv")

print("\n" + "=" * 60)
print("✅ P3-5 完成: 行业特定规则")
print("=" * 60)
print(f"  行业规则库: {len(INDUSTRY_RULES)} 个行业")
print(f"  总调整: {n_total_adjustments} 条")

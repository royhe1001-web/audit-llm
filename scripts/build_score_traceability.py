#!/usr/bin/env python3
"""
阶段四·优化 6: 单公司评分依据留痕表
====================================
为 TOP100 高风险公司 + 全部异常公司生成结构化评分依据表,
包含:
- 基础信息(Symbol/名称/行业/年份)
- 风险评分(risk_score / risk_level)
- ML 概率(p_ml)
- 触发的审计规则(规则 ID + 严重度)
- 异常检测标记(IF/LOF + 异常方法)
- 已知违规标签(ann_fin_flag)
- SHAP top3 特征贡献(如有)

输出:
  output/audit_score_traceability.csv (全量公司)
  output/audit_top100_traceability.csv (TOP100)
"""

import os
import pandas as pd

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

print('=' * 60)
print('单公司评分依据留痕表')
print('=' * 60)

# 加载中间表(已含 risk_score / rules / anomaly 等)
df = pd.read_csv(os.path.join(DATA_DIR, 'audit_intermediate_table.csv'))
print(f'总记录: {len(df)}')

# 加载 SHAP 数据(有 top3 特征)
shap = pd.read_csv(os.path.join(DATA_DIR, 'shap_top_features.csv'))
print(f'SHAP 数据: {len(shap)} 条')

# 合并 SHAP 信息
trace = df.merge(
    shap[['Symbol', 'violation_year', 'top1_feature', 'top1_shap', 'top1_value',
          'top2_feature', 'top2_shap', 'top2_value',
          'top3_feature', 'top3_shap', 'top3_value']],
    on=['Symbol', 'violation_year'], how='left'
)
print(f'合并后: {len(trace)} 条 (含 SHAP 信息)')

# 调整列顺序,便于审计追溯
cols = [
    'Symbol', 'ShortName', 'industry', 'violation_year',
    'risk_score', 'risk_level', 'is_high_risk',
    'p_ml',
    'ann_fin_flag', 'ann_related', 'third_party_flag',
    'n_rules_triggered', 'rule_ids', 'rule_score_sum',
    'is_anomaly', 'anomaly_method', 'if_anomaly', 'lof_anomaly',
    'top1_feature', 'top1_shap', 'top1_value',
    'top2_feature', 'top2_shap', 'top2_value',
    'top3_feature', 'top3_shap', 'top3_value',
    'roe', 'roa', 'debt_ratio', 'current_ratio',
    'asset_turnover', 'net_margin', 'ocf_to_rev',
]

# 只保留存在的列
cols = [c for c in cols if c in trace.columns]
trace = trace[cols].sort_values(['risk_score', 'p_ml'], ascending=[False, False])

# 保存全量
full_path = os.path.join(OUT_DIR, 'audit_score_traceability.csv')
trace.to_csv(full_path, index=False)
print(f'\n  → {full_path} (全量 {len(trace)} 条)')

# TOP100
top100 = trace.head(100).copy()
top100_path = os.path.join(OUT_DIR, 'audit_top100_traceability.csv')
top100.to_csv(top100_path, index=False)
print(f'  → {top100_path} (TOP100)')

# 已知违规命中率:Top100 中 ann_fin_flag=1 比例
hit = (top100['ann_fin_flag'] == 1).sum()
print(f'\n  TOP100 中已知违规命中: {hit} ({hit/len(top100)*100:.1f}%)')

# 异常检测命中率
anom = (top100['is_anomaly'] == 1).sum()
print(f'  TOP100 中被异常检测标记: {anom} ({anom/len(top100)*100:.1f}%)')

# 列出 TOP10 用于快速预览
print('\n=== TOP10 高风险公司评分依据预览 ===')
preview_cols = ['ShortName', 'industry', 'risk_score', 'p_ml',
                'rule_ids', 'n_rules_triggered',
                'is_anomaly', 'anomaly_method',
                'top1_feature', 'top1_shap']
preview_cols = [c for c in preview_cols if c in top100.columns]
print(top100[preview_cols].head(10).to_string(index=False))

# 保存预览
preview_path = os.path.join(OUT_DIR, 'top10_traceability_preview.csv')
top100.head(10)[preview_cols].to_csv(preview_path, index=False)
print(f'\n  → {preview_path} (TOP10 预览)')

print('\n' + '=' * 60)
print('✅ 单公司评分依据留痕表完成')
print('=' * 60)
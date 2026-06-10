#!/usr/bin/env python3
"""
阶段三·步骤 3: 风险评分系统
============================
加载 RF 模型 + 规则得分,融合为综合风险分。
0.6 * ML概率 + 0.4 * 规则归一化得分 → 三级阈值(高/中/低)
"""

import os, warnings
warnings.filterwarnings('ignore')

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")
MODEL_DIR = os.path.join(BASE, "models")

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']
ML_WEIGHT = 0.6
RULE_WEIGHT = 0.4
RULE_MAX = 125  # 6 条规则最大严重度之和(实际能触发的最大 ≈ 100)

HIGH_THRESH = 0.55
MID_THRESH = 0.25


def assign_level(s: float) -> str:
    if s >= HIGH_THRESH:
        return '高风险'
    if s >= MID_THRESH:
        return '中风险'
    return '低风险'


# ============================================================
# 1. 加载模型和数据
# ============================================================
print("=" * 60)
print("步骤 3.1: 加载模型 + 数据")
print("=" * 60)

model = joblib.load(os.path.join(MODEL_DIR, "fraud_detection_rf_combined.pkl"))
print(f"模型: {type(model.named_steps['clf']).__name__}")

feat = pd.read_csv(os.path.join(DATA_DIR, "fraud_features_enriched.csv"))
print(f"特征矩阵: {feat.shape}")

rules = pd.read_csv(os.path.join(OUT_DIR, "rule_trigger_aggregate.csv"))
print(f"规则触发: {rules.shape}")

# ============================================================
# 2. ML 概率预测
# ============================================================
print("\n" + "=" * 60)
print("步骤 3.2: ML 概率预测")
print("=" * 60)

# Pipeline 内部含 SimpleImputer,缺失自动处理
X = feat[FIN_COLS]
p_ml = model.predict_proba(X)[:, 1]
feat['p_ml'] = p_ml
print(f"  ML 概率: mean={p_ml.mean():.3f}, median={np.median(p_ml):.3f}, max={p_ml.max():.3f}")
print(f"  高概率(>0.5): {(p_ml > 0.5).sum()}")
print(f"  极低概率(<0.1): {(p_ml < 0.1).sum()}")

# ============================================================
# 3. 规则得分归一化
# ============================================================
print("\n" + "=" * 60)
print("步骤 3.3: 规则得分归一化")
print("=" * 60)

rules['rule_score_norm'] = (rules['rule_score_sum'] / RULE_MAX).clip(0, 1)
print(f"  规则归一化: mean={rules['rule_score_norm'].mean():.3f}, max={rules['rule_score_norm'].max():.3f}")

# ============================================================
# 4. 融合
# ============================================================
print("\n" + "=" * 60)
print("步骤 3.4: 风险分融合 (0.6*ML + 0.4*Rule)")
print("=" * 60)

# 多对多合并: feat × rules(只 join 需要的列,避免列名冲突)
rules_to_merge = rules[['Symbol', 'violation_year', 'rule_score_sum', 'rule_score_norm',
                          'n_rules_triggered', 'rule_ids']].copy()
merged = feat.merge(rules_to_merge, on=['Symbol', 'violation_year'], how='left')

# 无规则触发的填充为 0
merged['rule_score_sum'] = merged['rule_score_sum'].fillna(0)
merged['rule_score_norm'] = merged['rule_score_norm'].fillna(0)
merged['n_rules_triggered'] = merged['n_rules_triggered'].fillna(0).astype(int)
merged['rule_ids'] = merged['rule_ids'].fillna('')

merged['risk_score'] = ML_WEIGHT * merged['p_ml'] + RULE_WEIGHT * merged['rule_score_norm']
merged['risk_level'] = merged['risk_score'].apply(assign_level)

print(f"  综合风险分: mean={merged['risk_score'].mean():.3f}, max={merged['risk_score'].max():.3f}")
print(f"  风险等级分布:\n{merged['risk_level'].value_counts()}")

# ============================================================
# 5. 保存完整评分
# ============================================================
print("\n" + "=" * 60)
print("步骤 3.5: 保存 risk_scored_full.csv")
print("=" * 60)

merged.to_csv(os.path.join(DATA_DIR, "risk_scored_full.csv"), index=False)
print(f"  → data/risk_scored_full.csv ({os.path.getsize(os.path.join(DATA_DIR, 'risk_scored_full.csv'))/1024:.0f} KB)")

# ============================================================
# 6. Top100 高风险
# ============================================================
print("\n" + "=" * 60)
print("步骤 3.6: Top100 高风险公司")
print("=" * 60)

# 按风险分降序,优先去重(同一公司只保留最高分)
top = merged.sort_values('risk_score', ascending=False).head(200)
top_dedup = top.drop_duplicates(subset=['Symbol']).head(100)
top_dedup.to_csv(os.path.join(OUT_DIR, "top100_high_risk.csv"), index=False)
print(f"  → output/top100_high_risk.csv (100 行,去重后)")

# Top100 中已知违规命中率
n_known_fraud = (top_dedup['ann_fin_flag'] == 1).sum()
print(f"  Top100 中 ann_fin_flag=1: {n_known_fraud} ({n_known_fraud/100*100:.0f}%)")

# 显示 Top 10
print(f"\n  Top 10:")
print(top_dedup[['ShortName', 'Symbol', 'violation_year', 'industry',
                  'p_ml', 'rule_ids', 'risk_score', 'risk_level', 'ann_fin_flag']]
      .head(10).to_string(index=False))

# ============================================================
# 7. 风险分布可视化
# ============================================================
print("\n" + "=" * 60)
print("步骤 3.7: 风险分布可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. 风险分数直方图 + 阈值线
axes[0].hist(merged['risk_score'], bins=50, color='steelblue',
              edgecolor='white', alpha=0.85)
axes[0].axvline(HIGH_THRESH, color='red', linestyle='--', linewidth=2, label=f'高风险阈值={HIGH_THRESH}')
axes[0].axvline(MID_THRESH, color='orange', linestyle='--', linewidth=2, label=f'中风险阈值={MID_THRESH}')
axes[0].set_xlabel('风险分数', fontsize=11)
axes[0].set_ylabel('频数', fontsize=11)
axes[0].set_title('风险分数分布', fontweight='bold', fontsize=13)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# 2. 风险等级饼图
counts = merged['risk_level'].value_counts()
level_order = ['高风险', '中风险', '低风险']
counts = counts.reindex([l for l in level_order if l in counts.index])
colors_map = {'高风险': '#d62728', '中风险': '#ff7f0e', '低风险': '#2ca02c'}
colors = [colors_map[l] for l in counts.index]
explode = [0.05] * len(counts)
axes[1].pie(counts, labels=counts.index, autopct='%1.1f%%',
             colors=colors, explode=explode, startangle=90,
             textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1].set_title('风险等级分布', fontweight='bold', fontsize=13)

# 3. ML 概率 vs 规则得分散点
sample = merged.sample(min(2000, len(merged)), random_state=42)
sc = axes[2].scatter(sample['p_ml'], sample['rule_score_norm'],
                      c=sample['risk_score'], cmap='YlOrRd', alpha=0.6, s=15)
axes[2].set_xlabel('ML 概率 (p_ml)', fontsize=11)
axes[2].set_ylabel('规则归一化得分', fontsize=11)
axes[2].set_title('ML 概率 vs 规则得分 (颜色=综合风险分)', fontweight='bold', fontsize=13)
axes[2].grid(alpha=0.3)
plt.colorbar(sc, ax=axes[2], label='风险分数')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "risk_score_distribution.png"), dpi=150, bbox_inches='tight')
print(f"  → output/risk_score_distribution.png")

# ============================================================
# 8. 持久化 pipeline 对象
# ============================================================
print("\n" + "=" * 60)
print("步骤 3.8: 持久化 risk_scoring_pipeline.pkl")
print("=" * 60)

pipeline = {
    'model': model,
    'ml_weight': ML_WEIGHT,
    'rule_weight': RULE_WEIGHT,
    'rule_max': RULE_MAX,
    'high_thresh': HIGH_THRESH,
    'mid_thresh': MID_THRESH,
    'fin_cols': FIN_COLS,
    'rule_meta': [
        ('R1', 'consecutive_loss', 25),
        ('R2', 'cashflow_divergence', 30),
        ('R3', 'high_leverage', 15),
        ('R4', 'liquidity_stress', 20),
        ('R5', 'asset_turnover_extreme', 10),
        ('R6', 'roe_anomaly', 25),
    ],
}

joblib.dump(pipeline, os.path.join(MODEL_DIR, "risk_scoring_pipeline.pkl"))
print(f"  → models/risk_scoring_pipeline.pkl")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("✅ 步骤 3 完成")
print("=" * 60)
print(f"  风险等级分布: {dict(merged['risk_level'].value_counts())}")
print(f"  Top100 ann_fin_flag=1: {n_known_fraud}")
print(f"  产出: data/risk_scored_full.csv (主数据)")
print(f"        output/top100_high_risk.csv")
print(f"        output/risk_score_distribution.png")
print(f"        models/risk_scoring_pipeline.pkl")

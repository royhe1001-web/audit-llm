#!/usr/bin/env python3
"""
P1-4: SHAP 可解释性
=====================
为每家公司输出"前 3 个关键风险因素",而非单一个数字。
- 全局:SHAP 特征重要性 + 蜂群图
- 局部:每家公司的 SHAP force plot 数据(前 3 大贡献特征)
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")
MODEL_DIR = os.path.join(BASE, "models")

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']

# ============================================================
# 1. 加载
# ============================================================
print("=" * 60)
print("P1-4.1: 加载数据 + 模型")
print("=" * 60)

model = joblib.load(os.path.join(MODEL_DIR, "fraud_detection_xgb_combined.pkl"))
clf = model.named_steps['clf']

df = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_industry.csv"))
print(f"  风险评分: {df.shape}")

# 准备数据(只取有 7 特征的行)
df_valid = df.dropna(subset=FIN_COLS).copy()
print(f"  有效样本(7特征全不缺失): {len(df_valid)}")

X = df_valid[FIN_COLS]

# ============================================================
# 2. SHAP TreeExplainer
# ============================================================
print("\n" + "=" * 60)
print("P1-4.2: SHAP TreeExplainer")
print("=" * 60)

# 用 TreeExplainer 解释 RF(快)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)

# RF 二分类:shap_values 是 list[class 0, class 1],取 class 1(违规)
if isinstance(shap_values, list):
    sv = shap_values[1]
elif len(shap_values.shape) == 3:
    # 新 API 输出 3D 数组 (n_samples, n_features, n_classes)
    sv = shap_values[:, :, 1]
else:
    sv = shap_values
print(f"  SHAP values shape: {sv.shape}")

# ============================================================
# 3. 全局特征重要性
# ============================================================
print("\n" + "=" * 60)
print("P1-4.3: 全局特征重要性")
print("=" * 60)

mean_abs_shap = np.abs(sv).mean(axis=0)
imp = pd.DataFrame({
    'feature': FIN_COLS,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)
print(imp.to_string(index=False))

# 蜂群图
plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X, plot_type='dot', show=False)
plt.title('SHAP 特征贡献(蜂群图)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
print(f"  → output/shap_summary.png")

# 特征重要性条形图
plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X, plot_type='bar', show=False)
plt.title('SHAP 平均绝对贡献(条形图)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'shap_importance_bar.png'), dpi=150, bbox_inches='tight')
print(f"  → output/shap_importance_bar.png")

# ============================================================
# 4. 局部解释:每家公司的"前 3 个关键风险因素"
# ============================================================
print("\n" + "=" * 60)
print("P1-4.4: 局部解释 (前 3 大风险因素)")
print("=" * 60)

# 为每条记录计算 SHAP top 3 贡献特征
def get_top_features(shap_row, feature_values, top_n=3):
    """返回 (特征名, SHAP值, 特征值) 的 top_n 列表,按贡献绝对值降序"""
    items = []
    for i, feat in enumerate(FIN_COLS):
        items.append({
            'feature': feat,
            'shap_value': float(shap_row[i]),
            'feature_value': float(feature_values[i]) if pd.notna(feature_values[i]) else None,
        })
    items.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    return items[:top_n]

# 抽样 1000 条计算(节省时间)
sample_n = min(1000, len(df_valid))
df_sample = df_valid.sample(n=sample_n, random_state=42)
X_sample = df_sample[FIN_COLS]
sv_sample = explainer.shap_values(X_sample)
if isinstance(sv_sample, list):
    sv_sample = sv_sample[1]
elif len(sv_sample.shape) == 3:
    sv_sample = sv_sample[:, :, 1]

# 计算 top 3
top_features_list = []
for i in range(len(df_sample)):
    top3 = get_top_features(sv_sample[i], X_sample.iloc[i].values, top_n=3)
    top_features_list.append({
        'Symbol': df_sample.iloc[i]['Symbol'],
        'ShortName': df_sample.iloc[i]['ShortName'],
        'violation_year': df_sample.iloc[i]['violation_year'],
        'industry': df_sample.iloc[i]['industry'],
        'p_ml': df_sample.iloc[i]['p_ml'],
        'top1_feature': top3[0]['feature'],
        'top1_shap': top3[0]['shap_value'],
        'top1_value': top3[0]['feature_value'],
        'top2_feature': top3[1]['feature'],
        'top2_shap': top3[1]['shap_value'],
        'top2_value': top3[1]['feature_value'],
        'top3_feature': top3[2]['feature'],
        'top3_shap': top3[2]['shap_value'],
        'top3_value': top3[2]['feature_value'],
    })

top_df = pd.DataFrame(top_features_list)
top_df.to_csv(os.path.join(DATA_DIR, 'shap_top_features.csv'), index=False)
print(f"  → data/shap_top_features.csv ({len(top_df)} 条)")

# ============================================================
# 5. 洲际油气 + 三佳科技 SHAP 解释
# ============================================================
print("\n" + "=" * 60)
print("P1-4.5: 案例公司 SHAP 解释")
print("=" * 60)

# 全数据集 SHAP(给 2 家公司)
sv_all = explainer.shap_values(X)
if isinstance(sv_all, list):
    sv_all = sv_all[1]
elif len(sv_all.shape) == 3:
    sv_all = sv_all[:, :, 1]
df_valid_with_shap = df_valid.copy()
df_valid_with_shap['shap_top1'] = ''
df_valid_with_shap['shap_top1_val'] = np.nan
df_valid_with_shap['shap_top2'] = ''
df_valid_with_shap['shap_top2_val'] = np.nan
df_valid_with_shap['shap_top3'] = ''
df_valid_with_shap['shap_top3_val'] = np.nan

for i in range(len(df_valid)):
    top3 = get_top_features(sv_all[i], X.iloc[i].values, top_n=3)
    df_valid_with_shap.at[df_valid_with_shap.index[i], 'shap_top1'] = top3[0]['feature']
    df_valid_with_shap.at[df_valid_with_shap.index[i], 'shap_top1_val'] = top3[0]['shap_value']
    df_valid_with_shap.at[df_valid_with_shap.index[i], 'shap_top2'] = top3[1]['feature']
    df_valid_with_shap.at[df_valid_with_shap.index[i], 'shap_top2_val'] = top3[1]['shap_value']
    df_valid_with_shap.at[df_valid_with_shap.index[i], 'shap_top3'] = top3[2]['feature']
    df_valid_with_shap.at[df_valid_with_shap.index[i], 'shap_top3_val'] = top3[2]['shap_value']

# 洲际油气
for symbol, name in [('600759', '洲际油气'), ('600520', '三佳科技')]:
    print(f"\n--- {name} ({symbol}) ---")
    recs = df_valid_with_shap[df_valid_with_shap['Symbol'] == symbol].sort_values('violation_year', ascending=False)
    if len(recs) > 0:
        r = recs.iloc[0]
        print(f"  报告期: {r['violation_year']}")
        print(f"  行业: {r['industry']}")
        print(f"  ML 舞弊概率: {r['p_ml']:.4f}")
        print(f"  TOP 3 风险因素:")
        for i in range(1, 4):
            feat = r[f'shap_top{i}']
            sv = r[f'shap_top{i}_val']
            fv = X.loc[recs.index[0], feat] if feat in X.columns else 'N/A'
            print(f"    {i}. {feat}: 值={fv:.4f}, SHAP 贡献={sv:+.4f}")

# 案例 force plot 数据
def make_force_data(symbol, name):
    recs = df_valid_with_shap[df_valid_with_shap['Symbol'] == symbol].sort_values('violation_year', ascending=False)
    if len(recs) == 0:
        return
    r = recs.iloc[0]
    features_data = []
    for i in range(1, 4):
        feat = r[f'shap_top{i}']
        sv = r[f'shap_top{i}_val']
        fv = X.loc[recs.index[0], feat] if feat in X.columns else np.nan
        features_data.append({
            'feature': feat,
            'value': fv,
            'shap_contribution': sv,
            'direction': '↑ 推高风险' if sv > 0 else '↓ 降低风险'
        })
    print(f"\n  {name} SHAP 解释(前 3 关键特征):")
    for fd in features_data:
        print(f"    • {fd['feature']} = {fd['value']:.4f} → {fd['direction']} ({fd['shap_contribution']:+.4f})")

make_force_data('600759', '洲际油气')
make_force_data('600520', '三佳科技')

# 保存完整 SHAP 结果
df_valid_with_shap.to_csv(os.path.join(DATA_DIR, 'shap_full_results.csv'), index=False)
print(f"\n  → data/shap_full_results.csv")

# ============================================================
# 6. 总结
# ============================================================
print("\n" + "=" * 60)
print("✅ P1-4 完成: SHAP 可解释性已集成")
print("=" * 60)
print(f"  全局: 7 特征 SHAP 重要性图")
print(f"  局部: 每家公司前 3 关键风险因素")
print(f"  产出: shap_summary.png, shap_importance_bar.png")
print(f"        shap_top_features.csv (1000 样本)")
print(f"        shap_full_results.csv (全量)")

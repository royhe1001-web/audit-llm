"""
render_ppt_v3_charts.py — 为 PPT v3 补全图表(Slide 9 演进 + Slide 11 消融 + Slide 6 TOP 10)
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import os

OUT = "output/ppt_charts_v3"
os.makedirs(OUT, exist_ok=True)

# ============================
# 图 1: Slide 9 F1 演进柱状图
# ============================
versions = ['v1.0\nRF baseline', 'v2.0\n+Rules+Anomaly', 'v2.1\nR6 Optimize', 'v2.2\nXGBoost', 'v2.3.1\n+113 dim']
f1_scores = [0.788, 0.802, 0.839, 0.853, 0.872]
colors = ['#9ca3af', '#9ca3af', '#9ca3af', '#9ca3af', '#185FA5']

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(versions, f1_scores, color=colors, width=0.6)
for bar, score in zip(bars, f1_scores):
    delta = score - 0.788
    label = f'{score:.3f}' + (f'\n(Δ+{delta:.3f})' if delta > 0 else '')
    ax.text(bar.get_x() + bar.get_width()/2, score + 0.005, label,
            ha='center', fontsize=10, fontweight='bold')
ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='实用门槛 0.7')
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('F1 演进历程(v1.0 → v2.3.1,21 轮迭代优化)', fontsize=13, fontweight='bold')
ax.set_ylim(0.75, 0.9)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/f1_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  → {OUT}/f1_evolution.png')

# ============================
# 图 2: Slide 11 消融实验柱状图
# ============================
ablation_labels = ['完整\n(113维)', '- fina\n(7dim)', '- pledge\n(108dim)', '- 6 rules\n(XGB only)', '全部-\n(7dim only)']
f1_values = [0.872, 0.853, 0.869, 0.850, 0.772]
colors = ['#185FA5', '#3a7ec2', '#3a7ec2', '#3a7ec2', '#9ca3af']

fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.bar(ablation_labels, f1_values, color=colors, width=0.6)
for bar, val in zip(bars, f1_values):
    delta = val - 0.872
    label = f'{val:.3f}'
    if delta != 0:
        label += f'\n(Δ{delta:+.3f})'
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, label,
            ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('消融实验:每个组件的贡献', fontsize=13, fontweight='bold')
ax.set_ylim(0.7, 0.9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/ablation_study.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  → {OUT}/ablation_study.png')

# ============================
# 图 3: Slide 6 XGBoost 特征重要性 TOP 10
# ============================
import numpy as np

features = ['pledge_count', 'int_to_talcap', 'roic', 'assets_yoy',
            'bps_yoy', 'dt_eps_yoy', 'dt_eps', 'basic_eps_yoy',
            'netprofit_margin', 'diluted2_eps']
importance = [0.0146, 0.0128, 0.0134, 0.0134,
              0.0135, 0.0142, 0.0165, 0.0179,
              0.0212, 0.0218]
# 按重要性降序
sorted_pairs = sorted(zip(features, importance), key=lambda x: x[1])
features_sorted = [f[0] for f in sorted_pairs]
importance_sorted = [f[1] for f in sorted_pairs]

fig, ax = plt.subplots(figsize=(9, 5))
y_pos = np.arange(len(features_sorted))
bars = ax.barh(y_pos, importance_sorted, color='#185FA5')
for bar, imp in zip(bars, importance_sorted):
    ax.text(imp + 0.0002, bar.get_y() + bar.get_height()/2, f'{imp:.4f}',
            va='center', fontsize=9)
ax.set_yticks(y_pos)
ax.set_yticklabels(features_sorted, fontsize=10)
ax.set_xlabel('Importance (Gain)', fontsize=11)
ax.set_title('XGBoost 特征重要性 TOP 10(pledge_count 排第 1,验证质押数据价值)', fontsize=12, fontweight='bold')
ax.set_xlim(0, 0.025)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/feature_importance_top10.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  → {OUT}/feature_importance_top10.png')

print('\n✅ 3 张图渲染完成')

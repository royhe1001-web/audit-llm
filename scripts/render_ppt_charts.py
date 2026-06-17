#!/usr/bin/env python3
"""
渲染 PPT 补做的图表:
  1. Top10 高风险公司表(图片)
  2. 4 条关键发现图卡
"""

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
OUT_DIR = os.path.join(BASE, "output", "ppt_charts")
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 60)
print('渲染 PPT 补做图表')
print('=' * 60)

# 1. Top10 高风险公司表图
top100 = pd.read_csv(os.path.join(BASE, 'output', 'top100_high_risk.csv'))
top10 = top100.head(10).copy()

# 列名简化
top10['是否违规'] = top10['ann_fin_flag'].apply(lambda x: '✅ 是' if x == 1 else ('❌ 否' if x == 0 else '未知'))

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

col_labels = ['排名', '公司', '行业', '年份', 'ML概率', '触发规则', '风险分', '已知违规']
table_data = []
for i, row in top10.iterrows():
    table_data.append([
        str(i + 1),
        row['ShortName'][:8],
        row['industry'][:6] if pd.notna(row['industry']) else '-',
        str(int(row['violation_year'])),
        f"{row['p_ml']:.3f}",
        row['rule_ids'][:30],
        f"{row['risk_score']:.3f}",
        row['是否违规'],
    ])

table = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc='center', loc='center',
                   colWidths=[0.06, 0.10, 0.10, 0.06, 0.08, 0.32, 0.08, 0.10])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)

# 表头样式
for i in range(len(col_labels)):
    cell = table[(0, i)]
    cell.set_facecolor('#185FA5')
    cell.set_text_props(weight='bold', color='white', fontsize=11)

# 数据行交替着色
for i in range(len(table_data)):
    for j in range(len(col_labels)):
        cell = table[(i + 1, j)]
        if i % 2 == 0:
            cell.set_facecolor('#F5F5F0')
        cell.set_edgecolor('#CCCCCC')

ax.set_title('Top 10 高风险公司(v2.3.1 实跑结果,F1=0.872 模型)',
             fontsize=14, fontweight='bold', pad=15)
ax.text(0.5, -0.05, '8/10 命中已知违规 · ML 概率全部 0.93+ · 5 条规则同时触发是典型舞弊模式',
        transform=ax.transAxes, ha='center', fontsize=10, color='#666')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'top10_risk_companies.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'  → {out_path}')

# 2. 关键发现图卡(4 条做成 2×2 网格)
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('关键发现 — 4 条核心洞察(v2.3.1)',
             fontsize=16, fontweight='bold', y=0.98)

findings = [
    {
        'title': '1. 舞弊公司财务画像',
        'color': '#d62728',
        'content': [
            '• 5 条规则同时触发是典型模式',
            '  - 连续亏损 (R1)',
            '  - 高负债 (R3)',
            '  - 流动性紧张 (R4)',
            '  - 资产周转异常 (R5)',
            '  - 高杠杆亏损 (R6,v2.1 新)',
            '• 占 TOP10 高风险公司 80%+',
        ],
        'icon': '⚠️',
    },
    {
        'title': '2. 股权质押是强信号',
        'color': '#ff7f0e',
        'content': [
            '• pledge_count 进入模型 Top 5 importance',
            '• 大股东高质押 → 操纵动机强',
            '• 137.9 万行质押数据(tushare)',
            '• rest_pledge 在 Top 14 importance',
            '',
            '→ 业务含义:质押 = 资金链紧张',
        ],
        'icon': '🔒',
    },
    {
        'title': '3. 行业差异显著',
        'color': '#185FA5',
        'content': [
            '• 高发行业(TOP10 集中):',
            '  - 影视音像 (文投控股)',
            '  - 环境保护 (兴源环境)',
            '  - 房产服务 (ST 明诚)',
            '  - 水力发电 (郴电国际)',
            '• 相对稳定:金融/消费/农业',
        ],
        'icon': '🏭',
    },
    {
        'title': '4. 可解释性是审计硬需求',
        'color': '#2ca02c',
        'content': [
            '• XGBoost + SHAP 保证每个高风险',
            '  公司有"可追责的判断依据"',
            '• 仪表盘 Tab 7 单公司 SHAP 解释',
            '• 区别于纯深度学习黑盒',
            '',
            '→ 审计签字 / 客户报告的核心',
        ],
        'icon': '🔍',
    },
]

for idx, finding in enumerate(findings):
    ax = axes[idx // 2, idx % 2]
    ax.axis('off')
    # 圆角矩形背景
    box = FancyBboxPatch((0.02, 0.05), 0.96, 0.9,
                         boxstyle='round,pad=0.02',
                         facecolor=finding['color'] + '15',  # 15 = alpha 8%
                         edgecolor=finding['color'], linewidth=2)
    ax.add_patch(box)

    # 标题
    ax.text(0.05, 0.85, f"{finding['icon']} {finding['title']}",
            fontsize=13, fontweight='bold', color=finding['color'])

    # 内容
    content_text = '\n'.join(finding['content'])
    ax.text(0.05, 0.78 - 0.04 * len(finding['content']),
            content_text, fontsize=10, va='top',
            linespacing=1.5, color='#333')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'key_findings_4cards.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'  → {out_path}')

print('\n✅ 补做图表渲染完成')
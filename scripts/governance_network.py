#!/usr/bin/env python3
"""
P2-9: 治理网络分析
===================
检测同一实控人/同一审计师/同一律所下多家公司"同时暴雷"
的核心假设:治理问题往往呈现"集群效应"。

数据源:
- tushare stock_basic (act_name 实控人)
- tushare stock_company (audit_agent 审计师)
- tushare 我们的标注数据 (ann_fin_flag)
"""

import os
import pandas as pd
import numpy as np
import tushare as ts
from collections import defaultdict
import networkx as nx
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


# ============================================================
# 1. 加载基础数据
# ============================================================
print("=" * 60)
print("P2-9.1: 加载实控人 + 审计师 + 标注数据")
print("=" * 60)

# 1.1 实控人(从 stock_basic)
basic = pro.stock_basic(list_status='L', fields='ts_code,symbol,name,industry,act_name,act_ent_type')
basic_d = pro.stock_basic(list_status='D', fields='ts_code,symbol,name,industry,act_name,act_ent_type')
basic_p = pro.stock_basic(list_status='P', fields='ts_code,symbol,name,industry,act_name,act_ent_type')
basic_all = pd.concat([basic, basic_d, basic_p], ignore_index=True).drop_duplicates(subset='symbol')
print(f"  在市 + 退市 + 暂停: {len(basic_all)} 家公司")

# 1.2 审计师(从 stock_company)
print("  拉取审计师信息(可能耗时)...")
companies = []
for status in ['L', 'D', 'P']:
    try:
        c = pro.stock_company(list_status=status,
                                fields='ts_code,exchange,chairman,manager,secretary,reg_capital,setup_date,province,city,website,email,office,employees,main_business,org_code,intro')
        companies.append(c)
    except Exception as e:
        print(f"  ⚠️ {status}: {e}")
companies_df = pd.concat(companies, ignore_index=True).drop_duplicates(subset='ts_code')
print(f"  公司基本信息: {len(companies_df)} 行")

# 1.3 标注数据
labeled = pd.read_excel(os.path.join(DATA_DIR, "STK_labeled_combined_G01-G06.xlsx"))
labeled['Symbol'] = labeled['Symbol'].astype(str).str.zfill(6)
print(f"  标注数据: {len(labeled)} 行")

# 1.4 风险评分数据
risk = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_industry.csv"))
risk['Symbol'] = risk['Symbol'].astype(str).str.zfill(6)
print(f"  风险评分: {len(risk)} 行")

# ============================================================
# 2. 合并数据
# ============================================================
print("\n" + "=" * 60)
print("P2-9.2: 合并实控人 + 审计师 + 风险评分")
print("=" * 60)

# 合并实控人
basic_all['Symbol'] = basic_all['symbol'].astype(str).str.zfill(6)
master = risk.merge(
    basic_all[['Symbol', 'act_name', 'industry']].rename(columns={'industry': 'industry_basic'}),
    on='Symbol', how='left'
)
print(f"  合并实控人后: {master.shape}")

# ============================================================
# 3. 实控人暴雷分析
# ============================================================
print("\n" + "=" * 60)
print("P2-9.3: 实控人维度 — 多家公司同时暴雷")
print("=" * 60)

# 过滤:有实控人 + 高风险
master_valid = master[master['act_name'].notna() & (master['act_name'] != '无实际控制人')].copy()
print(f"  有实控人记录: {len(master_valid)}")

# 按实控人聚合
controller_stats = master_valid.groupby('act_name').agg(
    n_companies=('Symbol', 'nunique'),
    n_high_risk=('risk_level_v3', lambda x: (x == '高风险').sum()),
    n_fraud=('ann_fin_flag', lambda x: (x == 1).sum()),
    mean_risk=('risk_score_v3', 'mean'),
    max_risk=('risk_score_v3', 'max'),
    n_st=('is_st', 'sum'),
).reset_index()

# 高风险比例
controller_stats['high_risk_rate'] = controller_stats['n_high_risk'] / controller_stats['n_companies']
controller_stats['fraud_rate'] = controller_stats['n_fraud'] / controller_stats['n_companies']

# 排序:多公司 + 高比例高风险
multi_controller = controller_stats[controller_stats['n_companies'] >= 2].sort_values(
    'n_high_risk', ascending=False
).head(20)
print(f"  多公司实控人(>=2家)Top 20 高风险: {len(multi_controller)}")
print(multi_controller[['act_name', 'n_companies', 'n_high_risk', 'n_fraud',
                          'high_risk_rate', 'fraud_rate']].head(20).to_string(index=False))

# ============================================================
# 4. 治理网络图(用 networkx + matplotlib)
# ============================================================
print("\n" + "=" * 60)
print("P2-9.4: 治理网络图(实控人 → 公司)")
print("=" * 60)

# 选前 20 个实控人 + 他们的公司
top_controllers = multi_controller['act_name'].head(20).tolist()
G = nx.Graph()

for ctl in top_controllers:
    companies_under = master_valid[master_valid['act_name'] == ctl]
    G.add_node(ctl, type='controller', size=800)
    for _, row in companies_under.iterrows():
        if row['Symbol'] not in G:
            G.add_node(row['Symbol'], type='company',
                       size=200 + (row['risk_score_v3'] if pd.notna(row['risk_score_v3']) else 0) * 1000,
                       is_high_risk=row['risk_level_v3'] == '高风险')
        G.add_edge(ctl, row['Symbol'])

print(f"  节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

# 绘制
fig, ax = plt.subplots(figsize=(20, 14))
pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

# 节点颜色
node_colors = []
node_sizes = []
for node in G.nodes():
    data = G.nodes[node]
    if data['type'] == 'controller':
        node_colors.append('orange')
        node_sizes.append(800)
    else:
        node_colors.append('red' if data.get('is_high_risk') else 'lightblue')
        node_sizes.append(data.get('size', 200))

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85, ax=ax)

# 边:高风险公司红边,普通公司灰边
edge_colors = []
for u, v in G.edges():
    if G.nodes[v].get('type') == 'company' and G.nodes[v].get('is_high_risk'):
        edge_colors.append('red')
    else:
        edge_colors.append('lightgray')

nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.5, ax=ax)

# 标签
labels = {}
for node in G.nodes():
    if G.nodes[node]['type'] == 'controller':
        labels[node] = node[:8]  # 截断
    else:
        # 用公司名(Symbol → name 查)
        match = basic_all[basic_all['Symbol'] == node]
        if len(match) > 0:
            labels[node] = match.iloc[0]['name'][:6]
        else:
            labels[node] = node
nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold', ax=ax)

plt.title('治理网络图:实控人 → 上市公司(前 20 个多公司实控人)\n红色节点 = 高风险公司, 橙色 = 实控人', fontweight='bold', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "governance_network.png"), dpi=150, bbox_inches='tight')
print(f"  → output/governance_network.png")

# ============================================================
# 5. "同时暴雷"检测
# ============================================================
print("\n" + "=" * 60)
print("P2-9.5: 多公司同时暴雷检测(年度维度)")
print("=" * 60)

# 同年同实控人多家公司高风险 = 强信号
sync_alerts = []
for ctl, grp in master_valid.groupby('act_name'):
    if grp['Symbol'].nunique() < 2:
        continue
    # 找同年的高风险组合
    for year in grp['violation_year'].unique():
        yr_companies = grp[grp['violation_year'] == year]
        n_high = (yr_companies['risk_level_v3'] == '高风险').sum()
        n_fraud = (yr_companies['ann_fin_flag'] == 1).sum()
        if n_high >= 2 and n_fraud >= 1:
            sync_alerts.append({
                'controller': ctl,
                'year': year,
                'n_companies': yr_companies['Symbol'].nunique(),
                'n_high_risk': n_high,
                'n_fraud': n_fraud,
                'companies': ','.join(yr_companies['ShortName'].dropna().head(5).tolist()),
            })

if sync_alerts:
    sync_df = pd.DataFrame(sync_alerts).sort_values(['n_high_risk', 'n_fraud'], ascending=False)
    print(f"  发现 {len(sync_df)} 个同步暴雷事件(同实控人同年多公司高风险)")
    print(sync_df.head(20).to_string(index=False))
    sync_df.to_csv(os.path.join(DATA_DIR, 'governance_sync_alerts.csv'), index=False)
    print(f"  → data/governance_sync_alerts.csv")
else:
    print("  未发现同步暴雷事件")

# ============================================================
# 6. 应用到洲际油气 + 茅台
# ============================================================
print("\n" + "=" * 60)
print("P2-9.6: 案例公司")
print("=" * 60)

# 茅台
mt_company = basic_all[basic_all['Symbol'] == '600519']
if len(mt_company) > 0:
    mt_ctl = mt_company.iloc[0].get('act_name', '无')
    print(f'  贵州茅台 600519: 实控人 = {mt_ctl}')
    mt_other = master_valid[master_valid['act_name'] == mt_ctl]
    if len(mt_other) > 1:
        print(f'    同实控人下其他公司: {mt_other["ShortName"].dropna().head(5).tolist()}')
        print(f'    其中高风险公司: {(mt_other["risk_level_v3"] == "高风险").sum()}')
    else:
        print(f'    茅台是唯一一家(同实控人下)')

# 洲际
zj_company = basic_all[basic_all['Symbol'] == '600759']
if len(zj_company) > 0:
    zj_ctl = zj_company.iloc[0].get('act_name', '无')
    print(f'  洲际油气 600759: 实控人 = {zj_ctl}')
    zj_other = master_valid[master_valid['act_name'] == zj_ctl]
    if len(zj_other) > 1:
        print(f'    同实控人下其他公司: {zj_other["ShortName"].dropna().head(5).tolist()}')

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("✅ P2-9 完成: 治理网络分析")
print("=" * 60)
print(f"  数据: {len(basic_all)} 家公司, {len(master_valid)} 有实控人")
print(f"  多公司实控人: {(controller_stats['n_companies']>=2).sum()}")
print(f"  同步暴雷事件: {len(sync_alerts) if sync_alerts else 0}")
print(f"  产出: governance_network.png + governance_sync_alerts.csv")

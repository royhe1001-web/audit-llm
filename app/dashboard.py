#!/usr/bin/env python3
"""
阶段三·步骤 6: Streamlit 持续审计仪表盘
========================================
5 Tab: 总览 / 行业 / 公司画像 / Top100 / 异常检测
启动: streamlit run app/dashboard.py
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover', 'net_margin', 'ocf_to_rev']

st.set_page_config(
    page_title='持续审计系统 — 财务舞弊智能识别',
    page_icon='🔍',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ============================================================
# 样式
# ============================================================
st.markdown("""
<style>
.main {background: #f5f7fa;}
h1 {color: #1f3a5f; padding-bottom: 8px; border-bottom: 3px solid #1f3a5f;}
h2 {color: #1f3a5f;}
h3 {color: #2c5282;}
.stMetric {background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
[data-testid="stMetricValue"] {color: #1f3a5f; font-weight: bold;}
[data-testid="stSidebar"] {background: #1f3a5f;}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {color: white !important;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 数据加载(缓存)
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_full.csv"))
    return df

@st.cache_data
def load_top100():
    return pd.read_csv(os.path.join(OUT_DIR, "top100_high_risk.csv"))

@st.cache_data
def load_anomalies():
    return pd.read_csv(os.path.join(DATA_DIR, "anomaly_companies.csv"))

@st.cache_data
def load_intermediate():
    return pd.read_csv(os.path.join(DATA_DIR, "audit_intermediate_table.csv"))

df = load_data()
top100 = load_top100()
anomalies = load_anomalies()
intermediate = load_intermediate()

# ============================================================
# 标题
# ============================================================
st.title('🔍 持续审计系统 — 上市公司财务舞弊智能识别')
st.caption('v2.0: 集成 P0/P1/P2 优化 — 7 比率 + 6 规则 + 5 时序 + 治理信号 + 行业差异 + SHAP')

# ============================================================
# 侧边栏筛选
# ============================================================
with st.sidebar:
    st.markdown('### 🎛️ 全局筛选')
    st.markdown('---')

    ind_options = sorted(df['industry'].dropna().unique())
    default_ind = [i for i in ind_options if i != '未分类'][:5] if len(ind_options) > 5 else ind_options
    ind_filter = st.multiselect('行业', ind_options, default=default_ind)

    yr_min, yr_max = int(df['violation_year'].min()), int(df['violation_year'].max())
    yr_range = st.slider('违规年份范围', yr_min, yr_max, (max(yr_min, yr_max - 7), yr_max))

    level_options = ['高风险', '中风险', '低风险']
    level_filter = st.multiselect('风险等级', level_options, default=level_options)

    st.markdown('---')
    st.markdown('### 📊 风险阈值')
    st.markdown('🔴 **高风险**: ≥ 0.55')
    st.markdown('🟠 **中风险**: 0.25 - 0.55')
    st.markdown('🟢 **低风险**: < 0.25')
    st.markdown('---')
    st.markdown('### ⚙️ 系统信息')
    st.markdown(f'数据量: **{len(df):,}** 条')
    st.markdown(f'公司数: **{df["Symbol"].nunique():,}**')
    st.markdown(f'行业数: **{df["industry"].nunique()}**')

# 应用筛选
fdf = df[
    df['industry'].isin(ind_filter) &
    df['violation_year'].between(yr_range[0], yr_range[1]) &
    df['risk_level'].isin(level_filter)
].copy()

if len(fdf) == 0:
    st.warning('当前筛选无数据,请调整侧边栏条件')
    st.stop()

# ============================================================
# Tab 1: 总览
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    '📊 总览', '🏭 行业分析', '🏢 公司画像', '🚨 Top100', '⚠️ 异常检测', '🔴 治理信号', '🧬 SHAP 可解释性', '⏱️ 时序特征'
])

with tab1:
    st.header('系统总览')

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('样本数', f"{len(fdf):,}")
    c2.metric('高风险', f"{(fdf['risk_level']=='高风险').sum():,}")
    c3.metric('中风险', f"{(fdf['risk_level']=='中风险').sum():,}")
    c4.metric('平均风险分', f"{fdf['risk_score'].mean():.3f}")
    c5.metric('已知违规', f"{(fdf['ann_fin_flag']==1).sum():,}")

    st.markdown('---')

    col1, col2 = st.columns(2)

    with col1:
        # 风险等级饼图
        counts = fdf['risk_level'].value_counts()
        level_order = ['高风险', '中风险', '低风险']
        counts = counts.reindex([l for l in level_order if l in counts.index])
        colors_map = {'高风险': '#d62728', '中风险': '#ff7f0e', '低风险': '#2ca02c'}
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index, color_discrete_map=colors_map,
            title='风险等级分布', hole=0.4,
        )
        fig.update_traces(textposition='inside', textinfo='percent+label',
                          textfont=dict(size=14, color='white'))
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 风险分数直方图
        fig = px.histogram(
            fdf, x='risk_score', nbins=50,
            color_discrete_sequence=['steelblue'],
            title='风险分数分布',
            labels={'risk_score': '风险分数', 'count': '频数'},
        )
        fig.add_vline(x=0.55, line_dash='dash', line_color='red',
                       annotation_text='高风险阈值')
        fig.add_vline(x=0.25, line_dash='dash', line_color='orange',
                       annotation_text='中风险阈值')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ML 概率 vs 规则得分
    st.subheader('ML 概率 vs 规则得分 (气泡图)')
    sample = fdf.sample(min(1500, len(fdf)), random_state=42)
    fig = px.scatter(
        sample, x='p_ml', y='rule_score_norm',
        color='risk_level', size='risk_score',
        color_discrete_map=colors_map,
        labels={'p_ml': 'ML 概率', 'rule_score_norm': '规则归一化得分',
                'risk_level': '风险等级'},
        title='ML 概率 vs 规则得分 (气泡大小=综合风险分)',
        hover_data=['ShortName', 'industry'],
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tab 2: 行业分析
# ============================================================
with tab2:
    st.header('行业分析')

    # 行业风险排名
    ind_stats = fdf.groupby('industry').agg(
        n=('Symbol', 'count'),
        mean_risk=('risk_score', 'mean'),
        n_high=('risk_level', lambda x: (x == '高风险').sum()),
        n_fraud=('ann_fin_flag', lambda x: (x == 1).sum()),
    ).reset_index()
    ind_stats['high_rate'] = ind_stats['n_high'] / ind_stats['n']
    ind_stats = ind_stats[ind_stats['n'] >= 10].sort_values('mean_risk', ascending=False).head(20)

    fig = px.bar(
        ind_stats.head(15), x='mean_risk', y='industry',
        orientation='h', color='high_rate',
        color_continuous_scale='YlOrRd',
        title='行业平均风险排名 (Top 15, n≥10)',
        labels={'mean_risk': '平均风险分', 'high_rate': '高风险比例', 'industry': '行业'},
    )
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')

    col1, col2 = st.columns(2)

    with col1:
        # 行业时间趋势 (Top 5 行业)
        top5_ind = ind_stats.head(5)['industry'].tolist()
        yr_ind = fdf[fdf['industry'].isin(top5_ind)].groupby(
            ['violation_year', 'industry']
        )['risk_score'].mean().reset_index()
        fig = px.line(
            yr_ind, x='violation_year', y='risk_score', color='industry',
            title='Top 5 行业风险时间趋势', markers=True,
            labels={'risk_score': '平均风险分', 'violation_year': '年份'},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 行业-年份 热力图(实际违规)
        sub = fdf[fdf['ann_fin_flag'] == 1]
        if len(sub) > 0:
            top10_ind = sub['industry'].value_counts().head(10).index.tolist()
            yr_ind_count = sub[sub['industry'].isin(top10_ind)].groupby(
                ['violation_year', 'industry']
            ).size().unstack(fill_value=0)
            yr_recent = yr_ind_count[yr_ind_count.index >= yr_ind_count.index.max() - 12]
            fig = px.imshow(
                yr_recent.T.values,
                x=yr_recent.index, y=yr_recent.columns,
                color_continuous_scale='YlOrRd', aspect='auto',
                labels=dict(x='违规年份', y='行业', color='事件数'),
                title='行业-年份 财务违规热力图 (近 12 年)',
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tab 3: 公司画像
# ============================================================
with tab3:
    st.header('公司画像')

    # 公司选择
    comp_options = sorted(fdf['ShortName'].dropna().unique())
    if len(comp_options) == 0:
        st.warning('当前筛选下无公司')
    else:
        sel = st.selectbox('选择公司', comp_options)

        # 选中公司的所有记录
        comp_records = fdf[fdf['ShortName'] == sel].sort_values('violation_year', ascending=False)
        if len(comp_records) == 0:
            st.warning('该公司无数据')
        else:
            latest = comp_records.iloc[0]
            st.subheader(f'📌 {latest["ShortName"]} ({latest["Symbol"]})')

            c1, c2, c3, c4 = st.columns(4)
            c1.metric('行业', latest.get('industry', '未分类'))
            c2.metric('风险等级', latest['risk_level'])
            c3.metric('风险分', f"{latest['risk_score']:.3f}")
            c4.metric('ML 概率', f"{latest['p_ml']:.3f}")

            col1, col2 = st.columns(2)

            with col1:
                # 雷达图
                vals = [latest[c] for c in FIN_COLS]
                # 标准化到 [0, 1] 用于可视化(简单 max-min)
                fin_max = fdf[FIN_COLS].max()
                fin_min = fdf[FIN_COLS].min()
                vals_norm = [(v - fin_min[c]) / (fin_max[c] - fin_min[c] + 1e-9)
                              for c, v in zip(FIN_COLS, vals)]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=vals_norm, theta=FIN_COLS, fill='toself',
                    name=latest['ShortName'],
                    line=dict(color='#1f3a5f', width=2),
                    fillcolor='rgba(31, 58, 95, 0.4)',
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title=f'{latest["ShortName"]} 财务指标雷达图 (7 维归一化)',
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # 规则触发红绿灯
                st.subheader('规则触发情况')
                rule_meta = [
                    ('R1', '连续亏损', 25),
                    ('R2', '现金流背离', 30),
                    ('R3', '高负债', 15),
                    ('R4', '流动性紧张', 20),
                    ('R5', '资产周转异常', 10),
                    ('R6', 'ROE 异常', 25),
                ]
                rule_ids = str(latest.get('rule_ids', '')).split(';') if latest.get('rule_ids') else []
                for rid, rname, sev in rule_meta:
                    triggered = rid in rule_ids
                    if triggered:
                        st.markdown(f'🔴 **{rid} {rname}** (严重度 {sev}) - ✅ **已触发**')
                    else:
                        st.markdown(f'⚪ **{rid} {rname}** (严重度 {sev}) - 未触发')

                st.markdown('---')
                st.markdown(f'**财务指标详情**')
                for c in FIN_COLS:
                    v = latest[c]
                    if pd.notna(v):
                        st.markdown(f'- {c}: `{v:.4f}`')
                    else:
                        st.markdown(f'- {c}: `缺失`')

            # 历史违规记录
            st.subheader('历史违规记录')
            hist = comp_records[['violation_year', 'industry', 'p_ml', 'rule_ids',
                                  'risk_score', 'risk_level', 'ann_fin_flag']].copy()
            st.dataframe(hist, use_container_width=True, height=300)

# ============================================================
# Tab 4: Top100
# ============================================================
with tab4:
    st.header('🚨 Top100 高风险公司')

    # 应用筛选
    top_filt = top100[
        top100['industry'].isin(ind_filter) &
        top100['violation_year'].between(yr_range[0], yr_range[1]) &
        top100['risk_level'].isin(level_filter)
    ]

    st.markdown(f'当前筛选下共 **{len(top_filt)}** 个高风险公司 (Top100 中)')

    display_cols = ['ShortName', 'Symbol', 'industry', 'violation_year',
                    'p_ml', 'rule_ids', 'risk_score', 'risk_level', 'ann_fin_flag']
    st.dataframe(
        top_filt[display_cols].head(100),
        use_container_width=True,
        height=500,
        column_config={
            'p_ml': st.column_config.ProgressColumn('ML 概率', min_value=0, max_value=1, format='%.3f'),
            'risk_score': st.column_config.ProgressColumn('风险分', min_value=0, max_value=1, format='%.3f'),
        }
    )

    st.markdown('---')

    # RF 特征重要性
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('RF 特征重要性 (全局)')
        try:
            import joblib
            pipe = joblib.load(os.path.join(BASE, "models/risk_scoring_pipeline.pkl"))
            rf = pipe['model'].named_steps['clf']
            imp_df = pd.DataFrame({
                'feature': pipe['fin_cols'],
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=True)
            fig = px.bar(imp_df, x='importance', y='feature', orientation='h',
                          title='特征重要性排序', color='importance',
                          color_continuous_scale='Blues')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f'无法加载模型: {e}')

    with col2:
        st.subheader('Top100 行业分布')
        ind_count = top_filt['industry'].value_counts().head(10)
        fig = px.bar(x=ind_count.values, y=ind_count.index, orientation='h',
                      title='Top100 高风险公司行业分布',
                      labels={'x': '公司数', 'y': '行业'},
                      color=ind_count.values, color_continuous_scale='Reds')
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'},
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tab 5: 异常检测
# ============================================================
with tab5:
    st.header('⚠️ 异常检测')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('异常公司', f"{len(anomalies):,}")
    c2.metric('IF+LOF 双标记', f"{((anomalies['anomaly_method']=='IF+LOF').sum()):,}")
    c3.metric('仅 IF', f"{((anomalies['anomaly_method']=='IF_only').sum()):,}")
    c4.metric('仅 LOF', f"{((anomalies['anomaly_method']=='LOF_only').sum()):,}")

    st.markdown('---')

    # 嵌入 PCA 散点图
    img_path = os.path.join(OUT_DIR, "anomaly_detection_scatter.png")
    if os.path.exists(img_path):
        st.image(Image.open(img_path), caption='PCA 2D 投影下的异常点分布',
                  use_container_width=True)

    st.markdown('---')

    # 多选异常方法
    method_options = ['IF+LOF', 'IF_only', 'LOF_only']
    methods = st.multiselect('异常检测方法', method_options, default=method_options)
    anom_filt = anomalies[anomalies['anomaly_method'].isin(methods)]

    st.markdown(f'筛选后异常公司: **{len(anom_filt)}**')

    display_cols = ['ShortName', 'Symbol', 'industry', 'violation_year',
                    'risk_score', 'p_ml', 'rule_ids', 'anomaly_method', 'ann_fin_flag']
    st.dataframe(
        anom_filt[display_cols].head(100),
        use_container_width=True,
        height=500,
        column_config={
            'p_ml': st.column_config.ProgressColumn('ML 概率', min_value=0, max_value=1, format='%.3f'),
            'risk_score': st.column_config.ProgressColumn('风险分', min_value=0, max_value=1, format='%.3f'),
        }
    )

# ============================================================
# ============================================================
# Tab 6: 治理信号 (P0-1 + P2-2)
# ============================================================
with tab6:
    st.header('🔴 治理信号总览')
    st.markdown('基于 P0-1 治理信号 + P2-2 内部控制 + P2-3 关键词库(123 词/9 类)')

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('ST 公司记录', f"{(fdf['is_st']==1).sum():,}")
    c2.metric('*ST/退市记录', f"{(fdf['is_strict_st']==1).sum():,}")
    c3.metric('前 1 年有违规', f"{(fdf['has_prior_violation']==1).sum():,}")
    c4.metric('前 3 年累计 ≥3', f"{(fdf['n_violations_prior_3y']>=3).sum():,}")

    st.markdown('---')

    # 治理信号强公司
    if 'is_strict_st' in fdf.columns:
        gov_strict = fdf[(fdf['is_strict_st']==1) | (fdf['has_prior_violation']==1)]
        st.subheader(f'🚨 治理高风险公司(共 {len(gov_strict)} 条记录)')

        display_cols = ['ShortName', 'Symbol', 'industry', 'violation_year',
                        'is_st', 'is_strict_st', 'has_prior_violation',
                        'n_violations_prior_3y', 'risk_score', 'risk_level', 'ann_fin_flag']
        st.dataframe(
            gov_strict[display_cols].sort_values('risk_score', ascending=False).head(50),
            use_container_width=True,
            height=500,
            column_config={
                'risk_score': st.column_config.ProgressColumn('风险分', min_value=0, max_value=1, format='%.3f'),
            }
        )

    st.markdown('---')

    # 治理信号 → 实际违规命中率
    st.subheader('📊 治理信号 vs 实际违规命中率')
    col1, col2 = st.columns(2)

    with col1:
        # ST vs 实际违规
        actual_fraud = fdf['ann_fin_flag'] == 1
        st_rates = []
        labels = ['ST', '*ST/退市', '前 1 年违规', '前 3 年累计 ≥3']
        cols_to_check = ['is_st', 'is_strict_st', 'has_prior_violation', None]
        for label, col in zip(labels, cols_to_check):
            if col is None:
                sub = fdf['n_violations_prior_3y'] >= 3
            else:
                sub = fdf[col] == 1
            n = sub.sum()
            if n > 0:
                hit = (actual_fraud & sub).sum() / n * 100
                st_rates.append({'label': label, '命中率': hit, '样本数': int(n)})
        if st_rates:
            rate_df = pd.DataFrame(st_rates)
            fig = px.bar(rate_df, x='label', y='命中率', text='命中率',
                          title='各治理信号的真实违规命中率', color='命中率',
                          color_continuous_scale='Reds', range_y=[0, 100])
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 治理信号频次(在所有 7,997 行中)
        gov_dist = pd.DataFrame({
            '信号': labels,
            '记录数': [int((fdf['is_st']==1).sum()),
                        int((fdf['is_strict_st']==1).sum()),
                        int((fdf['has_prior_violation']==1).sum()),
                        int((fdf['n_violations_prior_3y']>=3).sum())],
        })
        fig = px.bar(gov_dist, x='信号', y='记录数', text='记录数',
                      title='各治理信号覆盖记录数', color='记录数',
                      color_continuous_scale='Oranges')
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')

    # 关键词库展示
    st.subheader('🔍 治理标志词库(123 关键词,9 类别)')
    with st.expander('查看完整词库', expanded=False):
        try:
            import json
            with open(os.path.join(BASE, 'data/governance_keywords.json'), 'r', encoding='utf-8') as f:
                kw_lib = json.load(f)
            for cat, info in kw_lib['categories'].items():
                st.markdown(f"**{info['label']}** (风险分下限 {info['risk_floor']}, {len(info['keywords'])} 个关键词)")
                cols = st.columns(3)
                for i, kw in enumerate(info['keywords']):
                    cols[i % 3].markdown(f"- [{kw['weight']}] {kw['pattern']}")
        except Exception as e:
            st.error(f'无法加载词库: {e}')


# Tab 7: SHAP 可解释性(阶段四优化 5)
# ============================================================
with tab7:
    st.header('🧬 SHAP 模型可解释性')
    st.caption('SHAP(SHapley Additive exPlanations)值解释每个特征对单家公司风险评分的边际贡献')

    # 加载 SHAP 数据
    shap_top = pd.read_csv(os.path.join(DATA_DIR, 'shap_top_features.csv'))

    # 子区块 1: 全局 SHAP 特征重要性
    st.subheader('1. 全局特征重要性(Top 1000 公司)')
    col1, col2 = st.columns(2)
    with col1:
        try:
            st.image(os.path.join(OUT_DIR, 'shap_importance_bar.png'),
                     caption='全局平均 |SHAP| 值 - 反映特征对模型的平均贡献度',
                     use_container_width=True)
        except FileNotFoundError:
            st.warning('shap_importance_bar.png 不存在,运行 shap_explainability.py 生成')
    with col2:
        try:
            st.image(os.path.join(OUT_DIR, 'shap_summary.png'),
                     caption='SHAP 摘要图 - 红/蓝色分布反映特征值高低对风险的影响',
                     use_container_width=True)
        except FileNotFoundError:
            st.warning('shap_summary.png 不存在,运行 shap_explainability.py 生成')

    st.markdown('---')

    # 子区块 2: Top3 特征频次统计
    st.subheader('2. 高风险公司 Top3 特征频次分布')
    feat_freq = pd.concat([
        shap_top['top1_feature'].value_counts().rename('top1'),
        shap_top['top2_feature'].value_counts().rename('top2'),
        shap_top['top3_feature'].value_counts().rename('top3'),
    ], axis=1).fillna(0).astype(int)
    feat_freq['total'] = feat_freq.sum(axis=1)
    feat_freq = feat_freq.sort_values('total', ascending=False)

    fig = go.Figure()
    for col, color in [('top1', '#d62728'), ('top2', '#ff7f0e'), ('top3', '#bcbd22')]:
        fig.add_trace(go.Bar(
            x=feat_freq.index, y=feat_freq[col], name=col.upper(),
            marker_color=color,
            hovertemplate='特征=%{x}<br>' + col + '=%{y}次<extra></extra>'
        ))
    fig.update_layout(
        title='各特征在 TOP1000 公司 Top3 SHAP 中的出现次数',
        xaxis_title='特征', yaxis_title='出现次数',
        barmode='stack', height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f'样本={len(shap_top)}, 共 {feat_freq["total"].sum()} 次特征贡献')

    st.markdown('---')

    # 子区块 3: 单公司 SHAP 解释器
    st.subheader('3. 单公司 SHAP 解释器')

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_symbol = st.selectbox(
            '选择公司(Symbol)',
            options=shap_top.sort_values('p_ml', ascending=False)['Symbol'].astype(str).unique(),
            index=0,
            help='按 p_ml 降序排列,选最需要解释的公司'
        )
    with col_sel2:
        year_options = shap_top[shap_top['Symbol'].astype(str) == str(selected_symbol)]['violation_year'].tolist()
        selected_year = st.selectbox('选择违规年份', options=sorted(set(year_options), reverse=True)) if year_options else None

    if selected_year:
        row = shap_top[(shap_top['Symbol'].astype(str) == str(selected_symbol)) &
                       (shap_top['violation_year'] == selected_year)]
        if len(row) > 0:
            row = row.iloc[0]
            st.markdown(f"### {row['ShortName']} ({row['Symbol']}) - {selected_year} 年")
            st.markdown(f"**行业**:{row['industry']}  |  **p_ml**:{row['p_ml']:.4f}")

            top_data = pd.DataFrame({
                '特征': [row['top1_feature'], row['top2_feature'], row['top3_feature']],
                'SHAP 值(对预测的边际贡献)': [row['top1_shap'], row['top2_shap'], row['top3_shap']],
                '特征实际取值': [row['top1_value'], row['top2_value'], row['top3_value']],
            })

            col_t1, col_t2 = st.columns([3, 2])
            with col_t1:
                fig = px.bar(
                    top_data, x='特征', y='SHAP 值(对预测的边际贡献)',
                    color='SHAP 值(对预测的边际贡献)',
                    color_continuous_scale='RdBu_r',
                    title='Top3 特征 SHAP 贡献(正值=推高风险,负值=降低风险)'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with col_t2:
                st.dataframe(
                    top_data.style.format({
                        'SHAP 值(对预测的边际贡献)': '{:.4f}',
                        '特征实际取值': '{:.4f}',
                    }),
                    use_container_width=True, hide_index=True
                )

            st.info(
                f"💡 **解读**:本样本中,对风险贡献最大的特征是 `{row['top1_feature']}`(贡献 +{row['top1_shap']:.4f}),"
                f"其次 `{row['top2_feature']}`(贡献 +{row['top2_shap']:.4f}) 和 `{row['top3_feature']}`(贡献 +{row['top3_shap']:.4f})。"
                f"建议审计师在核查该公司时,优先关注 `{row['top1_feature']}` 指标,深入追查该指标异常的业务原因。"
            )
        else:
            st.warning('所选公司/年份组合无 SHAP 数据')

    st.markdown('---')

    with st.expander('📚 关于 SHAP 的方法论说明', expanded=False):
        st.markdown('''
**SHAP 是什么**
- SHAP(SHapley Additive exPlanations)基于博弈论中的 Shapley 值,公平分配每个特征对单次预测的贡献
- 它能回答"为什么这个公司被模型标记为高风险"的反事实问题

**业务价值**
- **可解释性**:从黑盒预测到透明决策,符合监管对模型可解释性的要求
- **审计抽样依据**:SHAP 贡献最大的特征,可作为审计师现场核查的优先切入点
- **模型校准**:长期积累 SHAP 解释,可发现模型系统性偏差

**当前局限**
- 当前仅用 7 个财务比率,SHAP 分析的"宽度"受限
- SHAP 反映统计相关性,不直接等于因果关系
- 输出基于 1000 家样本,完整 8000+ 公司尚未覆盖
        ''')


# Tab 8: 时序特征工程(阶段四优化 7)
# ============================================================
with tab8:
    st.header('⏱️ 时序特征工程')
    st.caption('为每家公司提取违规前 t-1/t-2/t-3 年的财务比率 + yoy/CAGR/volatility 衍生指标')

    # 加载数据
    trace_v2 = pd.read_csv(os.path.join(DATA_DIR, 'audit_score_traceability_v2.csv'))
    temporal_df = pd.read_csv(os.path.join(DATA_DIR, 'temporal_features_v2.csv'))

    # 子区块 1: 覆盖度诊断
    st.subheader('1. 时序特征覆盖度诊断')
    col1, col2, col3 = st.columns(3)
    with col1:
        n_total = len(temporal_df)
        st.metric('总样本数', f'{n_total:,}')
    with col2:
        t1_cov = temporal_df['roe_t1'].notna().mean() * 100
        st.metric('t-1 年数据覆盖', f'{t1_cov:.1f}%', help='有 t-1 年财务数据的样本占比')
    with col3:
        cagr_cov = temporal_df['roe_cagr3y'].notna().mean() * 100
        st.metric('3 年 CAGR 覆盖', f'{cagr_cov:.1f}%', help='需要 3 年连续数据的样本占比')

    # 子区块 2: 时序特征定义说明
    st.markdown('---')
    st.subheader('2. 时序特征定义')

    st.markdown('''
**56 维时序特征**(从 raw_financials.csv 的 7 个比率衍生):

| 类型 | 公式 | 特征数 |
|---|---|---|
| 跨年值 | t-1, t-2, t-3 年的 7 比率值 | 21 |
| 同比增速 | yoy = (v_{t-1} - v_{t-2}) / |v_{t-2}| | 7 |
| 3 年复合增长 | CAGR = (v_{t-1}/v_{t-3})^(1/2) - 1 | 7 |
| 3 年波动率 | std(t-1, t-2, t-3) | 7 |
| 趋势方向 | sign(平均差分) | 7 |
| 趋势加速度 | yoy(t-1) - yoy(t-2) | 7 |
''')

    # 子区块 3: 时序 vs Baseline 模型对比
    st.markdown('---')
    st.subheader('3. 时序 vs Baseline 模型对比(5 折交叉验证)')

    if os.path.exists(os.path.join(OUT_DIR, 'temporal_vs_baseline.csv')):
        comp = pd.read_csv(os.path.join(OUT_DIR, 'temporal_vs_baseline.csv'), index_col=0)
        st.dataframe(
            comp.style.format('{:.4f}').background_gradient(cmap='RdYlGn', subset=['f1', 'recall']),
            use_container_width=True
        )

        st.info('''
💡 **关键发现**:时序增强模型(63 维)相比 baseline(7 维)在当前数据下**未带来提升**,F1 从 0.8385 略降至 0.8343。

可能原因:
- 时序特征 NaN 占比高(t-2, t-3 不到 30%),中位数填充引入噪音
- 现有 7 个比率已包含足够信号,模型容量饱和
- RandomForest 在高维时 `max_features='sqrt'` 反而稀释了关键特征
''')
    else:
        st.warning('temporal_vs_baseline.csv 不存在,请先运行 train_temporal_rf.py')

    # 子区块 4: 单公司时序画像
    st.markdown('---')
    st.subheader('4. 单公司时序画像')

    # 公司选择(挑有时序数据的)
    candidates = temporal_df[temporal_df['roe_t1'].notna()]['Symbol'].astype(str).unique()
    if len(candidates) == 0:
        st.warning('没有符号有完整的 t-1 年数据')
    else:
        # 按 p_ml 排
        if 'p_ml' in trace_v2.columns:
            merged = temporal_df.merge(
                trace_v2[['Symbol', 'violation_year', 'p_ml', 'ShortName']],
                on=['Symbol', 'violation_year'], how='left'
            )
            candidates_with_p = merged[merged['roe_t1'].notna() & merged['p_ml'].notna()]
            candidates_with_p = candidates_with_p.sort_values('p_ml', ascending=False)
            sel_options = (
                candidates_with_p['ShortName'].astype(str) + ' (' +
                candidates_with_p['Symbol'].astype(str) + ')'
            ).tolist()
            sel_label_map = dict(zip(sel_options, zip(
                candidates_with_p['Symbol'].astype(str),
                candidates_with_p['violation_year']
            )))
        else:
            sel_options = sorted(candidates)[:200]
            sel_label_map = {c: (c, None) for c in sel_options}

        sel = st.selectbox('选择公司(按 p_ml 降序)', options=sel_options[:200])
        if sel:
            symbol, vy = sel_label_map[sel]
            panel_wide = pd.read_csv(os.path.join(DATA_DIR, 'temporal_panel_wide.csv'))
            panel_wide['Symbol'] = panel_wide['Symbol'].astype(str).str.zfill(6)
            sym_panel = panel_wide[panel_wide['Symbol'] == symbol].sort_values('year')

            if len(sym_panel) > 0:
                # 多年趋势图
                ratio_to_show = st.selectbox(
                    '选择指标',
                    options=['roe', 'roa', 'debt_ratio', 'net_margin', 'ocf_to_rev'],
                    index=0
                )
                fig = px.line(
                    sym_panel, x='year', y=ratio_to_show,
                    markers=True,
                    title=f"{sel} - {ratio_to_show} 多年走势"
                )
                fig.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)

                # 显示该公司所有时序特征值
                if vy is not None:
                    row = temporal_df[(temporal_df['Symbol'].astype(str) == str(symbol)) &
                                     (temporal_df['violation_year'] == vy)]
                    if len(row) > 0:
                        st.markdown(f'**违规年份**:{vy} 年')
                        with st.expander('该公司所有 56 维时序特征', expanded=False):
                            row_t = row.drop(columns=['Symbol', 'violation_year']).T
                            row_t.columns = ['value']
                            st.dataframe(row_t, use_container_width=True)

    # 子区块 5: 数据可得性提醒
    st.markdown('---')
    with st.expander('📚 时序特征的方法论说明与局限', expanded=False):
        st.markdown('''
**为什么时序特征没有显著提升?**
- **数据稀疏**:t-3 年数据覆盖率仅约 3%(需要 4 年连续数据)
- **中位数填充噪音**:SimpleImputer 用中位数填充 NaN,可能掩盖真实模式
- **模型容量饱和**:RandomForest 在 7 维已经能捕捉主要信号,新增维度边际收益小
- **特征冗余**:yoy、CAGR、trend 等衍生指标高度相关,新增信息有限

**改进方向**
- 使用时序深度学习模型(LSTM/Transformer)而非 RandomForest
- 引入外部数据(行业平均、宏观指标)做相对偏离度
- 引入更长窗口(5-10 年)的累积信号

**当前局限**
- 完整时序面板只覆盖 2005-2025 年,部分老公司数据缺失
- 早期违规(2010 年前)历史数据尤其稀缺
- 当前未考虑行业平均值的相对偏离(同类公司比较)
''')


# 底部
# ============================================================
st.markdown('---')
st.caption('持续审计系统 v2.2 | XGBoost (主力,F1=0.853) + 22 规则 + 123 关键词 | 训练数据: 8,724 标注 + 7,997 财务特征 | 阶段四优化: R6 规则 + 行业兜底 + contamination=0.10 + SHAP 集成 + 时序特征(56维)+ XGBoost 替换 RF')

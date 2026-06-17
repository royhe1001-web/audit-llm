"""
app/demo.py — 演讲后交互演示页面
====================================
两种模式:
  1. 公司查询:输入 Symbol,展示该公司的风险评分、规则触发、SHAP 解释
  2. 自由预测:手动输入 7 个财务比率,实时预测风险概率

启动命令: streamlit run app/demo.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
OUT_DIR = os.path.join(BASE, "output")

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio',
            'asset_turnover', 'net_margin', 'ocf_to_rev']

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title='AuditLLM 实时演示',
    page_icon='🔍',
    layout='wide',
)

st.title('🔍 AuditLLM — 实时舞弊风险演示')
st.caption('基于 v2.3.1 XGBoost 模型(113 维特征,F1=0.872)· 输入财务数据即可获得风险评估')

# ============================================================
# 加载模型与数据(缓存)
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(MODEL_DIR, 'fraud_detection_xgb_combined.pkl'))

@st.cache_resource
def load_metadata():
    p = os.path.join(MODEL_DIR, 'fraud_detection_xgb_combined_metadata.pkl')
    if os.path.exists(p):
        return joblib.load(p)
    return None

@st.cache_data
def load_companies():
    df = pd.read_csv(os.path.join(DATA_DIR, 'audit_intermediate_table.csv'))
    return df

model = load_model()
metadata = load_metadata()
companies = load_companies()

# 显示模型版本
if metadata:
    st.sidebar.success(f"✅ 模型版本:{metadata.get('version', 'unknown')} | 特征:{metadata.get('n_features', '?')} 维")
else:
    st.sidebar.info("✅ 模型:v2.3.1 XGBoost")

# ============================================================
# 模式切换
# ============================================================
mode = st.sidebar.radio(
    '选择演示模式',
    ['🔎 公司查询', '🧮 自由预测'],
    help='公司查询:从已有数据中查;自由预测:手动输入财务比率',
)

# ============================================================
# 模式 1:公司查询
# ============================================================
if mode == '🔎 公司查询':
    st.header('1. 公司风险查询')

    # 双输入方式:快速选择 + 自由输入
    col1, col2 = st.columns([3, 2])
    with col1:
        # 方式 1:下拉选择(从已有公司选)
        selected_from_list = st.selectbox(
            '方式 1 · 从已覆盖公司快速选择',
            options=[None] + sorted(companies['Symbol'].astype(str).unique().tolist()),
            index=0,
            format_func=lambda x: '— 请选择 —' if x is None else x,
        )
    with col2:
        # 方式 2:自由输入任意 Symbol
        custom_symbol = st.text_input(
            '方式 2 · 自由输入 Symbol',
            placeholder='6 位股票代码(如 600715)',
            max_chars=6,
        ).strip()

    # 决定用哪个
    symbol = custom_symbol if custom_symbol else selected_from_list

    if not symbol:
        st.info('👆 请选择一家公司或直接输入 6 位股票代码')
        # 显示已知公司统计
        st.caption(f'已覆盖公司:{companies["Symbol"].nunique()} 家(共 {len(companies)} 行历史数据)')
        st.stop()

    # 标准化 Symbol
    symbol = str(symbol).zfill(6)
    sym_data = companies[companies['Symbol'].astype(str) == str(symbol)]
    if len(sym_data) == 0:
        # Symbol 不在数据库,提供解决方案
        st.error(f'❌ 未找到 Symbol **{symbol}** 的数据')
        st.markdown(f"""
**可能原因**:
- 该 Symbol 不在已覆盖的 {companies["Symbol"].nunique()} 家公司中
- 可能输入有误(如填了港股 0700,而非 A 股 6 位代码)

**解决方案**:
1. **检查输入**:确认是 6 位 A 股代码(如 `600715`、`300750`),前面加 0 补齐 6 位
2. **换家公司**:从上方下拉列表中选择已有公司
3. **拉取数据**(高级):如果想查新公司,可运行:
   ```bash
   cd /Users/Zhuanz/claude工作文件夹/审计数据分析大作业
   python scripts/pull_financials.py  # 拉取该公司历史财务数据
   ```
   然后重启 demo,即可查询。
""")
        st.stop()

    # 有数据
    sym_summary = sym_data.sort_values('violation_year', ascending=False).head(3)

    # 显示公司基本信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('公司', sym_data['ShortName'].iloc[0])
    with col2:
        st.metric('行业', sym_data['industry'].iloc[0])
    with col3:
        st.metric('已知违规', '✅ 是' if sym_data['ann_fin_flag'].max() == 1 else '否')
    with col4:
        st.metric('风险分', f"{sym_data['risk_score'].max():.3f}")

    st.markdown('---')

    # 历史年份风险趋势
    st.subheader('📊 历年风险评估')
    yearly = sym_data.sort_values('violation_year').tail(10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly['violation_year'],
        y=yearly['risk_score'],
        mode='lines+markers',
        name='综合风险分',
        line=dict(color='#185FA5', width=3),
        marker=dict(size=10),
    ))
    fig.add_trace(go.Scatter(
        x=yearly['violation_year'],
        y=yearly['p_ml'],
        mode='lines+markers',
        name='ML 概率',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
    ))
    fig.add_hline(y=0.55, line_dash='dot', line_color='red',
              annotation_text='高风险阈值')
    fig.update_layout(
        xaxis_title='年份', yaxis_title='分数',
        height=350, hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

    # 触发的规则
    st.subheader('📋 触发的审计规则')
    rule_data = sym_data[['violation_year', 'rule_ids', 'n_rules_triggered', 'p_ml', 'risk_score']].head(5)
    st.dataframe(rule_data, use_container_width=True, hide_index=True)

    # SHAP 解释(如果有)
    shap_path = os.path.join(DATA_DIR, 'shap_top_features.csv')
    if os.path.exists(shap_path):
        shap_data = pd.read_csv(shap_path)
        sym_shap = shap_data[shap_data['Symbol'].astype(str) == str(symbol)]
        if len(sym_shap) > 0:
            st.subheader('🧬 SHAP 特征贡献(Top 3)')
            row = sym_shap.iloc[0]
            shap_df = pd.DataFrame({
                '特征': [row['top1_feature'], row['top2_feature'], row['top3_feature']],
                'SHAP 值(对预测的边际贡献)': [row['top1_shap'], row['top2_shap'], row['top3_shap']],
                '特征实际取值': [row['top1_value'], row['top2_value'], row['top3_value']],
            })
            fig = go.Figure(go.Bar(
                x=shap_df['SHAP 值(对预测的边际贡献)'],
                y=shap_df['特征'],
                orientation='h',
                marker_color=['#d62728' if v > 0 else '#2ca02c' for v in shap_df['SHAP 值(对预测的边际贡献)']],
                text=[f"{v:.4f}" for v in shap_df['SHAP 值(对预测的边际贡献)']],
                textposition='outside',
            ))
            fig.update_layout(
                xaxis_title='SHAP 值(正值=推高风险)',
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption('该公司暂无 SHAP 解释数据')

# ============================================================
# 模式 2:自由预测(基于基准公司的 113 维特征)
# ============================================================
else:
    st.header('2. 自由预测(特征敏感性演示)')

    st.caption('以一家**健康公司**的 113 维特征为基准,通过滑块修改关键 7 个财务比率,实时观察风险评分如何变化')

    # 选基准公司(默认贵州茅台,代表健康公司)
    # 4 个基准公司全部有 113 维完整数据(comp + fina + pledge 都有)
    baseline_options = {
        '600519.SH 贵州茅台(酿酒,头部)': '600519',
        '000858.SZ 五粮液(酿酒)': '000858',
        '600036.SH 招商银行(银行)': '600036',
        '000651.SZ 格力电器(家电)': '000651',
    }
    baseline_name = st.selectbox('选择基准公司', list(baseline_options.keys()))
    baseline_symbol = baseline_options[baseline_name]

    # 加载基准公司的 113 维特征
    @st.cache_data
    def load_baseline_features(symbol):
        """从多源数据加载该公司 113 维特征,含 fallback 补全"""
        # === 加载所有数据源 ===
        full_df = pd.read_csv(os.path.join(DATA_DIR, 'fina_indicator_full.csv'))
        full_df['Symbol'] = full_df['ts_code'].str.split('.').str[0]
        full_df['feature_year'] = full_df['end_date'].astype(str).str[:4].astype(int)

        pled = pd.read_csv(os.path.join(DATA_DIR, 'pledge_stat_full.csv'))
        pled['Symbol'] = pled['ts_code'].str.split('.').str[0]
        pled['feature_year'] = pled['end_date'].astype(str).str[:4].astype(int)
        pled_keep = ['Symbol', 'feature_year', 'pledge_count', 'unrest_pledge',
                     'rest_pledge', 'total_share', 'pledge_ratio']
        pled = pled[pled_keep].drop_duplicates(subset=['Symbol', 'feature_year'], keep='last')

        comp = pd.read_csv(os.path.join(DATA_DIR, 'fraud_features_combined.csv'))
        comp['Symbol'] = comp['Symbol'].astype(str).str.zfill(6)
        comp['feature_year'] = (comp['violation_year'] - 1).astype(int)

        # 新增:raw_financials(用于 ocf_to_rev 补全)
        raw_path = os.path.join(DATA_DIR, 'raw_financials.csv')
        if os.path.exists(raw_path):
            raw = pd.read_csv(raw_path)
            raw['Symbol'] = raw['fetch_symbol'].astype(str).str.zfill(6)
            raw['feature_year'] = raw['fetch_year'].astype(int)
        else:
            raw = pd.DataFrame()

        # === 选择最佳 feature_year ===
        comp_sym = comp[comp['Symbol'] == symbol].sort_values('feature_year', ascending=False)
        candidate_years = sorted(comp_sym['feature_year'].unique(), reverse=True) if len(comp_sym) > 0 else []

        # 如果 comp 无记录,尝试从 fina 或 raw 找年份
        if not candidate_years:
            alt_years = set(full_df[full_df['Symbol'] == symbol]['feature_year'].tolist() +
                            raw[raw['Symbol'] == symbol]['feature_year'].tolist() if len(raw) > 0 else [])
            candidate_years = sorted(alt_years, reverse=True)

        if not candidate_years:
            return None

        # 优先选同时有 fina + pledge 数据的年份
        feature_year = None
        for yr in candidate_years:
            has_fina = len(full_df[(full_df['Symbol'] == symbol) & (full_df['feature_year'] == yr)]) > 0
            has_pled = len(pled[(pled['Symbol'] == symbol) & (pled['feature_year'] == yr)]) > 0
            if has_fina and has_pled:
                feature_year = int(yr)
                break
        if feature_year is None:
            for yr in candidate_years:
                if len(full_df[(full_df['Symbol'] == symbol) & (full_df['feature_year'] == yr)]) > 0:
                    feature_year = int(yr)
                    break
        if feature_year is None:
            feature_year = int(candidate_years[0])

        # === 构建 7 比率(fallback 链)===
        # 来源 1:comp
        comp_row = comp_sym[comp_sym['feature_year'] == feature_year]
        comp_7 = {}
        if len(comp_row) > 0:
            comp_7 = {c: comp_row.iloc[0].get(c) for c in FIN_COLS}

        # 来源 2:fina_indicator(6 比率,无 ocf_to_rev)
        fina_row = full_df[(full_df['Symbol'] == symbol) & (full_df['feature_year'] == feature_year)]
        fina_7 = {}
        if len(fina_row) > 0:
            row = fina_row.iloc[0]
            fina_7 = {
                'roe': row.get('roe'),
                'roa': row.get('roa'),
                'debt_ratio': row.get('debt_to_assets'),  # fina 字段名
                'current_ratio': row.get('current_ratio'),
                'asset_turnover': row.get('assets_turn'),  # fina 字段名
                'net_margin': row.get('netprofit_margin'),  # fina 字段名
                # fina 没有 ocf_to_rev
            }

        # 来源 3:raw_financials(派生 ocf_to_rev)
        raw_row = raw[(raw['Symbol'] == symbol) & (raw['feature_year'] == feature_year)] if len(raw) > 0 else pd.DataFrame()
        raw_7 = {}
        if len(raw_row) > 0:
            row = raw_row.iloc[0]
            ocf = pd.to_numeric(row.get('ocf'), errors='coerce')
            rev = pd.to_numeric(row.get('revenue'), errors='coerce')
            if pd.notna(ocf) and pd.notna(rev) and rev != 0:
                raw_7['ocf_to_rev'] = float(ocf) / float(rev)
            # raw 还有其他 7 比率字段
            for src_c, dst_c in [('roe', 'roe'), ('roa', 'roa'),
                                  ('debt_ratio', 'debt_ratio'),
                                  ('current_ratio', 'current_ratio'),
                                  ('asset_turnover', 'asset_turnover'),
                                  ('net_margin', 'net_margin')]:
                v = row.get(src_c)
                if pd.notna(v):
                    raw_7[dst_c] = v

        # 来源 4:该公司其他年份平均
        other_years = comp_sym[comp_sym['feature_year'] != feature_year]
        other_avg = {}
        for c in FIN_COLS:
            vals = pd.to_numeric(other_years[c], errors='coerce').dropna()
            if len(vals) > 0:
                other_avg[c] = float(vals.mean())

        # === 多源 fallback 链 ===
        # 优先级:comp > raw > fina > 其他年份平均 > 行业平均
        # 计算行业平均(防止所有都缺)
        industry_avg = {}
        if len(comp_row) > 0 and 'industry' in comp_row.columns:
            ind = comp_row.iloc[0].get('industry')
            if ind:
                same_ind = comp[comp['industry'] == ind]
                for c in FIN_COLS:
                    vals = pd.to_numeric(same_ind[c], errors='coerce').dropna()
                    if len(vals) > 0:
                        industry_avg[c] = float(vals.median())

        # 全局默认(健康公司典型值)
        global_default = {
            'roe': 0.15, 'roa': 0.10, 'debt_ratio': 0.30,
            'current_ratio': 2.0, 'asset_turnover': 1.0,
            'net_margin': 0.15, 'ocf_to_rev': 0.15
        }

        # fallback 链
        features = {}
        for c in FIN_COLS:
            val = None
            for source in [comp_7.get(c), raw_7.get(c), fina_7.get(c),
                          other_avg.get(c), industry_avg.get(c),
                          global_default.get(c)]:
                if val is None and pd.notna(source):
                    val = source
            features[c] = val

        # === 113 维 fina + pledge ===
        fina_sym = full_df[(full_df['Symbol'] == symbol) & (full_df['feature_year'] == feature_year)]
        fina_cols = [c for c in full_df.columns if c not in ['ts_code', 'ann_date', 'end_date', 'Symbol', 'feature_year']]

        pled_sym = pled[(pled['Symbol'] == symbol) & (pled['feature_year'] == feature_year)]

        for c in fina_cols:
            if c in fina_sym.columns and len(fina_sym) > 0:
                features[c] = fina_sym[c].iloc[0]
            else:
                features[c] = np.nan

        # pledge 字段:即使无数据也要保留 key(填 0)
        for c in pled_keep:
            if c not in ['Symbol', 'feature_year']:
                if c in pled_sym.columns and len(pled_sym) > 0:
                    features[c] = pled_sym[c].iloc[0]
                else:
                    features[c] = 0

        return pd.Series(features)

    # 加载 metadata 获取模型特征列表
    @st.cache_data
    def load_model_features():
        if metadata and 'feature_names' in metadata:
            return list(metadata['feature_names'])
        return FIN_COLS

    model_features = load_model_features()
    baseline_features = load_baseline_features(baseline_symbol)

    if baseline_features is None:
        st.error(f'未找到 {baseline_name} 的 113 维特征数据')
    else:
        # 用滑块覆盖 7 个关键维度
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**盈利能力**')
            roe = st.slider('ROE', -2.0, 1.0, float(baseline_features.get('roe', 0.15)), 0.01,
                           help='ROE = 净利润 / 股东权益')
            roa = st.slider('ROA', -1.0, 0.5, float(baseline_features.get('roa', 0.10)), 0.01,
                           help='ROA = 净利润 / 总资产')
            net_margin = st.slider('Net Margin', -1.0, 1.0, float(baseline_features.get('net_margin', 0.15)), 0.01,
                                  help='Net Margin = 净利润 / 营收')

        with col2:
            st.markdown('**财务结构**')
            debt_ratio = st.slider('Debt Ratio', 0.0, 1.5, float(baseline_features.get('debt_ratio', 0.30)), 0.01,
                                  help='Debt Ratio = 总负债 / 总资产')
            current_ratio = st.slider('Current Ratio', 0.0, 5.0, float(baseline_features.get('current_ratio', 2.0)), 0.1,
                                    help='Current Ratio = 流动资产 / 流动负债')
            asset_turnover = st.slider('Asset Turnover', 0.0, 5.0, float(baseline_features.get('asset_turnover', 1.0)), 0.1,
                                      help='Asset Turnover = 营收 / 总资产')

        st.markdown('**现金流**')
        ocf_to_rev = st.slider('OCF to Revenue', -1.0, 1.0, float(baseline_features.get('ocf_to_rev', 0.15)), 0.01,
                              help='OCF to Revenue = 经营活动现金流 / 营收')

        # 关键提示
        st.info(f'💡 以 **{baseline_name}** 为基准(其他 106 维特征保持不变),仅修改上述 7 个比率')

        # 构建 113 维特征向量:基础 7 + fina 101 + pledge 5
        X_input = baseline_features.copy().to_frame().T
        # 用滑块值覆盖基础 7 个比率
        X_input['roe'] = roe
        X_input['roa'] = roa
        X_input['debt_ratio'] = debt_ratio
        X_input['current_ratio'] = current_ratio
        X_input['asset_turnover'] = asset_turnover
        X_input['net_margin'] = net_margin
        X_input['ocf_to_rev'] = ocf_to_rev
        # 重新对齐到模型特征顺序
        X_input = X_input[model_features]

        # 预测
        p_fraud = model.predict_proba(X_input.values)[0, 1]

        st.markdown('---')
        st.subheader('🎯 预测结果')

        if p_fraud >= 0.55:
            risk_level = '🔴 高风险'
            color = '#d62728'
        elif p_fraud >= 0.25:
            risk_level = '🟡 中风险'
            color = '#ff7f0e'
        else:
            risk_level = '🟢 低风险'
            color = '#2ca02c'

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<h1 style='text-align:center; color:{color};'>{p_fraud:.3f}</h1>",
                        unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>舞弊概率</p>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h2 style='text-align:center;'>{risk_level}</h2>",
                        unsafe_allow_html=True)
        with col3:
            if p_fraud >= 0.55:
                st.error('⚠️ 超过高风险阈值 0.55,建议立即审计介入')
            elif p_fraud >= 0.25:
                st.warning('⚡ 中风险,持续监控')
            else:
                st.success('✅ 低风险,常规监控')

        # gauge 图
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=float(p_fraud),
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text='风险概率'),
            gauge=dict(
                axis=dict(range=[0, 1]),
                bar=dict(color=color),
                steps=[
                    {'range': [0, 0.25], 'color': '#e8f5e9'},
                    {'range': [0.25, 0.55], 'color': '#fff3e0'},
                    {'range': [0.55, 1], 'color': '#ffebee'},
                ],
                threshold=dict(line=dict(color='red', width=4), thickness=0.75, value=0.55),
            ),
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

        # 与原始公司对比
        st.subheader('📊 与基准公司对比')
        orig_score = baseline_features.name if hasattr(baseline_features, 'name') else None
        # 找基准公司的实际历史风险分
        sym_data = companies[companies['Symbol'].astype(str) == str(baseline_symbol)]
        if len(sym_data) > 0:
            orig_risk = sym_data['risk_score'].mean()
            st.metric(f'{baseline_name} 历史平均风险分', f'{orig_risk:.3f}')
        st.metric('当前修改后风险分', f'{p_fraud:.3f}',
                  delta=f'{p_fraud - orig_risk:.3f}' if len(sym_data) > 0 else None)

        # 试算案例(覆盖滑块值)
        with st.expander('💡 试算案例(点击加载预设值)'):
            col1, col2, col3 = st.columns(3)
            if col1.button('📈 舞弊公司典型(高风险)'):
                st.session_state.update({
                    'roe': -0.30, 'roa': -0.15, 'net_margin': -0.20,
                    'debt_ratio': 0.85, 'current_ratio': 0.6, 'asset_turnover': 0.5,
                    'ocf_to_rev': -0.10,
                })
                st.rerun()
            if col2.button('🏥 健康公司(低风险)'):
                st.session_state.update({
                    'roe': 0.20, 'roa': 0.10, 'net_margin': 0.15,
                    'debt_ratio': 0.30, 'current_ratio': 2.5, 'asset_turnover': 1.2,
                    'ocf_to_rev': 0.20,
                })
                st.rerun()
            if col3.button('⚠️ 边缘公司(中风险)'):
                st.session_state.update({
                    'roe': 0.02, 'roa': 0.01, 'net_margin': 0.02,
                    'debt_ratio': 0.72, 'current_ratio': 0.95, 'asset_turnover': 0.4,
                    'ocf_to_rev': -0.05,
                })
                st.rerun()

# ============================================================
# 页脚
# ============================================================
st.markdown('---')
st.caption('AuditLLM 交互演示 v1.0 | 基于 v2.3.1 XGBoost | 5 折 CV F1=0.872')
st.caption('技术栈:Python 3.13 + Streamlit 1.58 + XGBoost 3.2 | github.com/royhe1001-web/audit-llm')
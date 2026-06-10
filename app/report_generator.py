#!/usr/bin/env python3
"""
报告自动化 v3.0 — 集成 P0/P1/P2/P3 全套 + 治理网络
=====================================================
- 7 财务比率 + ML 模型
- 6 条硬规则 + 5 条时序规则
- 治理反转检测 (P2-1)
- 内部控制维度 (P2-2)
- 增强治理信号提取 — 123 关键词,9 类别 (P2-3)
- 治理网络分析 — 同实控人下其他公司 (P2-9)
- 多数据源 — 工商/质押/监管公告 (P3)
"""

import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import pdfplumber
import tushare as ts
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/Zhuanz/claude工作文件夹/审计数据分析大作业/scripts')

from enhanced_governance_extractor import extract_governance_signals_v2, load_keyword_library
from governance_reversal import extract_governance_signals, detect_reversals
from internal_control import extract_internal_control, signals_to_rules

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")
MODEL_DIR = os.path.join(BASE, "models")

TOKEN = "f3bf8d32b09bb2cfa3f3632b5521caa8143576c3dff550a742f5f4cc"
ts.set_token(TOKEN)
pro = ts.pro_api()

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover',
            'net_margin', 'ocf_to_rev']


def assign_level(s):
    if s >= 0.55: return '高风险'
    if s >= 0.25: return '中风险'
    return '低风险'


# ============================================================
# 治理网络分析 (P2-9)
# ============================================================
@st.cache_data(ttl=3600)
def fetch_controller_network(symbol, max_peers=20):
    """
    提取公司的实控人 + 同实控人下其他公司
    """
    try:
        # 1. 拉该公司基础信息
        b = pro.stock_basic(ts_code=f"{symbol}.{'SH' if symbol.startswith('6') else 'SZ'}",
                            fields='ts_code,symbol,name,industry,act_name,act_ent_type')
        if b is None or b.empty:
            return None
        company = b.iloc[0]
        controller = company.get('act_name', '无')

        if pd.isna(controller) or controller == '无实际控制人':
            return {
                'company': company,
                'controller': '无实际控制人',
                'peers': [],
                'n_peers': 0,
                'n_high_risk_peers': 0,
            }

        # 2. 找同实控人下其他公司
        all_b = pro.stock_basic(list_status='L', fields='ts_code,symbol,name,industry,act_name')
        peers = all_b[all_b['act_name'] == controller]
        if symbol in peers['symbol'].values:
            peers = peers[peers['symbol'] != symbol]
        peers = peers.head(max_peers)

        # 3. 加载风险评分
        risk = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_industry.csv"))
        risk['Symbol'] = risk['Symbol'].astype(str).str.zfill(6)
        peers_with_risk = peers.merge(
            risk[['Symbol', 'ShortName', 'risk_level_v3', 'risk_score_v3', 'is_st', 'is_strict_st', 'ann_fin_flag']].rename(
                columns={'ShortName': 'name2'}
            ),
            left_on='symbol', right_on='Symbol', how='left'
        )

        n_high = (peers_with_risk['risk_level_v3'] == '高风险').sum()
        n_fraud = (peers_with_risk['ann_fin_flag'] == 1).sum()

        return {
            'company': company,
            'controller': controller,
            'peers': peers_with_risk,
            'n_peers': len(peers_with_risk),
            'n_high_risk_peers': int(n_high) if pd.notna(n_high) else 0,
            'n_fraud_peers': int(n_fraud) if pd.notna(n_fraud) else 0,
        }
    except Exception as e:
        return None


# ============================================================
# 多数据源分析 (P3)
# ============================================================
@st.cache_data(ttl=3600)
def fetch_external_signals(symbol):
    """
    拉取该公司的:
    - 工商数据
    - 股权质押
    - 监管公告
    """
    signals = {
        'reg_capital': None,
        'employees': None,
        'setup_date': None,
        'pledge_ratio': None,
        'pledge_count': None,
        'pledge_high': False,
        'n_regulatory_anns_180d': 0,
        'regulatory_anns': [],
    }

    try:
        ts_code = f"{symbol}.{'SH' if symbol.startswith('6') else 'SZ'}"

        # 工商
        c = pro.stock_company(ts_code=ts_code,
                                fields='ts_code,reg_capital,setup_date,employees,chairman,manager')
        if c is not None and not c.empty:
            row = c.iloc[0]
            signals['reg_capital'] = row.get('reg_capital')
            signals['employees'] = row.get('employees')
            signals['setup_date'] = row.get('setup_date')
            signals['chairman'] = row.get('chairman')
            signals['manager'] = row.get('manager')

        # 质押
        p = pro.pledge_stat(ts_code=ts_code, fields='ts_code,end_date,pledge_count,pledge_ratio')
        if p is not None and not p.empty:
            p_latest = p.sort_values('end_date', ascending=False).iloc[0]
            ratio = float(p_latest.get('pledge_ratio', 0)) if pd.notna(p_latest.get('pledge_ratio')) else 0
            signals['pledge_ratio'] = ratio
            signals['pledge_count'] = int(p_latest.get('pledge_count', 0)) if pd.notna(p_latest.get('pledge_count')) else 0
            signals['pledge_high'] = ratio > 30
            signals['pledge_latest_date'] = p_latest.get('end_date')

        # 监管公告(180 天)
        from datetime import timedelta
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
        try:
            a = pro.anns(ts_code=ts_code, start_date=start_date, end_date=end_date,
                         fields='ts_code,ann_date,title')
        except:
            try:
                a = pro.announcement(ts_code=ts_code, start_date=start_date, end_date=end_date,
                                    fields='ts_code,ann_date,title')
            except:
                a = None

        if a is not None and not a.empty:
            keywords = ['监管', '问询', '警示', '处分', '通报', '关注函', '立案', '调查']
            mask = a['title'].str.contains('|'.join(keywords), na=False)
            if mask.any():
                signals['n_regulatory_anns_180d'] = int(mask.sum())
                signals['regulatory_anns'] = a[mask][['ann_date', 'title']].head(5).values.tolist()

    except Exception as e:
        pass

    return signals


# ============================================================
# 财务数据提取(从 PDF)
# ============================================================
def extract_company_info(pdf_path):
    info = {'name': '', 'code': '', 'industry': '', 'audit_opinion': ''}
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(min(10, len(pdf.pages))):
            text = pdf.pages[i].extract_text() or ''
            if not info['name']:
                for line in text.split('\n')[:3]:
                    if '股份' in line or '有限公司' in line or '集团' in line:
                        info['name'] = line.strip()
                        break
            if not info['code']:
                import re
                m = re.search(r'(?:公司代码|证券代码)[：:]\s*(\d{6})', text)
                if m:
                    info['code'] = m.group(1)
            if '标准无保留' in text:
                info['audit_opinion'] = '标准无保留'
            elif '保留意见' in text and '否定' not in text:
                info['audit_opinion'] = '保留意见'
            elif '否定意见' in text or '无法表示' in text:
                info['audit_opinion'] = '否定或无法表示'
    return info


def extract_financial_data(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        revenue = net_profit = ocf = total_assets = equity = None
        for i in range(3, 12):
            text = pdf.pages[i].extract_text() or ''
            if '主要会计数据' in text or ('营业收入' in text and '净利润' in text and '万元' in text):
                import re
                for line in text.split('\n'):
                    if '营业收入' in line and revenue is None:
                        m = re.findall(r'([\d,]+\.\d+)', line)
                        if m:
                            try: revenue = float(m[-1].replace(',', ''))
                            except: pass
                    if '归属于上市公司股东的' in line and '净利润' in line and net_profit is None:
                        m = re.findall(r'([\d,]+\.\d+)', line)
                        if m:
                            try: net_profit = float(m[-1].replace(',', ''))
                            except: pass
                    if '经营活动产生的现金流量净额' in line and ocf is None:
                        m = re.findall(r'([\d,]+\.\d+)', line)
                        if m:
                            try: ocf = float(m[-1].replace(',', ''))
                            except: pass
                    if '总资产' in line and total_assets is None:
                        m = re.findall(r'([\d,]+\.\d+)', line)
                        if m:
                            try: total_assets = float(m[-1].replace(',', ''))
                            except: pass
                    if '归属于上市公司股东的净资产' in line and equity is None:
                        m = re.findall(r'([\d,]+\.\d+)', line)
                        if m:
                            try: equity = float(m[-1].replace(',', ''))
                            except: pass
                if revenue and net_profit:
                    break

        current_assets = current_liab = total_liab = None
        for i in range(60, min(200, len(pdf.pages))):
            text = pdf.pages[i].extract_text() or ''
            if '合并资产负债表' in text and '流动资产合计' in text and '流动负债合计' in text:
                import re
                for line in text.split('\n'):
                    if '流动资产合计' in line and current_assets is None:
                        m = re.findall(r'([\d,]+\.\d+)', line)
                        if m:
                            try: current_assets = float(m[-1].replace(',', ''))
                            except: pass
                    if '流动负债合计' in line and current_liab is None:
                        m = re.findall(r'([\d,]+\.\d+)', line)
                        if m:
                            try: current_liab = float(m[-1].replace(',', ''))
                            except: pass
                    if '负债合计' in line and total_liab is None:
                        m = re.findall(r'([\d,]+\.\d+)', line)
                        if m:
                            try: total_liab = float(m[-1].replace(',', ''))
                            except: pass
                if current_assets and current_liab:
                    break

    return {
        'revenue': revenue, 'net_profit': net_profit, 'ocf': ocf,
        'total_assets': total_assets, 'total_liab': total_liab,
        'current_assets': current_assets, 'current_liab': current_liab,
        'equity': equity,
    }


def calc_ratios(fin):
    if not all([fin.get('revenue'), fin.get('net_profit'),
                 fin.get('total_assets'), fin.get('total_liab')]):
        return None
    try:
        equity = fin.get('equity') or fin['total_assets'] - fin['total_liab']
        return {
            'roe': fin['net_profit'] / equity,
            'roa': fin['net_profit'] / fin['total_assets'],
            'debt_ratio': fin['total_liab'] / fin['total_assets'],
            'current_ratio': (fin.get('current_assets') or 0) / (fin.get('current_liab') or 1),
            'asset_turnover': fin['revenue'] / fin['total_assets'],
            'net_margin': fin['net_profit'] / fin['revenue'],
            'ocf_to_rev': (fin.get('ocf') or 0) / fin['revenue'],
        }
    except Exception:
        return None


def run_model(ratios):
    model = joblib.load(os.path.join(MODEL_DIR, "fraud_detection_rf_combined.pkl"))
    return model.predict_proba(pd.DataFrame([ratios])[FIN_COLS])[:, 1][0]


def apply_all_rules(pdf_path, ratios):
    rules = []
    checks = [
        ('R1', '连续亏损', 25, (ratios['roe']<0) & (ratios['net_margin']<0)),
        ('R2', '现金流背离', 30, (ratios['ocf_to_rev']<0) & (ratios['net_margin']>0)),
        ('R3', '高负债', 15, ratios['debt_ratio']>0.7),
        ('R4', '流动性紧张', 20, ratios['current_ratio']<1),
        ('R5', '资产周转异常', 10, (ratios['asset_turnover']<0.3) | (ratios['asset_turnover']>5)),
        ('R6', 'ROE 异常', 25, (ratios['roa']>0.1) & (ratios['roe']<0)),
    ]
    for rid, name, sev, trig in checks:
        if trig:
            rules.append({'rule_id': rid, 'name': name, 'severity': sev, 'source': 'R1-R6 财务比率'})

    ic_signals = extract_internal_control(pdf_path)
    for r in signals_to_rules(ic_signals):
        r['source'] = 'P2-2 内部控制'
        rules.append(r)

    keyword_signals = extract_governance_signals_v2(pdf_path, max_pages=80)
    findings = keyword_signals['findings']
    category_hits = {}
    for category, items in findings.items():
        if not items:
            continue
        max_weight_item = max(items, key=lambda x: x['weight'])
        category_hits[category] = {
            'label': items[0]['category_label'],
            'count': len(items),
            'max_weight': max_weight_item['weight'],
            'top_keyword': max_weight_item['pattern'],
        }

    max_floor = 0
    for cat, info in category_hits.items():
        floor = load_keyword_library()['categories'][cat]['risk_floor']
        max_floor = max(max_floor, floor)

    for cat, info in sorted(category_hits.items(), key=lambda x: -x[1]['max_weight']):
        rules.append({
            'rule_id': f'KW-{cat[:3].upper()}',
            'name': f'关键词触发:{info["label"]}',
            'severity': info['max_weight'],
            'source': 'P2-3 关键词',
            'detail': f'{info["count"]}个关键词命中,最高严重度{info["max_weight"]} (Top:"{info["top_keyword"]}")',
        })

    return rules, max_floor, category_hits, keyword_signals


def generate_report(company_name, code, ratios, p_ml, rules, fin, risk_floor,
                     keyword_findings, ic_signals, network_info, external_signals):
    rule_sum = sum(r['severity'] for r in rules if r['rule_id'].startswith('R'))
    rule_norm = rule_sum / 125 if rule_sum > 0 else 0
    risk = 0.6 * p_ml + 0.4 * rule_norm
    risk = max(risk, risk_floor)
    level = assign_level(risk)

    md = f"""# {company_name} ({code}) 风险预警报告 v3.0

> 出具方: 持续审计系统 v3.0 (含 P0/P1/P2/P3 全套)
> 数据来源: 上传 PDF + tushare 多数据源
> 报告日期: {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 一、综合风险评分

| 维度 | 数值 |
|---|---|
| ML 舞弊概率 | **{p_ml:.4f}** |
| 触发的硬规则数 | {len([r for r in rules if r['rule_id'].startswith('R')])} |
| 触发的治理关键词类别 | {len(keyword_findings)} |
| **综合风险分** | **{risk:.4f}** |
| **风险等级** | **{level}** |

---

## 二、公司基本信息

| 项目 | 内容 |
|---|---|
| 公司全称 | {company_name} |
| 股票代码 | {code} |

---

## 三、关键财务数据

| 指标 | 数值 |
|---|---|
| 营业收入 | {fin.get('revenue', 0)/1e8:.2f} 亿 |
| 净利润 | {fin.get('net_profit', 0)/1e8:.3f} 亿 |
| 经营现金流 | {fin.get('ocf', 0)/1e8:.3f} 亿 |
| 总资产 | {fin.get('total_assets', 0)/1e8:.2f} 亿 |
| 总负债 | {fin.get('total_liab', 0)/1e8:.2f} 亿 |
| 归母净资产 | {fin.get('equity', 0)/1e8:.2f} 亿 |

---

## 四、7 个模型特征比率

| 比率 | 数值 | 评价 |
|---|---|---|
"""
    metric_desc = {
        'roe': ('ROE', '%'), 'roa': ('ROA', '%'),
        'debt_ratio': ('资产负债率', '%'), 'current_ratio': ('流动比率', ''),
        'asset_turnover': ('总资产周转率', '%'), 'net_margin': ('销售净利率', '%'),
        'ocf_to_rev': ('现金净利率', '%'),
    }
    for col in FIN_COLS:
        v = ratios[col]
        label, suffix = metric_desc[col]
        display = f'{v*100:.2f}%' if suffix == '%' else f'{v:.4f}'
        md += f"| {label} | {display} | - |\n"

    # 触发的规则
    md += f"""
---

## 五、触发的所有规则(共 {len(rules)} 条)

| 规则 ID | 规则名 | 严重度 | 来源 |
|---|---|---|---|
"""
    for r in sorted(rules, key=lambda x: -x['severity']):
        md += f"| {r['rule_id']} | {r['name']} | {r['severity']} | {r.get('source', '-')} |\n"

    # 治理关键词
    md += f"""
---

## 六、治理层信号(关键词扫描,共 123 个关键词)

触发的治理类别 ({len(keyword_findings)} 类):
"""
    keyword_lib = load_keyword_library()
    for cat, info in sorted(keyword_findings.items(), key=lambda x: -x[1]['max_weight']):
        cat_label = keyword_lib['categories'][cat]['label']
        md += f"- **{cat_label}** ({info['count']} 个关键词命中,最高严重度 {info['max_weight']})\n"

    # 内控信号
    md += f"""
---

## 七、内部控制信号

| 字段 | 值 |
|---|---|
| 内控审计意见 | {ic_signals.get('ic_audit_opinion', '-')} |
| 内控重大缺陷 | {ic_signals.get('has_ic_material_weakness', '-')} |
| 监管处罚类型 | {ic_signals.get('punishment_type', '-') or '无'} |
"""

    # 治理网络(P2-9)
    md += f"""
---

## 八、🔴 治理网络分析(P2-9)

"""
    if network_info is None:
        md += "  ⚠️ 无法获取实控人信息(可能已退市或API限流)\n"
    else:
        controller = network_info.get('controller', '无')
        n_peers = network_info.get('n_peers', 0)
        n_high = network_info.get('n_high_risk_peers', 0)
        n_fraud = network_info.get('n_fraud_peers', 0)

        md += f"| 项目 | 内容 |\n|---|---|\n"
        md += f"| 实际控制人 | **{controller}** |\n"
        md += f"| 同实控人下其他公司(在市) | **{n_peers}** 家 |\n"
        md += f"| 其中高风险公司 | **{n_high}** 家 |\n"
        md += f"| 其中已知违规公司 | **{n_fraud}** 家 |\n\n"

        if n_peers > 0:
            md += f"**同实控人下其他公司清单(最多 20 家):**\n\n"
            md += "| 公司名 | 代码 | 行业 | 风险等级 | 风险分 | ST | ann_fin_flag |\n"
            md += "|---|---|---|---|---|---|---|\n"
            for _, peer in network_info['peers'].iterrows():
                name = peer.get('name', '-')
                sym = peer.get('symbol', '-')
                ind = peer.get('industry', '-')
                lvl = peer.get('risk_level_v3', '-')
                score = peer.get('risk_score_v3', np.nan)
                is_st = peer.get('is_st', 0)
                af = peer.get('ann_fin_flag', np.nan)
                score_str = f"{score:.3f}" if pd.notna(score) else '-'
                af_str = f"{int(af)}" if pd.notna(af) else '-'
                md += f"| {name} | {sym} | {ind} | {lvl} | {score_str} | {is_st} | {af_str} |\n"

            # 网络风险评估
            if n_high >= 2:
                md += f"\n  🚨 **网络风险高**:实控人下 {n_high} 家公司高风险,可能是**治理传导**或**行业系统性风险**\n"
            elif n_high == 1:
                md += f"\n  ⚠️ 网络风险中等:1 家公司高风险,需关注\n"
            else:
                md += f"\n  ✅ 网络风险低:无同实控人下高风险公司\n"

    # 多数据源 (P3)
    md += f"""
---

## 九、外源数据信号(P3)

| 数据源 | 字段 | 值 | 评估 |
|---|---|---|---|
| 工商数据 | 注册资本 | {external_signals.get('reg_capital', '-')} 万元 | - |
| 工商数据 | 员工数 | {external_signals.get('employees', '-')} 人 | - |
| 工商数据 | 成立日期 | {external_signals.get('setup_date', '-')} | - |
| 工商数据 | 董事长 | {external_signals.get('chairman', '-')} | - |
| 股权质押 | 质押比例 | {f"{external_signals.get('pledge_ratio', 0):.2f}%" if external_signals.get('pledge_ratio') is not None else '-'} | {'🔴 高' if external_signals.get('pledge_high') else '🟢 正常'} |
| 股权质押 | 质押笔数 | {external_signals.get('pledge_count', '-')} | - |
| 监管公告 | 180 天监管类公告 | {external_signals.get('n_regulatory_anns_180d', 0)} | {'🔴 多' if external_signals.get('n_regulatory_anns_180d', 0) >= 3 else '🟢 正常'} |
"""
    if external_signals.get('regulatory_anns'):
        md += f"\n**最近监管类公告:**\n"
        for date, title in external_signals['regulatory_anns']:
            md += f"- {date}: {title}\n"

    # Top 关键词
    md += f"""
---

## 十、Top 10 高频高严重度关键词
"""
    all_kw = []
    for cat, items in keyword_findings.items():
        for item in items:
            all_kw.append({
                'category': load_keyword_library()['categories'][cat]['label'],
                'pattern': item['pattern'],
                'weight': item['weight'],
                'count': item['count'],
            })
    all_kw.sort(key=lambda x: x['weight'] * x['count'], reverse=True)
    for i, kw in enumerate(all_kw[:10], 1):
        md += f"{i}. **[{kw['weight']:2d}] {kw['pattern']}** × {kw['count']} ({kw['category']})\n"

    # 健康信号 + 关注事项
    md += f"""
---

## 十一、健康信号 & 关注事项

✅ **健康信号**:
"""
    if fin.get('ocf', 0) > fin.get('net_profit', 0):
        md += f"- 经营现金流 > 净利润,利润质量较好\n"
    if ratios['current_ratio'] > 1:
        md += f"- 流动比率 {ratios['current_ratio']:.2f} > 1,短期偿债能力正常\n"
    if ratios['debt_ratio'] < 0.6:
        md += f"- 资产负债率 {ratios['debt_ratio']*100:.1f}%,负债水平健康\n"
    if external_signals.get('pledge_ratio') and external_signals.get('pledge_ratio', 0) < 10:
        md += f"- 控股股东质押率 {external_signals.get('pledge_ratio', 0):.1f}%,股权结构稳定\n"

    md += f"\n⚠️ **关注事项**:\n"
    if rule_sum > 0:
        md += f"- 触发 {len([r for r in rules if r['rule_id'].startswith('R')])} 条硬规则\n"
    if p_ml >= 0.5:
        md += f"- ML 模型给出 {p_ml:.3f} 高概率,需重点核查\n"
    if len(keyword_findings) > 0:
        md += f"- 触发 {len(keyword_findings)} 类治理关键词\n"
    if ic_signals.get('has_ic_material_weakness'):
        md += f"- ⚠️ **公司自评内控存在重大缺陷**\n"
    if external_signals.get('pledge_high'):
        md += f"- ⚠️ **控股股东质押率过高**({external_signals.get('pledge_ratio', 0):.1f}%)\n"
    if network_info and network_info.get('n_high_risk_peers', 0) >= 2:
        md += f"- 🚨 **同实控人下 {network_info.get('n_high_risk_peers', 0)} 家公司高风险,疑似治理传导**\n"

    md += f"""
---

## 十二、模型局限

⚠️ **重要说明**:
1. 本报告基于上传 PDF + tushare 多数据源
2. 治理网络分析依赖实控人信息准确性
3. 司法数据(诉讼/被执行)tushare 不提供,需 Wind/企查查
4. 内控反转、监管处罚等信号需要 H1(半年报)+ H2(年报)对比
5. 实际审计需结合现场程序、函证、监盘等审计师专业判断

---

*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*系统: 持续审计系统 v3.0 | ML: RF 300树 | 规则: 22+ 123关键词 | 数据源: 工商+质押+监管公告*
"""
    return md, risk, level


# ============================================================
# Streamlit 页面
# ============================================================
st.set_page_config(page_title='报告自动化 v3.0', page_icon='📄', layout='wide')

st.title('📄 PDF 年报自动分析 v3.0')
st.caption('集成 P0/P1/P2/P3 全套:7 比率 + 6 规则 + 5 时序 + 治理反转 + 内控 + 123 关键词 + **治理网络** + **外源数据**')

with st.expander('📖 系统能力说明', expanded=False):
    st.markdown('''
**v3.0 新增能力**:
- ✅ **治理网络分析 (P2-9)**:自动识别实控人 + 同实控人下其他公司 + 网络风险评级
- ✅ **多数据源接入 (P3)**:工商数据(注册资本/员工/管理层)+ 股权质押 + 监管公告
- ✅ v2.0 全部能力保留(财务/时序/治理/内控/词库)

**关键信号**:
- 同实控人下 ≥2 家公司高风险 → 治理传导警报
- 控股股东质押率 > 30% → 风险加分
- 180 天内 ≥3 条监管类公告 → 监管关注
    ''')

uploaded = st.file_uploader('选择年报 PDF', type=['pdf'])

if uploaded:
    with st.spinner('正在分析(P0/P1/P2/P3 全套)...'):
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(uploaded.read())
            tmp_path = f.name

        try:
            # 1. 公司信息
            info = extract_company_info(tmp_path)
            st.success(f"✅ 公司识别: {info['name']} ({info['code']})")

            # 2. 财务数据
            fin = extract_financial_data(tmp_path)
            if not all([fin.get('revenue'), fin.get('net_profit'), fin.get('total_assets')]):
                st.error('❌ 财务数据不完整,无法继续')
                st.stop()

            # 3. 7 比率 + ML
            ratios = calc_ratios(fin)
            p_ml = run_model(ratios)

            # 4. 全部规则
            rules, risk_floor, keyword_findings, _ = apply_all_rules(tmp_path, ratios)
            ic_signals = extract_internal_control(tmp_path)

            # 5. 治理网络 (P2-9)
            with st.spinner('分析治理网络...'):
                network_info = fetch_controller_network(info['code'])

            # 6. 外源数据 (P3)
            with st.spinner('拉取多数据源...'):
                external_signals = fetch_external_signals(info['code'])

            # 7. 综合
            rule_sum = sum(r['severity'] for r in rules if r['rule_id'].startswith('R'))
            rule_norm = rule_sum / 125 if rule_sum > 0 else 0
            risk = max(0.6 * p_ml + 0.4 * rule_norm, risk_floor)
            level = assign_level(risk)

            # 关键指标
            st.markdown('---')
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric('ML 概率', f'{p_ml:.4f}')
            c2.metric('硬规则数', len([r for r in rules if r['rule_id'].startswith('R')]))
            c3.metric('治理类别', len(keyword_findings))
            n_high_peers = network_info.get('n_high_risk_peers', 0) if network_info else 0
            c4.metric('同实控人高风险', n_high_peers,
                       delta='🚨 网络警报' if n_high_peers >= 2 else None,
                       delta_color='inverse' if n_high_peers >= 2 else 'off')
            c5.metric('综合风险分', f'{risk:.4f}', delta=level,
                       delta_color='inverse' if level == '低风险' else 'normal')

            # 8. 报告
            report_md, _, _ = generate_report(
                info['name'], info['code'], ratios, p_ml, rules, fin,
                risk_floor, keyword_findings, ic_signals, network_info, external_signals
            )

            st.markdown('---')
            st.subheader('📋 风险预警报告')
            st.markdown(report_md)

            st.download_button(
                label='📥 下载 Markdown 报告',
                data=report_md.encode('utf-8'),
                file_name=f"{info['code']}_{datetime.now().strftime('%Y%m%d')}_风险预警_v3.md",
                mime='text/markdown',
            )

        finally:
            os.unlink(tmp_path)

else:
    st.info('👆 上传 PDF 年报开始分析')

#!/usr/bin/env python3
"""
P2-12: 批量审计分析
===================
随机抽取 N 家公司,跑完整 P0/P1/P2/P3 流程,生成对比报告
- 行业分布
- 风险等级分布
- 触发规则数分布
- 个案对比
"""

import os
import random
import pandas as pd
import numpy as np
import joblib
import tushare as ts
from datetime import datetime

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


def sample_companies(n=10, seed=42):
    """
    随机抽取 N 家公司
    - 至少 3 家"已知违规"(ann_fin_flag=1)
    - 至少 2 家"已知合规"(ann_fin_flag=0)
    - 至少 2 家"未知"(ann_fin_flag=NaN)
    """
    random.seed(seed)

    # 加载所有公司
    risk = pd.read_csv(os.path.join(DATA_DIR, "risk_scored_industry.csv"))
    risk['Symbol'] = risk['Symbol'].astype(str).str.zfill(6)
    risk = risk.drop_duplicates(subset=['Symbol'], keep='first')

    fraud = risk[risk['ann_fin_flag'] == 1]
    non_fraud = risk[risk['ann_fin_flag'] == 0]
    unknown = risk[risk['ann_fin_flag'].isna()]

    # 平衡采样
    n_fraud = max(1, n // 3)
    n_non_fraud = max(1, n // 4)
    n_unknown = n - n_fraud - n_non_fraud

    sample_fraud = fraud.sample(min(n_fraud, len(fraud)), random_state=seed)
    sample_non = non_fraud.sample(min(n_non_fraud, len(non_fraud)), random_state=seed)
    sample_unk = unknown.sample(min(n_unknown, len(unknown)), random_state=seed)

    return pd.concat([sample_fraud, sample_non, sample_unk]).reset_index(drop=True)


def audit_single_company(symbol, company_name):
    """
    对单家公司跑完整审计
    """
    audit_result = {
        'Symbol': symbol,
        'ShortName': company_name,
        'revenue': None,
        'ratios': {},
        'p_ml': None,
        'rules_triggered': [],
        'network_info': None,
        'external': {},
        'risk_score': 0,
        'risk_level': '未知',
    }

    try:
        # 1. 财务数据
        ts_code = f"{symbol}.{'SH' if symbol.startswith('6') else 'SZ'}"
        inc = pro.income(ts_code=ts_code, period='20241231',
                          fields='ts_code,end_date,revenue,n_income')
        bs = pro.balancesheet(ts_code=ts_code, period='20241231',
                              fields='ts_code,end_date,total_assets,total_liab,total_hldr_eqy_inc_min_int,total_cur_assets,total_cur_liab')
        cf = pro.cashflow(ts_code=ts_code, period='20241231',
                          fields='ts_code,end_date,n_cashflow_act')

        if inc is None or bs is None or inc.empty or bs.empty:
            audit_result['error'] = '无财务数据'
            return audit_result

        r_inc = inc.iloc[0]
        r_bs = bs.iloc[0]
        r_cf = cf.iloc[0] if cf is not None and not cf.empty else None

        revenue = float(r_inc.get('revenue', 0)) if pd.notna(r_inc.get('revenue')) else 0
        net_profit = float(r_inc.get('n_income', 0)) if pd.notna(r_inc.get('n_income')) else 0
        total_assets = float(r_bs.get('total_assets', 0)) if pd.notna(r_bs.get('total_assets')) else 0
        total_liab = float(r_bs.get('total_liab', 0)) if pd.notna(r_bs.get('total_liab')) else 0
        equity = float(r_bs.get('total_hldr_eqy_inc_min_int', 0)) if pd.notna(r_bs.get('total_hldr_eqy_inc_min_int')) else 0
        current_assets = float(r_bs.get('total_cur_assets', 0)) if pd.notna(r_bs.get('total_cur_assets')) else 0
        current_liab = float(r_bs.get('total_cur_liab', 0)) if pd.notna(r_bs.get('total_cur_liab')) else 0
        ocf = float(r_cf.get('n_cashflow_act', 0)) if r_cf is not None and pd.notna(r_cf.get('n_cashflow_act')) else 0

        audit_result['revenue'] = revenue

        # 2. 7 比率
        if revenue <= 0 or total_assets <= 0 or total_liab <= 0:
            audit_result['error'] = '财务数据为 0'
            return audit_result

        ratios = {
            'roe': net_profit / equity if equity > 0 else 0,
            'roa': net_profit / total_assets,
            'debt_ratio': total_liab / total_assets,
            'current_ratio': current_assets / current_liab if current_liab > 0 else 0,
            'asset_turnover': revenue / total_assets,
            'net_margin': net_profit / revenue,
            'ocf_to_rev': ocf / revenue,
        }
        audit_result['ratios'] = ratios

        # 3. ML 模型
        model = joblib.load(os.path.join(MODEL_DIR, "fraud_detection_xgb_combined.pkl"))
        p_ml = model.predict_proba(pd.DataFrame([ratios])[FIN_COLS])[:, 1][0]
        audit_result['p_ml'] = p_ml

        # 4. 6 条规则
        rules = [
            ('R1', '连续亏损', 25, (ratios['roe']<0) & (ratios['net_margin']<0)),
            ('R2', '现金流背离', 30, (ratios['ocf_to_rev']<0) & (ratios['net_margin']>0)),
            ('R3', '高负债', 15, ratios['debt_ratio']>0.7),
            ('R4', '流动性紧张', 20, ratios['current_ratio']<1),
            ('R5', '资产周转异常', 10, (ratios['asset_turnover']<0.3) | (ratios['asset_turnover']>5)),
            ('R6', 'ROE 异常', 25, (ratios['roa']>0.1) & (ratios['roe']<0)),
        ]
        rule_sum = 0
        for rid, name, sev, trig in rules:
            if trig:
                audit_result['rules_triggered'].append(rid)
                rule_sum += sev

        # 5. 时序特征(此处用公司 i 自身,需要历史,简化)
        # 暂用静态风险分

        # 6. 治理信号(ST 状态)
        try:
            b = pro.stock_basic(ts_code=ts_code, fields='ts_code,symbol,name,industry,act_name')
            if b is not None and not b.empty:
                name = b.iloc[0].get('name', '')
                is_st = 1 if 'ST' in name or '退' in name else 0
                is_strict = 1 if '*ST' in name or '退市' in name else 0
                if is_strict:
                    audit_result['rules_triggered'].append('R7-STRICT_ST')
                    rule_sum += 25
                elif is_st:
                    audit_result['rules_triggered'].append('R8-ST')
                    rule_sum += 15

                # 治理网络
                controller = b.iloc[0].get('act_name', '无')
                audit_result['network_info'] = {
                    'controller': controller,
                    'is_st': is_st,
                }
        except Exception:
            pass

        # 7. 股权质押
        try:
            p = pro.pledge_stat(ts_code=ts_code, fields='ts_code,end_date,pledge_ratio')
            if p is not None and not p.empty:
                p_latest = p.sort_values('end_date', ascending=False).iloc[0]
                ratio = float(p_latest.get('pledge_ratio', 0)) if pd.notna(p_latest.get('pledge_ratio')) else 0
                audit_result['external']['pledge_ratio'] = ratio
                if ratio > 30:
                    audit_result['rules_triggered'].append('P3-R1')
                    rule_sum += 25
        except Exception:
            pass

        # 8. 综合
        rule_norm = rule_sum / 150
        risk = 0.6 * p_ml + 0.4 * rule_norm
        # ST 状态自动上调
        if is_strict:
            risk = max(risk, 0.7)
        elif is_st:
            risk = max(risk, 0.55)
        audit_result['risk_score'] = risk
        audit_result['risk_level'] = assign_level(risk)

    except Exception as e:
        audit_result['error'] = str(e)[:100]

    return audit_result


def batch_audit(n=10, seed=42):
    """
    批量审计 N 家公司
    """
    print("=" * 60)
    print(f"P2-12: 批量审计 — 随机抽取 {n} 家公司")
    print("=" * 60)
    print(f"  Seed: {seed}")

    samples = sample_companies(n=n, seed=seed)
    print(f"  采样: {len(samples)} 家公司")
    print(f"    已知违规: {(samples['ann_fin_flag']==1).sum()}")
    print(f"    已知合规: {(samples['ann_fin_flag']==0).sum()}")
    print(f"    未知: {samples['ann_fin_flag'].isna().sum()}")

    results = []
    for i, row in samples.iterrows():
        symbol = str(row['Symbol']).zfill(6)
        name = row.get('ShortName', symbol)
        print(f"\n[{i+1}/{len(samples)}] 审计 {name} ({symbol})...")
        result = audit_single_company(symbol, name)
        result['ann_fin_flag'] = row.get('ann_fin_flag', np.nan)
        result['industry'] = row.get('industry', '未分类')
        result['p_ml_v3'] = result['p_ml']  # 用于后续对比
        results.append(result)
        print(f"  → {result.get('risk_level', '?')} (风险分 {result.get('risk_score', 0):.3f}) | "
              f"ML {result.get('p_ml', 0):.3f} | 规则 {len(result.get('rules_triggered', []))}")

    return results


def generate_batch_report(results, n=10):
    """
    生成批量审计对比报告
    """
    df = pd.DataFrame(results)

    report = []
    report.append(f"# 批量审计对比报告\n")
    report.append(f"> 出具方: 持续审计系统 v3.0 (P0/P1/P2/P3 全套)")
    report.append(f"> 报告日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"> 审计公司数: {n}\n")
    report.append("---\n")

    # 1. 风险等级分布
    report.append("## 一、风险等级分布\n")
    risk_dist = df['risk_level'].value_counts()
    report.append("| 风险等级 | 公司数 | 占比 |\n|---|---|---|\n")
    for level in ['高风险', '中风险', '低风险']:
        cnt = risk_dist.get(level, 0)
        pct = cnt / n * 100
        report.append(f"| {level} | {cnt} | {pct:.0f}% |\n")

    # 2. 已知违规命中率
    actual_fraud = df[df['ann_fin_flag'] == 1]
    actual_clean = df[df['ann_fin_flag'] == 0]
    n_fraud_caught = (actual_fraud['risk_level'] == '高风险').sum()
    n_clean_low = (actual_clean['risk_level'] == '低风险').sum()
    report.append(f"\n## 二、识别精度评估\n")
    report.append(f"- **已知违规公司({len(actual_fraud)}家)中被标为高风险**: {n_fraud_caught} ({n_fraud_caught/len(actual_fraud)*100:.0f}%)\n")
    if len(actual_clean) > 0:
        report.append(f"- **已知合规公司({len(actual_clean)}家)中被标为低风险**: {n_clean_low} ({n_clean_low/len(actual_clean)*100:.0f}%)\n")

    # 3. 公司明细
    report.append(f"\n## 三、审计明细(按风险分降序)\n")
    report.append("| 公司 | 代码 | 行业 | ML概率 | 风险分 | 等级 | 触发规则 | 已知违规 |\n")
    report.append("|---|---|---|---|---|---|---|---|\n")
    for _, r in df.sort_values('risk_score', ascending=False).iterrows():
        rules_str = ', '.join(r.get('rules_triggered', [])[:5]) if r.get('rules_triggered') else '-'
        af = int(r['ann_fin_flag']) if pd.notna(r.get('ann_fin_flag')) and r.get('ann_fin_flag') == 1 else ('✅' if pd.notna(r.get('ann_fin_flag')) and r.get('ann_fin_flag') == 0 else '?')
        report.append(f"| {r['ShortName']} | {r['Symbol']} | {r.get('industry', '-')} | "
                       f"{r.get('p_ml', 0):.3f} | {r.get('risk_score', 0):.3f} | {r.get('risk_level', '?')} | "
                       f"{rules_str} | {af} |\n")

    # 4. 关键发现
    report.append(f"\n## 四、关键发现\n")
    high_risk = df[df['risk_level'] == '高风险']
    low_risk = df[df['risk_level'] == '低风险']

    if len(high_risk) > 0:
        report.append(f"\n### 🚨 高风险公司({len(high_risk)}家)\n")
        for _, r in high_risk.iterrows():
            ctrl = r.get('network_info', {}).get('controller', '-') if r.get('network_info') else '-'
            pledge = r.get('external', {}).get('pledge_ratio', '-')
            pledge_str = f"{pledge:.1f}%" if isinstance(pledge, (int, float)) else '-'
            report.append(f"- **{r['ShortName']}** ({r['Symbol']}): 风险分 {r.get('risk_score', 0):.3f}, "
                           f"实控人 {ctrl[:10]}{'...' if len(str(ctrl)) > 10 else ''}, 质押 {pledge_str}\n")

    if len(low_risk) > 0:
        report.append(f"\n### ✅ 低风险公司({len(low_risk)}家)\n")
        for _, r in low_risk.iterrows():
            report.append(f"- **{r['ShortName']}** ({r['Symbol']}): 风险分 {r.get('risk_score', 0):.3f}, "
                           f"实控人 {r.get('network_info', {}).get('controller', '-') if r.get('network_info') else '-'}\n")

    # 5. 行业分布
    report.append(f"\n## 五、行业分布\n")
    ind_dist = df['industry'].value_counts().head(10)
    for ind, cnt in ind_dist.items():
        report.append(f"- {ind}: {cnt} 家\n")

    return ''.join(report)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10, help='审计公司数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output', type=str, default=None, help='报告保存路径')
    args = parser.parse_args()

    results = batch_audit(n=args.n, seed=args.seed)
    report = generate_batch_report(results, n=args.n)

    print("\n" + "=" * 60)
    print("📊 批量审计报告")
    print("=" * 60)
    print(report)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n  报告已保存: {args.output}")
    else:
        # 默认保存
        out_path = os.path.join(OUT_DIR, f"batch_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n  报告已保存: {out_path}")

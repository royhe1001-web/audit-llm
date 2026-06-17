#!/usr/bin/env python3
"""
P2-10: 持续监控系统
===================
每周自动运行全市场扫描:
1. 从 tushare 拉取所有上市公司最新财务数据
2. 跑 RF 模型 + 风险评分 + 治理信号检测
3. 检测新出现的"高风险升级"
4. 通知:邮件(可选) / 控制台 / 文件

使用方式:
    # 单次运行
    python continuous_monitor.py --once

    # 每周定时(用 cron 0 9 * * 1 每周一 9 点)
    python continuous_monitor.py --schedule

    # 模拟回测(扫描历史某一天)
    python continuous_monitor.py --simulate 2025-12-31
"""

import os
import sys
import json
import argparse
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import tushare as ts
import joblib

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")
MODEL_DIR = os.path.join(BASE, "models")

TOKEN = "f3bf8d32b09bb2cfa3f3632b5521caa8143576c3dff550a742f5f4cc"
ts.set_token(TOKEN)
pro = ts.pro_api()

FIN_COLS = ['roe', 'roa', 'debt_ratio', 'current_ratio', 'asset_turnover',
            'net_margin', 'ocf_to_rev']


# ============================================================
# 1. 全市场扫描
# ============================================================
def scan_market(date=None, n_stocks=None):
    """
    扫描全市场(在市 + 退市 + 暂停)
    """
    print(f"[{datetime.now()}] 开始扫描: {date or '当前最新'}")

    # 1.1 拉取所有公司基础信息
    all_companies = []
    for status in ['L', 'D', 'P']:
        try:
            b = pro.stock_basic(list_status=status, fields='ts_code,symbol,name,industry,area,list_date,act_name')
            all_companies.append(b)
        except Exception as e:
            print(f"  ⚠️ {status}: {e}")
    all_companies = pd.concat(all_companies, ignore_index=True).drop_duplicates(subset='symbol')
    all_companies['Symbol'] = all_companies['symbol'].astype(str).str.zfill(6)
    print(f"  公司总数: {len(all_companies)}")

    if n_stocks:
        all_companies = all_companies.head(n_stocks)

    # 1.2 拉取最新年度财务数据
    period = date.replace('-', '') + '31' if date else '20241231'
    print(f"  拉取 {period} 财务数据...")

    financials = []
    symbols = all_companies['symbol'].tolist()
    for i, sym in enumerate(symbols):
        if i % 200 == 0:
            print(f"    进度: {i}/{len(symbols)}")
        try:
            # 利润表
            inc = pro.income(ts_code=f"{sym}.{'SH' if sym.startswith('6') else 'SZ'}",
                             period=period, fields='ts_code,end_date,revenue,n_income')
            # 资产负债表
            bs = pro.balancesheet(ts_code=f"{sym}.{'SH' if sym.startswith('6') else 'SZ'}",
                                  period=period, fields='ts_code,end_date,total_assets,total_liab,total_hldr_eqy_inc_min_int,total_cur_assets,total_cur_liab')
            # 现金流量表
            cf = pro.cashflow(ts_code=f"{sym}.{'SH' if sym.startswith('6') else 'SZ'}",
                              period=period, fields='ts_code,end_date,n_cashflow_act')

            if inc is None or inc.empty or bs is None or bs.empty:
                continue

            r_inc = inc.iloc[0]
            r_bs = bs.iloc[0]
            r_cf = cf.iloc[0] if cf is not None and not cf.empty else None

            fin = {
                'Symbol': str(sym).zfill(6),
                'revenue': r_inc.get('revenue'),
                'net_profit': r_inc.get('n_income'),
                'total_assets': r_bs.get('total_assets'),
                'total_liab': r_bs.get('total_liab'),
                'equity': r_bs.get('total_hldr_eqy_inc_min_int'),
                'current_assets': r_bs.get('total_cur_assets'),
                'current_liab': r_bs.get('total_cur_liab'),
                'ocf': r_cf.get('n_cashflow_act') if r_cf is not None else None,
            }
            financials.append(fin)
        except Exception:
            pass

    df_fin = pd.DataFrame(financials)
    print(f"  拉到有效财务: {len(df_fin)} 家公司")

    # 1.3 合并公司名
    df_fin = df_fin.merge(all_companies[['Symbol', 'name', 'industry', 'act_name']],
                            on='Symbol', how='left')
    return df_fin, all_companies


# ============================================================
# 2. 计算 7 比率 + 风险评分
# ============================================================
def calc_risks(df_fin):
    """
    对每家公司计算 7 比率 + 跑模型 + 6 规则
    """
    # 7 比率
    df = df_fin.copy()
    df['roe'] = df['net_profit'] / df['equity']
    df['roa'] = df['net_profit'] / df['total_assets']
    df['debt_ratio'] = df['total_liab'] / df['total_assets']
    df['current_ratio'] = df['current_assets'] / df['current_liab']
    df['asset_turnover'] = df['revenue'] / df['total_assets']
    df['net_margin'] = df['net_profit'] / df['revenue']
    df['ocf_to_rev'] = df['ocf'] / df['revenue']

    # 加载模型
    model = joblib.load(os.path.join(MODEL_DIR, "fraud_detection_xgb_combined.pkl"))
    df['p_ml'] = model.predict_proba(df[FIN_COLS])[:, 1]

    # 6 规则
    df['R1'] = ((df['roe']<0) & (df['net_margin']<0)).astype(int)
    df['R2'] = ((df['ocf_to_rev']<0) & (df['net_margin']>0)).astype(int)
    df['R3'] = (df['debt_ratio']>0.7).astype(int)
    df['R4'] = (df['current_ratio']<1).astype(int)
    df['R5'] = ((df['asset_turnover']<0.3) | (df['asset_turnover']>5)).astype(int)
    df['R6'] = ((df['roa']>0.1) & (df['roe']<0)).astype(int)
    df['n_rules'] = df[['R1','R2','R3','R4','R5','R6']].sum(axis=1)

    # 风险分
    rule_score = df['R1']*25 + df['R2']*30 + df['R3']*15 + df['R4']*20 + df['R5']*10 + df['R6']*25
    rule_norm = rule_score / 125
    df['risk_score'] = 0.6 * df['p_ml'] + 0.4 * rule_norm

    # ST 状态
    df['is_st'] = df['name'].str.contains('ST|退|PT', na=False).astype(int)
    df['is_strict_st'] = df['name'].str.contains(r'\*ST|退|PT', na=False, regex=True).astype(int)

    # 风险等级
    def level(s, is_st, is_strict):
        if is_strict: return '高风险'
        if is_st: return '高风险'
        if s >= 0.55: return '高风险'
        if s >= 0.25: return '中风险'
        return '低风险'
    df['risk_level'] = df.apply(lambda r: level(r['risk_score'], r['is_st'], r['is_strict_st']), axis=1)

    return df


# ============================================================
# 3. 风险升级检测
# ============================================================
def detect_escalations(new_scan, prev_scan_path):
    """
    对比本次扫描与上次扫描,找出"风险升级"的公司
    """
    if not os.path.exists(prev_scan_path):
        print(f"  无上次扫描数据,跳过升级检测")
        return pd.DataFrame()

    prev = pd.read_csv(prev_scan_path)
    prev['Symbol'] = prev['Symbol'].astype(str).str.zfill(6)

    new_scan = new_scan.copy()
    new_scan['Symbol'] = new_scan['Symbol'].astype(str).str.zfill(6)

    # 合并
    merged = new_scan[['Symbol', 'name', 'risk_level', 'risk_score']].merge(
        prev[['Symbol', 'risk_level', 'risk_score']].rename(columns={
            'risk_level': 'prev_level', 'risk_score': 'prev_score'
        }),
        on='Symbol', how='inner'
    )

    # 升级:中 → 高, 或 低 → 中/高
    level_order = {'低风险': 0, '中风险': 1, '高风险': 2}
    merged['cur_lvl'] = merged['risk_level'].map(level_order)
    merged['prev_lvl'] = merged['prev_level'].map(level_order)
    merged['upgrade'] = merged['cur_lvl'] - merged['prev_lvl']

    escalations = merged[merged['upgrade'] > 0].sort_values('upgrade', ascending=False)
    return escalations


# ============================================================
# 4. 通知
# ============================================================
def notify_console(alerts):
    """
    控制台通知
    """
    print(f"\n{'='*60}")
    print(f"🚨 监控警报 — {len(alerts)} 条高风险")
    print(f"{'='*60}")
    if len(alerts) == 0:
        print("  无新增高风险公司")
        return
    print(alerts[['name', 'Symbol', 'risk_level', 'risk_score', 'p_ml']].head(20).to_string(index=False))


def notify_file(alerts, escalations, out_path):
    """
    文件通知(总是生成最新报告)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(out_path, f'monitor_report_{timestamp}.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 持续监控报告 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 扫描结果\n\n")
        f.write(f"- 总扫描: {len(alerts)} 家公司\n")
        f.write(f"- 高风险: {(alerts['risk_level']=='高风险').sum()}\n")
        f.write(f"- 中风险: {(alerts['risk_level']=='中风险').sum()}\n")
        f.write(f"- 低风险: {(alerts['risk_level']=='低风险').sum()}\n\n")

        f.write(f"## 高风险公司(按风险分排序)\n\n")
        cols = ['name', 'Symbol', 'industry', 'risk_level', 'risk_score', 'p_ml']
        high_risk = alerts[alerts['risk_level']=='高风险'].sort_values('risk_score', ascending=False)
        f.write(high_risk[cols].head(30).to_string(index=False))
        f.write("\n\n")

        if len(escalations) > 0:
            f.write(f"## 风险升级公司({len(escalations)} 条)\n\n")
            f.write(escalations[['name', 'Symbol', 'prev_level', 'risk_level', 'prev_score', 'risk_score']].head(20).to_string(index=False))
            f.write("\n\n")

    print(f"  报告已保存: {report_path}")
    return report_path


def notify_email(alerts, smtp_config=None):
    """
    邮件通知(需要配置 SMTP)
    """
    if not smtp_config:
        print("  邮件未配置,跳过")
        return

    high = alerts[alerts['risk_level']=='高风险']
    if len(high) == 0:
        return

    msg = MIMEText(f"持续监控发现 {len(high)} 家公司高风险\n\n{high[['name', 'Symbol', 'risk_score']].head(20).to_string()}")
    msg['Subject'] = f"🚨 持续监控警报 — {len(high)} 家高风险公司"
    msg['From'] = smtp_config['from']
    msg['To'] = smtp_config['to']

    try:
        with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as s:
            s.starttls()
            s.login(smtp_config['user'], smtp_config['password'])
            s.send_message(msg)
        print(f"  邮件已发送: {smtp_config['to']}")
    except Exception as e:
        print(f"  邮件失败: {e}")


# ============================================================
# 5. 主函数
# ============================================================
def run_monitor(date=None, n_stocks=None, mode='once', smtp_config=None):
    print("=" * 60)
    print(f"持续监控系统 v1.0 — 模式: {mode}")
    print("=" * 60)

    # 1. 扫描
    df_fin, all_companies = scan_market(date=date, n_stocks=n_stocks)

    # 2. 风险评分
    df = calc_risks(df_fin)

    # 3. 检测升级
    prev_path = os.path.join(DATA_DIR, 'monitor_prev_scan.csv')
    escalations = detect_escalations(df, prev_path)

    # 4. 通知
    notify_console(df)

    # 保存本次扫描为"上次扫描"
    save_cols = ['Symbol', 'name', 'industry', 'risk_level', 'risk_score', 'p_ml', 'is_st', 'is_strict_st']
    df[save_cols].to_csv(prev_path, index=False)
    print(f"\n  本次扫描已保存: {prev_path}")

    # 文件通知
    report_path = notify_file(df, escalations, OUT_DIR)

    # 邮件通知(可选)
    notify_email(df, smtp_config)

    return df, escalations, report_path


# ============================================================
# 6. CLI 入口
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='持续监控 — 上市公司财务舞弊风险')
    parser.add_argument('--once', action='store_true', help='单次运行')
    parser.add_argument('--schedule', action='store_true', help='定时模式(每周)')
    parser.add_argument('--simulate', type=str, help='模拟回测:指定日期 YYYY-MM-DD')
    parser.add_argument('--n', type=int, default=None, help='限制公司数(测试用)')

    args = parser.parse_args()

    if args.simulate:
        df, esc, report = run_monitor(date=args.simulate, n_stocks=args.n)
    elif args.schedule:
        # 用 schedule 库(可选)
        try:
            import schedule
            import time

            def job():
                df, esc, report = run_monitor()

            schedule.every().monday.at("09:00").do(job)
            print("  定时任务:每周一 9:00 运行")
            while True:
                schedule.run_pending()
                time.sleep(60)
        except ImportError:
            print("  schedule 未安装,用 cron 替代:0 9 * * 1 python continuous_monitor.py --once")
            run_monitor(n_stocks=args.n)
    else:
        df, esc, report = run_monitor(n_stocks=args.n)

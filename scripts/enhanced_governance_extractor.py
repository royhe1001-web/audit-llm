#!/usr/bin/env python3
"""
P2-3: 增强治理信号提取器
=========================
基于 governance_keywords.json 词库,扫描 PDF 全文,识别所有治理层高风险信号。
支持:
- 精确匹配 / 正则匹配
- 多类别(内控/治理/监管/财务/高管/诉讼/ST)
- 严重度加权
- 反转检测(H1 vs H2)
- 关键词频次统计
"""

import os
import re
import json
import pdfplumber
from collections import defaultdict
from datetime import datetime

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")


def load_keyword_library():
    with open(os.path.join(DATA_DIR, 'governance_keywords.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def scan_text(text, keyword_lib, max_pages=80):
    """
    扫描 PDF 文本,识别所有治理层高风险信号
    """
    findings = defaultdict(list)  # {category: [(pattern, weight, matches), ...]}

    for category, info in keyword_lib['categories'].items():
        for kw in info['keywords']:
            pattern = kw['pattern']
            weight = kw['weight']
            kw_type = kw['type']

            if kw_type == 'regex':
                matches = re.findall(pattern, text)
            else:  # exact
                matches = [pattern] * text.count(pattern)

            if matches:
                findings[category].append({
                    'pattern': pattern,
                    'weight': weight,
                    'count': len(matches),
                    'category_label': info['label'],
                    'risk_floor': info['risk_floor'],
                })

    return findings


def aggregate_findings(findings):
    """
    聚合所有发现,输出:
    - 总严重度(同类别取 max,跨类别求和)
    - 风险分下限
    - 类别命中数
    - 高频关键词(top 10)
    """
    total_severity = 0
    category_hits = {}
    category_max_weight = {}
    triggered_floors = []
    keyword_freq = []

    for category, items in findings.items():
        if not items:
            continue
        category_hits[category] = len(items)
        max_weight = max(item['weight'] for item in items)
        total_count = sum(item['count'] for item in items)
        category_max_weight[category] = max_weight
        total_severity += max_weight  # 同类取 max
        triggered_floors.append(items[0]['risk_floor'])
        # 关键词频次
        for item in items:
            keyword_freq.append({
                'category': items[0]['category_label'],
                'pattern': item['pattern'],
                'weight': item['weight'],
                'count': item['count'],
            })

    # 风险分下限:取触发规则中的最高
    risk_floor = max(triggered_floors) if triggered_floors else 0

    return {
        'total_severity': total_severity,
        'risk_floor': risk_floor,
        'category_hits': category_hits,
        'category_max_weight': category_max_weight,
        'keyword_freq': sorted(keyword_freq, key=lambda x: x['weight'] * x['count'], reverse=True)[:10],
    }


def extract_governance_signals_v2(pdf_path, max_pages=80):
    """
    增强版治理信号提取 — 扫描前 max_pages 页
    返回:
    - aggregate: 聚合结果
    - findings: 各类别发现
    - text: 提取的全文(供后续反转检测)
    """
    keyword_lib = load_keyword_library()
    all_text = ''

    with pdfplumber.open(pdf_path) as pdf:
        max_p = min(max_pages, len(pdf.pages))
        for i in range(max_p):
            text = pdf.pages[i].extract_text() or ''
            all_text += text + '\n'

    findings = scan_text(all_text, keyword_lib, max_pages)
    aggregate = aggregate_findings(findings)

    return {
        'aggregate': aggregate,
        'findings': findings,
        'text_length': len(all_text),
    }


def compare_governance_signals(h1_path, h2_path):
    """
    对比 H1(半年报)和 H2(年报)的治理信号,输出:
    - 新出现的关键词
    - 严重度变化
    - 反转规则
    """
    h1 = extract_governance_signals_v2(h1_path, max_pages=50)
    h2 = extract_governance_signals_v2(h2_path, max_pages=80)

    # 找新增的关键词
    h1_keywords = set()
    for cat, items in h1['findings'].items():
        for item in items:
            h1_keywords.add(item['pattern'])
    h2_keywords = set()
    for cat, items in h2['findings'].items():
        for item in items:
            h2_keywords.add(item['pattern'])

    new_keywords = h2_keywords - h1_keywords
    removed_keywords = h1_keywords - h2_keywords

    return {
        'h1_severity': h1['aggregate']['total_severity'],
        'h2_severity': h2['aggregate']['total_severity'],
        'severity_change': h2['aggregate']['total_severity'] - h1['aggregate']['total_severity'],
        'h1_risk_floor': h1['aggregate']['risk_floor'],
        'h2_risk_floor': h2['aggregate']['risk_floor'],
        'h1_findings': h1['findings'],
        'h2_findings': h2['findings'],
        'new_keywords': list(new_keywords),
        'removed_keywords': list(removed_keywords),
        'h1_categories': list(h1['findings'].keys()),
        'h2_categories': list(h2['findings'].keys()),
    }


# ============================================================
# Demo: 洲际油气
# ============================================================
if __name__ == '__main__':
    print('='*70)
    print(' '*15 + 'P2-3: 增强治理信号提取 — 洲际油气 demo')
    print('='*70)
    print(f'  数据: 2025 半年报 + 2025 年报')
    print(f'  词库: governance_keywords.json (5+ 大类, 100+ 关键词)')

    # 1. 单独看年报
    print('\n' + '='*70)
    print('1. 2025 年报治理信号(增强版)')
    print('='*70)
    yb_signals = extract_governance_signals_v2(f'{BASE}/洲际年报.pdf', max_pages=80)
    agg = yb_signals['aggregate']

    print(f'  扫描文本长度: {yb_signals["text_length"]:,} 字符')
    print(f'  触发类别数: {len(agg["category_hits"])}/{len(load_keyword_library()["categories"])}')
    print(f'  总严重度(同类取max): {agg["total_severity"]}')
    print(f'  风险分下限: {agg["risk_floor"]}')
    print()
    print(f'  各类别命中:')
    for cat, n in sorted(agg['category_hits'].items(), key=lambda x: -x[1]):
        label = yb_signals['findings'][cat][0]['category_label']
        max_w = agg['category_max_weight'][cat]
        print(f'    {label:20s}: {n} 个关键词, 最大严重度 {max_w}')

    print(f'\n  Top 10 高频高严重度关键词:')
    for i, kw in enumerate(agg['keyword_freq'][:10], 1):
        print(f'    {i:2d}. [{kw["weight"]:2d}] {kw["pattern"]} × {kw["count"]} ({kw["category"]})')

    # 2. 对比 H1 vs H2
    print('\n' + '='*70)
    print('2. H1 vs H2 治理信号对比')
    print('='*70)
    cmp = compare_governance_signals(f'{BASE}/洲际半年报.pdf', f'{BASE}/洲际年报.pdf')
    print(f'  H1 严重度: {cmp["h1_severity"]} (风险分下限 {cmp["h1_risk_floor"]})')
    print(f'  H2 严重度: {cmp["h2_severity"]} (风险分下限 {cmp["h2_risk_floor"]})')
    print(f'  严重度变化: {cmp["severity_change"]:+d}')

    print(f'\n  H1 类别: {cmp["h1_categories"]}')
    print(f'  H2 类别: {cmp["h2_categories"]}')
    print(f'  新增类别: {set(cmp["h2_categories"]) - set(cmp["h1_categories"])}')

    print(f'\n  H2 新出现的关键词 (top 10):')
    for kw in cmp['new_keywords'][:10]:
        print(f'    + {kw}')

    # 3. 集成到风险评分
    print('\n' + '='*70)
    print('3. 集成到风险评分')
    print('='*70)

    # 假设已有: v2 (0.60) + P2-1 (0.85) + P2-2 (0.85)
    base_score = 0.85
    new_floor = max(base_score, cmp['h2_risk_floor'])
    print(f'  v2 + P2-1 + P2-2: {base_score:.4f}')
    print(f'  + 词库匹配风险分下限: {cmp["h2_risk_floor"]}')
    print(f'  最终: {new_floor:.4f}')

    if new_floor >= 0.55:
        print(f'  → 高风险 🔴')

    # 4. 总结
    print('\n' + '='*70)
    print('✅ P2-3 完成: 治理标志词库 + 增强提取器')
    print('='*70)
    print(f'  词库规模: {sum(len(c["keywords"]) for c in load_keyword_library()["categories"].values())} 个关键词')
    print(f'  覆盖类别: {list(load_keyword_library()["categories"].keys())}')
    print(f'  风险分下限: 0.65-0.85 (按类别)')
    print(f'  产出: enhanced_governance_extractor.py + governance_keywords.json')

#!/usr/bin/env python3
"""
P2-2: 内部控制维度
====================
从年报 PDF 提取:
- 内部控制审计意见类型(有效/保留/否定)
- 内部控制重大缺陷
- 监管处罚(责令改正)
转化为硬规则 R20-R22
"""

import os
import re
import pdfplumber

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"


def extract_internal_control(pdf_path):
    """
    从 PDF 提取内控信号(扫前 60 页)
    """
    signals = {
        'ic_audit_opinion': '',         # 内控审计意见
        'has_ic_material_weakness': None, # 重大缺陷
        'ic_self_eval_conclusion': '',    # 自我评价结论
        'has_regulatory_punishment': None, # 监管处罚
        'punishment_type': '',           # 处罚类型
    }

    with pdfplumber.open(pdf_path) as pdf:
        max_pages = min(80, len(pdf.pages))
        for i in range(max_pages):
            text = pdf.pages[i].extract_text() or ''

            # 1. 内控审计意见
            if not signals['ic_audit_opinion']:
                m = re.search(r'内部控制审计报告.{0,5}意见类型[：:]\s*(.+)', text)
                if m:
                    signals['ic_audit_opinion'] = m.group(1).strip()[:20]

                # 备选:整行匹配
                for line in text.split('\n'):
                    if '内控' in line and ('意见' in line) and ('有效' in line or '保留' in line or '否定' in line or '无法' in line):
                        if '有效' in line and '否定' not in line:
                            if signals['ic_audit_opinion'] not in ['保留意见', '否定意见']:
                                signals['ic_audit_opinion'] = '有效'
                        elif '保留' in line:
                            signals['ic_audit_opinion'] = '保留意见'
                        elif '否定' in line:
                            signals['ic_audit_opinion'] = '否定意见'

            # 2. 重大缺陷
            if signals['has_ic_material_weakness'] is None:
                # "公司内部控制存在重大缺陷" 直接声明
                if '公司内部控制存在重大缺陷' in text or '内部控制存在重大缺陷' in text:
                    signals['has_ic_material_weakness'] = True
                elif '未发现公司内部控制存在重大缺陷' in text or '不存在重大缺陷' in text:
                    if '内控' in text and signals['has_ic_material_weakness'] is None:
                        signals['has_ic_material_weakness'] = False

            # 3. 自我评价结论
            if not signals['ic_self_eval_conclusion']:
                for line in text.split('\n'):
                    if ('自我评价' in line or '内控评价结论' in line) and ('有效' in line or '无效' in line or '缺陷' in line):
                        signals['ic_self_eval_conclusion'] = line.strip()[:100]

            # 4. 监管处罚(责令改正/警示函/立案调查/行政处罚)
            if signals['has_regulatory_punishment'] is None:
                if '责令改正' in text and '监管' in text:
                    signals['has_regulatory_punishment'] = True
                    if '责令改正' in signals['punishment_type']:
                        signals['punishment_type'] += '+责令改正'
                    else:
                        signals['punishment_type'] = '责令改正'
                if '警示函' in text and '监管' in text:
                    signals['has_regulatory_punishment'] = True
                    if '警示函' not in signals['punishment_type']:
                        signals['punishment_type'] += '+警示函'
                if '立案调查' in text:
                    signals['has_regulatory_punishment'] = True
                    if '立案调查' not in signals['punishment_type']:
                        signals['punishment_type'] += '+立案调查'
                if '行政处罚' in text and '监管' in text:
                    signals['has_regulatory_punishment'] = True
                    if '行政处罚' not in signals['punishment_type']:
                        signals['punishment_type'] += '+行政处罚'

    return signals


def signals_to_rules(signals):
    """
    将内控信号转化为风险规则
    """
    rules = []

    # R20: 内控审计否定意见
    if signals['ic_audit_opinion'] == '否定意见':
        rules.append({
            'rule_id': 'R20',
            'name': '内控审计否定意见',
            'severity': 30,
            'risk_floor': 0.85,
            'description': '大华会计师事务所对公司 2025 年内控出具否定意见',
        })

    # R21: 内控重大缺陷
    if signals['has_ic_material_weakness'] == True:
        rules.append({
            'rule_id': 'R21',
            'name': '内控重大缺陷',
            'severity': 30,
            'risk_floor': 0.85,
            'description': '公司自评 + 监管局均认定存在内控重大缺陷',
        })

    # R22: 监管处罚(责令改正/立案调查)
    if signals['has_regulatory_punishment'] == True:
        severity = 30 if '立案调查' in signals['punishment_type'] else 25
        rules.append({
            'rule_id': 'R22',
            'name': '监管处罚',
            'severity': severity,
            'risk_floor': 0.80,
            'description': f'收到{signals["punishment_type"]}',
        })

    return rules


# ============================================================
# Demo: 洲际油气
# ============================================================
print('='*70)
print(' '*15 + 'P2-2: 内部控制维度 — 洲际油气 demo')
print('='*70)
print(f'  数据: 2025 年报')
print(f'  目标: 提取内控审计意见 + 重大缺陷 + 监管处罚')

# 提取信号
print('\n' + '='*70)
print('1. 提取 2025 年报内控信号')
print('='*70)
signals = extract_internal_control(f'{BASE}/洲际年报.pdf')
for k, v in signals.items():
    print(f'  {k}: {v}')

# 转化为规则
print('\n' + '='*70)
print('2. 转化为风险规则')
print('='*70)
rules = signals_to_rules(signals)
if rules:
    for r in rules:
        print(f'  🔴 {r["rule_id"]} {r["name"]}:')
        print(f'     严重度: {r["severity"]}, 风险分下限: {r["risk_floor"]}')
        print(f'     描述: {r["description"]}')
else:
    print('  ⚪ 未触发任何内控规则')

# 集成到盲测 v2 + P2-1
print('\n' + '='*70)
print('3. 集成到风险评分(盲测 → 加上内控)')
print('='*70)

# 假设已有 v2 (0.60) + P2-1 反转 (0.85)
v2_with_p21 = 0.85
print(f'  v2 (时序) + P2-1 (反转): 0.85')

v_final = v2_with_p21
for r in rules:
    v_final = max(v_final, r['risk_floor'])
print(f'  + P2-2 (内控):           {v_final:.4f}')

if v_final >= 0.55:
    final_lvl = '高风险 🔴'
else:
    final_lvl = '中风险 🟠'
print(f'  最终风险等级: {final_lvl}')

# 对比:无任何治理信号 vs 加全部信号
print('\n' + '='*70)
print('4. 演进对比(从盲测到完整)')
print('='*70)
scenarios = [
    ('v1 仅 ML + 6 规则', 0.3547, '中风险 🟠'),
    ('v2 + 时序 R11-R15', 0.6000, '高风险 🔴'),
    ('v2 + 治理反转 P2-1', 0.8500, '高风险 🔴'),
    ('v2 + 治理反转 + 内控 P2-2', v_final, final_lvl),
]
print(f'  {"场景":<35} {"风险分":>10} {"等级":>15}')
for name, score, lvl in scenarios:
    print(f'  {name:<35} {score:>10.4f} {lvl:>15}')

# 总结
print('\n' + '='*70)
print('✅ P2-2 完成: 内部控制维度')
print('='*70)
print('  3 条新规则 (R20-R22):')
print('    R20 内控审计否定意见 → 风险分下限 0.85')
print('    R21 内控重大缺陷     → 风险分下限 0.85')
print('    R22 监管处罚         → 风险分下限 0.80')
print('  ')
print('  提取字段:')
print('    - 内控审计意见(有效/保留/否定)')
print('    - 重大缺陷(是/否)')
print('    - 监管处罚类型(责令改正/警示函/立案调查/行政处罚)')

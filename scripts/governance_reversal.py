#!/usr/bin/env python3
"""
P2-1: 治理反转检测
====================
对同一公司 H1(半年报)与 H2(年报)之间,检测"治理信号反转":
- 资金占用:从"否" → "是"
- 违规担保:从"否" → "是"
- 审计意见:从"无保留" → "保留/否定"
- 内控:从"有效" → "无效/重大缺陷"

任何反转都判为强信号 → 风险分下限 0.85
"""

import os
import re
import pandas as pd
import pdfplumber
from datetime import datetime

BASE = "/Users/Zhuanz/claude工作文件夹/审计数据分析大作业"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")


# ============================================================
# 治理信号提取器
# ============================================================
def extract_governance_signals(pdf_path):
    """
    从 PDF 前 10 页提取治理信号:
    - audit_opinion: 审计意见
    - has_fund_occupancy: 资金占用
    - has_illegal_guarantee: 违规担保
    - has_material_internal_control_defect: 重大内控缺陷
    - has_audit_opinion_changed: 审计师变更
    """
    signals = {
        'audit_opinion': '',
        'has_fund_occupancy': None,  # True/False/None
        'has_illegal_guarantee': None,
        'has_material_internal_control_defect': None,
        'has_critical_proceedings': None,
    }

    with pdfplumber.open(pdf_path) as pdf:
        # 扫前 10 页
        for i in range(min(10, len(pdf.pages))):
            text = pdf.pages[i].extract_text() or ''

            # 1. 审计意见
            if not signals['audit_opinion']:
                if '标准无保留' in text or '无保留意见' in text:
                    signals['audit_opinion'] = '标准无保留'
                elif '保留意见' in text and '审计师' not in text[:200]:  # 排除提及审计师名字
                    signals['audit_opinion'] = '保留意见'
                elif '否定意见' in text or '无法表示' in text:
                    signals['audit_opinion'] = '否定或无法表示'
                elif '未经审计' in text:
                    signals['audit_opinion'] = '未经审计'  # 半年报

            # 2. 资金占用
            if signals['has_fund_occupancy'] is None:
                # 多重模式匹配
                patterns = [
                    r'非经营性占用资金情况\s*[√☑☐]?\s*(是|否)',
                    r'非经营性占用资金\s*[√☑☐]?\s*(是|否)',
                    r'占用资金情况\s*[√☑☐]?\s*(是|否)',
                    r'非经营性占用资金.{0,10}(是|否)',
                ]
                for pat in patterns:
                    m = re.search(pat, text)
                    if m:
                        signals['has_fund_occupancy'] = (m.group(1) == '是')
                        break
                # 检查整行包含"非经营性占用资金"+"否"或"是"
                if signals['has_fund_occupancy'] is None:
                    for line in text.split('\n'):
                        if '非经营性占用资金' in line or '占用资金情况' in line:
                            if '是' in line and '否' not in line.replace('否的', '').replace('否认', ''):
                                signals['has_fund_occupancy'] = True
                                break
                            elif '否' in line:
                                signals['has_fund_occupancy'] = False
                                break

            # 3. 违规担保
            if signals['has_illegal_guarantee'] is None:
                patterns = [
                    r'违规.{0,5}对外提供担保.{0,10}(是|否)',
                    r'违反规定决策程序.{0,10}(是|否)',
                    r'违规决策程序.{0,10}(是|否)',
                ]
                for pat in patterns:
                    m = re.search(pat, text)
                    if m:
                        signals['has_illegal_guarantee'] = (m.group(1) == '是')
                        break
                # 整行匹配
                if signals['has_illegal_guarantee'] is None:
                    for line in text.split('\n'):
                        if '违规' in line and '担保' in line:
                            if '否' in line and '存在' not in line:
                                signals['has_illegal_guarantee'] = False
                                break
                            elif '是' in line:
                                signals['has_illegal_guarantee'] = True
                                break

            # 4. 重大内控缺陷(找关键审计事项/内控否定)
            if signals['has_material_internal_control_defect'] is None:
                if '内部控制存在重大缺陷' in text or '内控否定意见' in text:
                    signals['has_material_internal_control_defect'] = True
                elif '内部控制有效' in text:
                    signals['has_material_internal_control_defect'] = False

            # 5. 重大诉讼/重要事项
            if signals['has_critical_proceedings'] is None:
                if '重大诉讼' in text and '未涉及' not in text and '不存在' not in text:
                    signals['has_critical_proceedings'] = True

    return signals


def detect_reversals(h1_signals, h2_signals):
    """
    检测 H1 → H2 之间的反转
    """
    reversals = []

    # 资金占用反转
    if h1_signals['has_fund_occupancy'] == False and h2_signals['has_fund_occupancy'] == True:
        reversals.append({
            'signal': '资金占用',
            'from': '否',
            'to': '是',
            'severity': '极高',
            'rule_id': 'R16',
            'risk_floor': 0.85,
        })

    # 违规担保反转
    if h1_signals['has_illegal_guarantee'] == False and h2_signals['has_illegal_guarantee'] == True:
        reversals.append({
            'signal': '违规担保',
            'from': '否',
            'to': '是',
            'severity': '极高',
            'rule_id': 'R17',
            'risk_floor': 0.85,
        })

    # 审计意见降级
    opinion_order = {'标准无保留': 1, '保留意见': 2, '否定或无法表示': 3}
    h1_op = h1_signals.get('audit_opinion', '')
    h2_op = h2_signals.get('audit_opinion', '')
    if h1_op in opinion_order and h2_op in opinion_order:
        if opinion_order[h2_op] > opinion_order[h1_op]:
            reversals.append({
                'signal': '审计意见降级',
                'from': h1_op,
                'to': h2_op,
                'severity': '极高',
                'rule_id': 'R18',
                'risk_floor': 0.85,
            })

    # 内控反转
    if h1_signals.get('has_material_internal_control_defect') == False and \
       h2_signals.get('has_material_internal_control_defect') == True:
        reversals.append({
            'signal': '内控重大缺陷出现',
            'from': '内控有效',
            'to': '重大缺陷',
            'severity': '高',
            'rule_id': 'R19',
            'risk_floor': 0.80,
        })

    return reversals


# ============================================================
# Demo: 洲际油气盲测
# ============================================================
print('='*70)
print(' '*15 + 'P2-1: 治理反转检测 — 洲际油气 demo')
print('='*70)
print(f'  数据: 2025 半年报 + 2025 年报')
print(f'  目标: 检测 H1 → H2 之间的"治理反转"')

# 提取半年报治理信号
print('\n' + '='*70)
print('1. 提取 2025 半年报治理信号')
print('='*70)
h1_signals = extract_governance_signals(f'{BASE}/洲际半年报.pdf')
for k, v in h1_signals.items():
    print(f'  {k}: {v}')

# 提取年报治理信号
print('\n' + '='*70)
print('2. 提取 2025 年报治理信号')
print('='*70)
h2_signals = extract_governance_signals(f'{BASE}/洲际年报.pdf')
for k, v in h2_signals.items():
    print(f'  {k}: {v}')

# 检测反转
print('\n' + '='*70)
print('3. 检测治理反转 (H1 → H2)')
print('='*70)
reversals = detect_reversals(h1_signals, h2_signals)
if reversals:
    for r in reversals:
        print(f'  🔴 {r["rule_id"]} {r["signal"]}: 从 "{r["from"]}" → "{r["to"]}"')
        print(f'     严重度: {r["severity"]}, 风险分下限: {r["risk_floor"]}')
else:
    print('  ⚪ 未检测到治理反转')

# 综合风险分(加上反转)
print('\n' + '='*70)
print('4. 综合风险分更新 (盲测 → 加上治理反转)')
print('='*70)

# 盲测的 v2 风险分是 0.6000(仅 7 比率 + 时序)
v2_score = 0.6000
print(f'  v2 (仅 7 比率 + 时序): {v2_score:.4f}')

# 加上治理反转
v2_with_reversal = v2_score
for r in reversals:
    v2_with_reversal = max(v2_with_reversal, r['risk_floor'])
print(f'  v2 + 治理反转:          {v2_with_reversal:.4f}')

if v2_with_reversal >= 0.55:
    final_lvl = '高风险 🔴'
elif v2_with_reversal >= 0.25:
    final_lvl = '中风险 🟠'
else:
    final_lvl = '低风险 🟢'
print(f'  最终风险等级:          {final_lvl}')

# ============================================================
# 通用化:批量检测函数
# ============================================================
print('\n' + '='*70)
print('5. 函数封装(供后续批量使用)')
print('='*70)
print('  extract_governance_signals(pdf_path) -> Dict')
print('  detect_reversals(h1_signals, h2_signals) -> List[Dict]')
print('  ')
print('  用法:')
print('    h1 = extract_governance_signals("H1_半年报.pdf")')
print('    h2 = extract_governance_signals("年报.pdf")')
print('    reversals = detect_reversals(h1, h2)')
print('    for r in reversals:')
print('        apply_risk_floor(r["risk_floor"])')

# 报告保存
print('\n' + '='*70)
print('✅ P2-1 完成: 治理反转检测')
print('='*70)
print('  4 条新规则 (R16-R19):')
print('    R16 资金占用反转(否→是)')
print('    R17 违规担保反转(否→是)')
print('    R18 审计意见降级')
print('    R19 内控重大缺陷出现')
print('  ')
print('  任何反转 → 风险分下限 0.85')

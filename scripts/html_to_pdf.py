#!/usr/bin/env python3
"""
阶段四·步骤 8b: HTML → PDF
============================
用 playwright + chromium 渲染 (weasyprint 在 macOS 处理中文字体有 bug)
"""

import os
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = Path('/Users/Zhuanz/claude工作文件夹/审计数据分析大作业')
html_path = BASE / 'reports' / '审计数据分析大作业_何其轩_2023111180.md.html'
pdf_path = BASE / 'reports' / '审计数据分析大作业_何其轩_2023111180.pdf'

print("=" * 60)
print("HTML → PDF 转换 (Playwright + Chromium)")
print("=" * 60)

if not html_path.exists():
    print(f"❌ HTML 文件不存在: {html_path}")
    print("   请先运行: python scripts/md_to_html.py")
    raise SystemExit(1)

print(f"读取 HTML: {html_path}")
print(f"  文件大小: {html_path.stat().st_size / 1024:.0f} KB")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    # 加载本地 HTML 文件
    page.goto(f"file://{html_path}")
    page.wait_for_load_state('networkidle')

    # 渲染 PDF
    page.pdf(
        path=str(pdf_path),
        format='A4',
        margin={'top': '2cm', 'right': '2cm', 'bottom': '2cm', 'left': '2cm'},
        print_background=True,
    )

    browser.close()

size_kb = pdf_path.stat().st_size / 1024
print(f"\n  → {pdf_path}")
print(f"  文件大小: {size_kb:.0f} KB")

if size_kb < 100:
    print("⚠️ PDF 较小,可能内容渲染不完整")

print("\n" + "=" * 60)
print("✅ HTML → PDF 转换完成")
print("=" * 60)

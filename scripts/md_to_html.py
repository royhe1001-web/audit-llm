#!/usr/bin/env python3
"""
阶段四·步骤 8a: MD → HTML
============================
用 markdown 库渲染,图片转 base64 内嵌,生成自包含 HTML
"""

import os
import re
import base64
from pathlib import Path
import markdown

BASE = Path('/Users/Zhuanz/claude工作文件夹/审计数据分析大作业')
md_path = BASE / 'reports' / '审计数据分析大作业_何其轩_2023111180.md'
html_path = BASE / 'reports' / '审计数据分析大作业_何其轩_2023111180.md.html'

print("=" * 60)
print("MD → HTML 转换")
print("=" * 60)

# 1. 读取 MD
md_text = md_path.read_text(encoding='utf-8')
print(f"读取 MD: {len(md_text)} 字符")

# 2. 渲染 HTML
html_body = markdown.markdown(
    md_text,
    extensions=['tables', 'fenced_code', 'toc', 'codehilite', 'sane_lists']
)
print(f"渲染 HTML body: {len(html_body)} 字符")

# 3. 图片转 base64 内嵌(相对路径解析)
def inline_images(html, base_dir):
    pattern = r'<img\s+[^>]*src="([^"]+)"[^>]*>'
    matches = re.findall(pattern, html)
    inlined = 0
    for src in matches:
        if src.startswith('data:'):
            continue
        img_path = base_dir / src
        if not img_path.exists():
            # 尝试相对当前工作目录
            img_path = Path.cwd() / src
        if img_path.exists():
            try:
                ext = img_path.suffix.lstrip('.').lower()
                mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                        'gif': 'image/gif', 'svg': 'image/svg+xml'}.get(ext, 'image/png')
                b64 = base64.b64encode(img_path.read_bytes()).decode()
                new_src = f"data:{mime};base64,{b64}"
                html = html.replace(f'src="{src}"', f'src="{new_src}"', 1)
                inlined += 1
            except Exception as e:
                print(f"  ⚠️ 图片 {src} 内嵌失败: {e}")
    return html, inlined

html_body, n_images = inline_images(html_body, BASE)
print(f"图片内嵌: {n_images} 张")

# 4. 完整 HTML 模板
css = """
<style>
@page { size: A4; margin: 2cm; }
body {
    font-family: 'Arial Unicode MS', sans-serif;
    line-height: 1.8;
    color: #222;
    max-width: 900px;
    margin: 30px auto;
    padding: 0 20px;
    background: white;
}
h1 {
    color: #1f3a5f;
    border-bottom: 3px solid #1f3a5f;
    padding-bottom: 10px;
    margin-top: 40px;
    font-size: 28px;
}
h2 {
    color: #1f3a5f;
    border-bottom: 1px solid #1f3a5f;
    padding-bottom: 6px;
    margin-top: 30px;
    font-size: 22px;
}
h3 {
    color: #2c5282;
    margin-top: 24px;
    font-size: 18px;
}
h4 { color: #2c5282; }
p { margin: 12px 0; }
code {
    background: #f0f4f8;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Menlo', 'Consolas', monospace;
    font-size: 0.9em;
    color: #c7254e;
}
pre {
    background: #f8f9fa;
    border-left: 4px solid #1f3a5f;
    padding: 12px 16px;
    overflow-x: auto;
    border-radius: 4px;
    font-size: 0.9em;
}
pre code {
    background: none;
    color: #333;
    padding: 0;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
    font-size: 14px;
}
th {
    background: #1f3a5f;
    color: white;
    padding: 10px;
    text-align: left;
    border: 1px solid #ddd;
}
td {
    border: 1px solid #ddd;
    padding: 8px;
    vertical-align: top;
}
tr:nth-child(even) { background: #f8f9fa; }
tr:hover { background: #f0f4f8; }
img {
    max-width: 100%;
    display: block;
    margin: 20px auto;
    border-radius: 4px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
hr { border: none; border-top: 2px solid #e2e8f0; margin: 30px 0; }
blockquote {
    border-left: 4px solid #cbd5e0;
    padding-left: 16px;
    color: #4a5568;
    font-style: italic;
    margin: 16px 0;
}
a { color: #2c5282; text-decoration: none; border-bottom: 1px dotted #2c5282; }
ul, ol { padding-left: 30px; }
li { margin: 6px 0; }
</style>
"""

# 标题页
title_block = f"""
<div style="text-align: center; padding: 40px 0; border-bottom: 3px solid #1f3a5f; margin-bottom: 40px;">
    <div style="font-size: 12px; color: #888; letter-spacing: 2px;">审计数据分析大作业</div>
    <h1 style="border: none; font-size: 32px; margin: 10px 0;">上市公司财务舞弊智能识别<br/>与持续审计系统</h1>
    <div style="margin-top: 20px; color: #4a5568; font-size: 14px;">
        <strong>作者:</strong> 何其轩 &nbsp;|&nbsp; <strong>学号:</strong> 2023111180 &nbsp;|&nbsp; <strong>完成日期:</strong> 2026-06-06
    </div>
</div>
"""

# 完整 HTML
full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>上市公司财务舞弊智能识别与持续审计系统</title>
    {css}
</head>
<body>
    {title_block}
    {html_body}
    <hr/>
    <div style="text-align: center; color: #888; font-size: 12px; margin-top: 40px;">
        <p>本报告由审计数据分析大作业项目自动生成</p>
        <p>作者:何其轩 (2023111180) | 2026-06-06</p>
    </div>
</body>
</html>"""

# 5. 写入
html_path.write_text(full_html, encoding='utf-8')
size_kb = html_path.stat().st_size / 1024
print(f"\n  → {html_path}")
print(f"  文件大小: {size_kb:.0f} KB")
print(f"  内嵌图片: {n_images} 张")

print("\n" + "=" * 60)
print("✅ MD → HTML 转换完成")
print("=" * 60)

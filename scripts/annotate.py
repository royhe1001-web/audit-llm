#!/usr/bin/env python3
"""
财务舞弊 AI 标注脚本
====================
使用 Anthropic 协议调用三个候选模型，完成三个任务的标注。
每个任务产出独立的结果文件，供 F1 对比和最终合并。

用法:
  # Pilot: 对 5% 样本标注
  python annotate.py --model mimo    --task all --input sample_G04_for_human_label.csv --output results/
  python annotate.py --model minimax --task all --input sample_G04_for_human_label.csv --output results/
  python annotate.py --model deepseek --task all --input sample_G04_for_human_label.csv --output results/

  # 全量: 分任务执行（用选定模型）
  python annotate.py --model <best> --task task1 --input full_data.csv --output results/
"""

import os, sys, json, time, argparse, re
from pathlib import Path
import requests
import pandas as pd

# ============================================================
# 模型配置
# ============================================================
MODEL_CONFIGS = {
    "mimo": {
        "name": "Mimo-v2.5-pro",
        "base_url": "https://api.xiaomimimo.com/anthropic",
        "api_key": "sk-cyk8esbwcn8iigd71a3qyqj9xbknr2fov0gkzgjhq2w1b1u8",
        "model": "mimo-v2.5-pro",
    },
    "minimax": {
        "name": "MiniMax-M2.7",
        "base_url": "https://api.minimaxi.com/anthropic",
        "api_key": "sk-api-mpG58Vfbq_73kWdXkfCEAM1_PV4n6ehehBArCmHaVAUGeUNPTp_eWA2NcJwdt-sTkpPNmh4ulLNBCU1yYI-G61c0iUlVA6805UhKxgaTOO2fqOYIUgdLheI",
        "model": "MiniMax-M2.7",
    },
    "deepseek": {
        "name": "DeepSeek-V4-pro",
        "base_url": "https://api.deepseek.com/anthropic",
        "api_key": "sk-2f04c03bfcbc4115ac57600168b8fa09",
        "model": "deepseek-v4-pro",
    },
}

# ============================================================
# Prompt 定义
# ============================================================

# --- 任务一：年报相关性 ---
TASK1_SYSTEM = """你是一名中国证监会行政处罚决定书的专业分析助手。你的任务是判断一段违规行为描述是否导致上市公司年度报告中的信息存在错误或遗漏，并提取涉及的年份。

## 判断标准
**标注为 1（年报相关）的条件：**
- 违规行为直接导致上市公司**年度报告**（含合并报表）中存在虚假记载、重大遗漏或误导性陈述
- 无论是财务数据还是非财务信息（如重大事项披露、关联方披露等），只要影响年报即标 1
- 文本中明确提及"××年年度报告"存在问题

**标注为 0（非年报相关）的条件：**
- 仅影响季报、半年报，未影响年报
- 资金占用、违规担保（若不涉及年报信息披露差异）
- 实控人身份隐瞒、关联交易未按要求报批（若不导致年报中数据错误）
- 内幕交易、市场操纵类违规（不影响财务报告本身）
- 减持违规、短线交易等与年报无关的行为
- 仅涉及对交易所问询函的回复延迟（非年报内容本身错误）

**年份提取规则：**
- 若 ann_related=1，提取所有受影响的年份（四位数字），以列表形式返回
- 年份通常出现在"××年年度报告"、"2017年年报"等表述中
- 如果虽然 ann_related=1 但无法确定具体年份，填 null

## 输出格式（严格遵守）
你必须只输出一个 JSON 对象，不要包含任何其他文字。格式如下：
{"ann_related": 0或1, "ann_year": [年份列表] 或 null}"""

TASK1_USER = """请分析以下违规行为描述，判断是否影响年报，并输出 JSON：

{activity}"""

# --- 任务二：财务信息与会计要素 ---
TASK2_SYSTEM = """你是一名中国证监会行政处罚决定书的专业分析助手。本任务仅在已判定违规行为影响年报（ann_related=1）的前提下执行。

## 步骤一：判断是否影响财务信息
- **财务信息**：指年报中的财务报表及相关附注，包括资产负债表、利润表、现金流量表中的任何科目金额错误
- **非财务信息**：指年报中公司治理、重大事项、关联方信息等描述性内容，不含具体金额错误
- ann_fin_flag = 1：影响财务信息（金额不实）
- ann_fin_flag = 0：仅影响非财务信息（披露不完整但金额正确）

## 步骤二：识别受影响的会计要素
若 ann_fin_flag=1，识别受影响的会计要素。**只能从以下六大要素中选取：**
- **资产**（如货币资金、应收账款、存货、固定资产等）
- **负债**（如应付账款、短期借款、预收款项等）
- **所有者权益**（如实收资本、盈余公积、未分配利润等）
- **收入**（如营业收入、其他业务收入、投资收益等）
- **费用**（如营业成本、期间费用、所得税费用等）
- **利润**（如营业利润、净利润、综合收益等）

**严禁填写具体科目名称（如"应收账款""营业收入"），必须归入六大要素。**

## 步骤三：年份与要素一一对应
请将每个受影响年份与其对应的会计要素列表一一匹配。不同年份可能影响不同的要素组合。

## 输出格式（严格遵守）
你必须只输出一个 JSON 对象：
{"ann_fin_flag": 0或1, "ann_fin_info": [{"year": 年份, "elements": ["要素1","要素2",...]}] 或 null}

如果 ann_fin_flag=0，ann_fin_info 填 null。"""

TASK2_USER = """以下违规行为已确认影响年报（ann_related=1），受影响的年份为 {ann_year}。请判断是否影响财务信息，并识别受影响的会计要素。

违规行为描述：
{activity}

请输出 JSON："""

# --- 任务三：第三方配合造假 ---
TASK3_SYSTEM = """你是一名中国证监会行政处罚决定书的专业分析助手。本任务仅在已确认违规影响年报财务信息（ann_related=1 且 ann_fin_flag=1）的前提下执行。

## 核心定义

**第三方（Third Party）的界定：**
第三方是指上市公司本身以外的独立法人或自然人，且**不包括**：
- 上市公司自身（及其全资/控股子公司）
- 上市公司的实际控制人本身（直接参与舞弊但属内部决策者）

**常见类型：** 客户、供应商、银行/金融机构、券商/保荐机构、会计师事务所、评估机构、自然人、其他企业。

**配合（Collusion）的界定：**
第三方行为必须构成"主动配合"，即明知上市公司实施舞弊，仍提供实质性协助：
- 合谋虚构交易：签订虚假合同，提供不实发票或凭证
- 协助资金造假：参与资金循环出账，提供虚假银行流水或配合资金回流
- 提供关键便利：出借账户、协助周转舞弊资金、代持股份掩盖关联关系

**不属于配合的情况：**
- 正常业务往来
- 被动配合监管调查
- 仅以主体名义出现而无舞弊协助行为
- 审计机构出具了审计意见但未提及参与造假

## 字段要求
- third_party_flag: 1=有第三方配合, 0=无
- third_party_list: 若有，提取每个第三方的：
  - name: 完整法定名称（如"北京恒达鑫泰物流有限公司"），若文本只有简称则原样提取
  - type: 从「客户、供应商、银行/金融机构、券商/保荐机构、会计师事务所、评估机构、自然人、其他企业」中选取
  - role: 简述其配合行为（一句话）

## 输出格式（严格遵守）
你必须只输出一个 JSON 对象：
{"third_party_flag": 0或1, "third_party_list": [{"name":"...","type":"...","role":"..."}] 或 null}

如果 third_party_flag=0，third_party_list 填 null。"""

TASK3_USER = """以下违规行为已确认同时影响年报和财务信息。请判断是否存在第三方主动配合造假，并提取相关信息。

违规行为描述：
{activity}

请输出 JSON："""

# ============================================================
# API 调用
# ============================================================

def call_llm(cfg, system_prompt, user_prompt, max_retries=3):
    """调用 Anthropic 协议 API（直接 HTTP 请求，避免 SDK 连接复用问题）"""
    url = cfg["base_url"].rstrip("/") + "/v1/messages"
    headers = {
        "x-api-key": cfg["api_key"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": cfg["model"],
        "max_tokens": 4096,
        "system": system_prompt + "\n\nCRITICAL: Output ONLY a raw JSON object. No markdown, no extra text.",
        "messages": [{"role": "user", "content": user_prompt}],
        "thinking": {"type": "disabled"},
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                # 遍历所有 content 块，找 type=text 的（跳过 thinking 块）
                for block in r.json().get("content", []):
                    if block.get("type") == "text":
                        return block["text"]
                # 降级：如果全是 thinking，取最后一块（某些模型可能把最终答案放在 thinking 里）
                blocks = r.json().get("content", [])
                if blocks:
                    last = blocks[-1]
                    return last.get("text") or last.get("thinking", "")
                return None
            else:
                print(f"  ⚠ HTTP {r.status_code}: {r.text[:200]}")
        except requests.Timeout:
            print(f"  ⚠ 超时 (尝试 {attempt+1}/{max_retries})")
        except Exception as e:
            print(f"  ⚠ 尝试 {attempt+1}/{max_retries}: {e}")
        if attempt < max_retries - 1:
            time.sleep(2)
    time.sleep(0.8)  # 避免触发限流
    return None


def parse_json_response(text):
    """从 LLM 回复中提取 JSON"""
    if text is None:
        return None
    text = text.strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except:
        pass
    # 尝试提取 markdown code block
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass
    # 尝试找到第一个 { 和最后一个 }
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    print(f"  ❌ JSON 解析失败，原始输出: {text[:300]}")
    return None


# ============================================================
# 标注逻辑
# ============================================================

def run_task1(df, cfg, out_path):
    """任务一：年报相关性识别（全量），支持断点续传"""
    # 检查已有进度
    import os as _os
    ckpt_path = out_path.replace(".csv", "_ckpt.csv")
    results = []
    start = 0
    if _os.path.exists(ckpt_path):
        ckpt = pd.read_csv(ckpt_path)
        results = [{"ann_related": r.get("ann_related"), "ann_year": r.get("ann_year")}
                   for _, r in ckpt.iterrows()]
        start = len(results)
        print(f"  🔄 从第 {start} 行继续（已处理 {start}/{len(df)}）")

    total = len(df)
    for i in range(start, total):
        row = df.iloc[i]
        activity = row["Activity"]
        user_prompt = TASK1_USER.format(activity=activity)
        text = call_llm(cfg, TASK1_SYSTEM, user_prompt)
        parsed = parse_json_response(text)

        ann_related = None
        ann_year = None
        if parsed:
            ann_related = parsed.get("ann_related")
            ann_year = parsed.get("ann_year")

        results.append({"ann_related": ann_related, "ann_year": ann_year})

        if (i + 1) % 5 == 0 or i == start or i == total - 1:
            sys.stdout.write(f"  [{i+1}/{total}] ann_related=1:{sum(1 for r in results if r['ann_related']==1)}\n")
            sys.stdout.flush()
            ckpt = pd.DataFrame(results)
            ckpt.to_csv(ckpt_path, index=False)

    out_df = df.copy()
    out_df["ann_related"] = [r["ann_related"] for r in results]
    out_df["ann_year"] = [r["ann_year"] for r in results]
    out_df.to_csv(out_path, index=False)
    if _os.path.exists(ckpt_path):
        _os.remove(ckpt_path)
    print(f"  ✅ 任务一完成 → {out_path}")
    return out_df


def run_task2(df, cfg, out_path):
    """任务二：财务信息与会计要素（仅 ann_related=1 的行）"""
    results = []
    total = len(df)
    processed = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if row.get("ann_related") != 1:
            results.append({"ann_fin_flag": None, "ann_fin_info": None})
        else:
            processed += 1
            activity = row["Activity"]
            ann_year = row.get("ann_year", "未知")
            user_prompt = TASK2_USER.format(activity=activity, ann_year=ann_year)
            text = call_llm(cfg, TASK2_SYSTEM, user_prompt)
            parsed = parse_json_response(text)

            ann_fin_flag = None
            ann_fin_info = None
            if parsed:
                ann_fin_flag = parsed.get("ann_fin_flag")
                ann_fin_info = parsed.get("ann_fin_info")
            results.append({"ann_fin_flag": ann_fin_flag, "ann_fin_info": ann_fin_info})

        if (i + 1) % 5 == 0 or i == 0:
            c1 = sum(1 for r in results if r.get("ann_fin_flag") == 1)
            sys.stdout.write(f"  [{i+1}/{total}] need={processed}, fin=1:{c1}\n")
            sys.stdout.flush()

    out_df = df.copy()
    out_df["ann_fin_flag"] = [r["ann_fin_flag"] for r in results]
    out_df["ann_fin_info"] = [r["ann_fin_info"] for r in results]
    out_df.to_csv(out_path, index=False)
    print(f"  ✅ 任务二完成 → {out_path}")
    return out_df


def run_task3(df, cfg, out_path):
    """任务三：第三方配合（仅 ann_related=1 且 ann_fin_flag=1 的行）"""
    results = []
    total = len(df)
    processed = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if not (row.get("ann_related") == 1 and row.get("ann_fin_flag") == 1):
            results.append({"third_party_flag": None, "third_party_list": None})
        else:
            processed += 1
            activity = row["Activity"]
            user_prompt = TASK3_USER.format(activity=activity)
            text = call_llm(cfg, TASK3_SYSTEM, user_prompt)
            parsed = parse_json_response(text)

            third_party_flag = None
            third_party_list = None
            if parsed:
                third_party_flag = parsed.get("third_party_flag")
                third_party_list = parsed.get("third_party_list")
            results.append({"third_party_flag": third_party_flag, "third_party_list": third_party_list})

        if (i + 1) % 5 == 0 or i == 0:
            c1 = sum(1 for r in results if r.get("third_party_flag") == 1)
            sys.stdout.write(f"  [{i+1}/{total}] need={processed}, tp=1:{c1}\n")
            sys.stdout.flush()

    out_df = df.copy()
    out_df["third_party_flag"] = [r["third_party_flag"] for r in results]
    out_df["third_party_list"] = [r["third_party_list"] for r in results]
    out_df.to_csv(out_path, index=False)
    print(f"  ✅ 任务三完成 → {out_path}")
    return out_df


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="财务舞弊 AI 标注")
    parser.add_argument("--model", required=True, choices=["mimo", "minimax", "deepseek"],
                        help="选择模型")
    parser.add_argument("--task", required=True, choices=["task1", "task2", "task3", "all"],
                        help="任务类型 (all = 三个任务顺序执行)")
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", required=True, help="输出目录")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    print(f"模型: {cfg['name']} ({args.model})")
    print(f"任务: {args.task}")

    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f"数据: {len(df)} 行")

    prefix = f"pred_{args.model}"
    t0 = time.time()

    if args.task in ("task1", "all"):
        out = os.path.join(args.output, f"{prefix}_task1.csv")
        df = run_task1(df, cfg, out)

    if args.task in ("task2", "all"):
        out = os.path.join(args.output, f"{prefix}_task2.csv")
        df = run_task2(df, cfg, out)

    if args.task in ("task3", "all"):
        out = os.path.join(args.output, f"{prefix}_task3.csv")
        df = run_task3(df, cfg, out)

    elapsed = time.time() - t0
    print(f"\n🏁 全部完成，耗时 {elapsed/60:.1f} 分钟")


if __name__ == "__main__":
    main()

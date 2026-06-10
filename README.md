# Audit LLM — 上市公司财务舞弊识别 + 持续审计系统

> 基于审计数据分析的端到端项目:从数据采集、特征工程、舞弊识别到风险评分、异常检测、Streamlit 仪表盘与审计工作底稿。

## 📋 项目简介


本项目实现了一套**完整的上市公司财务舞弊识别 + 持续审计系统**,核心能力包括:

- 🔍 **舞弊识别**:RF/XGBoost/LightGBM/Stacking 多模型对比,CV F1 ≈ 0.76
- 📊 **风险评分**:模型概率 + 审计规则融合(0.6 模型 + 0.4 规则)
- 🚨 **异常检测**:IsolationForest + LOF 双算法无监督兜底
- 📋 **审计规则引擎**:6 条专家规则(连续亏损、现金流背离、高负债等)
- 🌐 **Streamlit 仪表盘**:5 Tab 总览/行业/公司画像/Top100/异常检测
- 📑 **审计中间表**:7,997 行 × 40+ 列,作为单一权威数据源
- 📊 **Excel 工作底稿**:8 Sheet(数据/特征/模型/Top100/规则/行业/异常/仪表盘说明)
- 📝 **审计报告**:8 章技术总结 + 单公司风险预警报告

---

## 🏗️ 项目架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      数据源层(STK_Violation_Main.xlsx)            │
└─────────────────┬───────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│              数据处理层(scripts/data_quality.py)                  │
│   清洗 · 补全 · 财务特征工程 · 行业数据补全(tushare)              │
└─────────────────┬───────────────────────────────────────────────┘
                  ↓
┌──────────────────────────────┬──────────────────────────────────┐
│    舞弊识别层(ML Models)      │    审计规则层(Expert Rules)      │
│   RF / XGB / LGBM / Stacking  │   6 条审计规则                   │
│   CV F1 ≈ 0.76                │   R1-R6 严重度评分              │
└─────────────────┬──────────────┴──────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│              风险评分层(模型 0.6 + 规则 0.4)                      │
│                  高中低三级(高 ≥0.7 / 中 0.4-0.7 / 低 <0.4)       │
└─────────────────┬───────────────────────────────────────────────┘
                  ↓
┌──────────────────────────────┬──────────────────────────────────┐
│   异常检测层(无监督)          │     持续审计层(动态规则)          │
│   IsolationForest + LOF       │     新增规则/调整阈值            │
└─────────────────┬──────────────┴──────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│   中间表 → Streamlit 仪表盘 + Excel 工作底稿 + 风险预警报告         │
│   审计中间表(7,997 行 × 40+ 列)· 8 Sheet 工作底稿 · 单公司报告   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 目录结构

```
audit-llm/
├── app/                          # Streamlit 应用
│   ├── dashboard.py              # 5 Tab 仪表盘
│   └── report_generator.py       # 报告生成
├── scripts/                      # 9 大处理脚本
│   ├── fetch_industry.py         # tushare 行业数据补全
│   ├── audit_rules.py            # 6 条审计规则引擎
│   ├── risk_scoring.py           # 模型 + 规则融合
│   ├── multi_dim_analysis.py     # 多维分析
│   ├── anomaly_detection.py      # 异常检测
│   ├── build_intermediate_table.py  # 审计中间表
│   ├── md_to_pdf.py              # 报告 → PDF
│   ├── html_to_pdf.py            # 报告 → PDF(HTML)
│   └── build_workpaper.py        # 8 Sheet 工作底稿
├── models/                       # 训练好的模型
│   ├── fraud_detection_rf_combined.pkl  # RF Pipeline
│   ├── fraud_detection_xgb.pkl          # XGBoost
│   ├── fraud_detection_lgb.pkl          # LightGBM
│   ├── fraud_detection_stacking.pkl     # Stacking
│   └── risk_scoring_pipeline.pkl        # 风险评分
├── notebooks/                    # 实验 notebook
│   ├── fraud_detection_model.py  # 模型训练
│   └── API_match_实战_公司名匹配.ipynb
├── output/                       # 输出结果
│   ├── audit_workpaper.xlsx      # 8 Sheet 工作底稿
│   ├── category_layer_stats.xlsx # 行业/年份/地区统计
│   ├── risk_scored_full.csv      # 7,997 行风险评分
│   ├── top100_high_risk.csv      # Top100 高风险
│   ├── anomaly_companies.csv     # 异常检测结果
│   ├── rule_trigger_aggregate.csv # 规则触发汇总
│   └── *.png                     # 12+ 可视化图表
├── reports/                      # 最终报告
│   ├── 三佳科技_600520_风险预警报告.md
│   └── 洲际油气_600759_风险预警报告.md
└── README.md                     # 本文件
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 3.11+
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn streamlit plotly openpyxl weasyprint tushare
```

### 2. 运行流水线(端到端)

```bash
cd audit-llm

# 1) 行业数据补全(tushare)
python scripts/fetch_industry.py

# 2) 审计规则引擎 + 6 条规则
python scripts/audit_rules.py

# 3) 风险评分(模型 + 规则融合)
python scripts/risk_scoring.py

# 4) 异常检测
python scripts/anomaly_detection.py

# 5) 多维分析(行业/年份/地区)
python scripts/multi_dim_analysis.py

# 6) 构建审计中间表
python scripts/build_intermediate_table.py

# 7) 构建 8 Sheet 工作底稿
python scripts/build_workpaper.py

# 8) 启动仪表盘
streamlit run app/dashboard.py
```

### 3. 查看结果

- **Excel 工作底稿**:`output/audit_workpaper.xlsx`(8 Sheet)
- **风险评分**:`output/risk_scored_full.csv`(7,997 行)
- **Top100 高风险**:`output/top100_high_risk.csv`
- **报告**:`reports/审计数据分析大作业_何其轩_2023111180.pdf`

---

## 🎯 核心设计

### 1. 风险评分融合公式

```
risk_score = 0.6 × P(模型预测舞弊) + 0.4 × (规则触发严重度 / 125)
```

| 阈值 | 等级 |
|---|---|
| ≥ 0.7 | 高风险(立即审计) |
| 0.4-0.7 | 中风险(关注) |
| < 0.4 | 低风险 |

### 2. 6 条审计规则

| # | 规则 | 触发条件 | 严重度 |
|---|---|---|---|
| R1 | 连续亏损 | roe<0 AND net_margin<0 | 25 |
| R2 | 现金流背离 | ocf_to_rev<0 AND net_margin>0 | 30 |
| R3 | 高负债 | debt_ratio>0.7 | 15 |
| R4 | 流动性紧张 | current_ratio<1 | 20 |
| R5 | 资产周转异常 | asset_turnover<0.3 OR >5 | 10 |
| R6 | ROE 异常 | roa>0.1 AND roe<0 | 25 |

(总分 125,归一化分母)

### 3. Streamlit 仪表盘(5 Tab)

| Tab | 内容 |
|---|---|
| 📊 总览 | 4 KPI 卡 + 风险饼图 + 分数直方图 + 全局筛选 |
| 🏭 行业 | 行业排名 + 时间趋势 + 行业-年份热力图 |
| 🏢 公司画像 | 7 维雷达图 + ML 概率条 + 6 规则红绿灯 + 历史违规 |
| 🚨 Top100 | 可排序 DataFrame + RF 特征重要性条形图 |
| ⚠️ 异常检测 | IF 散点图(PCA 投影) + 异常公司列表 |

### 4. 8 Sheet Excel 工作底稿

| Sheet | 内容 |
|---|---|
| 1. 数据概览 | enriched.csv 统计 |
| 2. 财务特征分析 | describe + 相关性矩阵 |
| 3. 模型评估 | 硬编码指标 + 特征重要性 |
| 4. Top100 高风险 | top100_high_risk.csv |
| 5. 规则触发 | rule_trigger_aggregate.csv |
| 6. 行业分析 | 行业 Sheet |
| 7. 异常检测 | anomaly_companies.csv |
| 8. 仪表盘说明 | 启动命令 + URL |

---

## 📊 关键指标

| 指标 | 数值 |
|---|---|
| 上市公司数 | 2,842 |
| 总记录数 | 7,997 |
| 高风险公司数(Top100) | 100 |
| 异常检测公司数 | ~400(5%) |
| 舞弊识别模型 CV F1 | 0.762 |
| 模型 Recall | 0.747 |
| Top100 Recall@100 | ~30% |

---

## 🎓 技术栈

| 类别 | 工具 |
|---|---|
| 数据处理 | pandas, numpy |
| 机器学习 | scikit-learn, XGBoost, LightGBM |
| 可视化 | matplotlib, seaborn, plotly |
| 仪表盘 | Streamlit |
| 报告生成 | weasyprint, openpyxl, markdown |
| 数据源 | tushare(行业/地区) |

---

## 📚 文档

- **主报告**:`reports/审计数据分析大作业_何其轩_2023111180.pdf`(8 章)
- **技术总结**:`reports/审计数据分析技术总结与可行方案.md`
- **风险预警报告**:
  - `reports/三佳科技_600520_风险预警报告.md`
  - `reports/洲际油气_600759_风险预警报告.md`

---

## 📝 License

MIT License - 仅供学习交流使用。

---



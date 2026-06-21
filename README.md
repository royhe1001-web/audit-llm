# 上市公司财务舞弊智能识别与持续审计系统


本仓库为论文/报告的**复现代码**。本工作构建上市公司财务舞弊识别 + 持续审计框架,核心是 XGBoost + 6 条专家规则 + 异常检测的三件套融合。

## 关键结果

- **模型**:XGBoost 3.2(113 维特征 = 7 基础 + 101 fina_indicator + 5 pledge)
- **指标**:F1 = **0.872**、Recall = 0.866、Precision = 0.879、AUC = 0.85
- **样本**:8,724 条标注 + 7,997 条财务特征 + 137.9 万行股权质押
- **覆盖**:A 股 2,312 家公司 × 21 年(2005-2025)

## 仓库内容

| 目录 | 内容 |
|---|---|
| `scripts/` | 49 个 Python 脚本:数据拉取、特征工程、模型训练、规则引擎、消融实验、报告生成 |
| `app/` | Streamlit 仪表盘 + 单公司画像 + 风险预警 |
| `notebooks/` | Jupyter 实战笔记(公司名匹配) + 模型训练参考 |
| `build_doc.py` | Markdown → Word 转换工具 |

**不包含**:`data/`、`models/`、`output/`、`reports/`、PPT、年报 PDF — 均为运行时产物,见 `.gitignore`。

## 4 步复现

```bash
# 1. 配置 tushare token(需自行注册)
export TUSHARE_TOKEN=<your_token>

# 2. 拉取数据(约 30-60 分钟,分步可断点续传)
python scripts/pull_financials.py
python scripts/pull_fina_indicator.py
python scripts/pull_pledge_stat.py
python scripts/fetch_industry.py

# 3. 特征工程 + 模型训练
python scripts/build_intermediate_table.py
python scripts/train_xgb_v3_indicators.py

# 4. 风险评分 + 仪表盘
python scripts/risk_scoring.py
python scripts/audit_rules.py
python scripts/anomaly_detection.py
streamlit run app/dashboard.py
```

## 6 条审计规则(R1-R6)

| ID | 名称 | 严重度 | 触发条件 |
|:---:|---|:---:|---|
| R1 | 连续亏损 | 25 | roe<0 且 net_margin<0 |
| R2 | 现金流背离 | **30** | ocf_to_rev<0 且 net_margin>0 |
| R3 | 高负债 | 15 | debt_ratio>0.7 |
| R4 | 流动性紧张 | 20 | current_ratio<1 |
| R5 | 资产周转异常 | 10 | asset_turnover<0.3 或 >5 |
| R6 | 高杠杆亏损 | 20 | debt_ratio>0.5 且 roe<0 |

风险评分公式:`risk_score = 0.6 × ML 概率 + 0.4 × 规则严重度归一化`

## 环境要求

- Python ≥ 3.10
- 主要依赖:`pandas`、`numpy`、`scikit-learn==1.8.0`、`xgboost==3.2`、`tushare`、`streamlit==1.58`、`plotly==6.7`
- 详见各脚本的 `import` 段

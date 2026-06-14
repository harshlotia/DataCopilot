# DataCopilot

**AI-powered data analytics platform built on Databricks, Spark, and Claude AI.**

Upload any dataset, ask questions in plain English, and get instant SQL-powered charts — with automated data quality monitoring included.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Railway-blueviolet?style=for-the-badge&logo=railway)](https://web-production-3bacd.up.railway.app/)

---

## What it does

### Ask Questions in Plain English
Type a question like *"What are the top 5 categories by revenue?"* and the app:
- Sends your question + database schema to Claude AI
- Claude generates valid SQL
- SQL runs instantly against your data
- Results appear as a table and chart with a plain-English insight

### Automatic Data Profiling
Load any CSV and instantly see:
- Null rates per column (color-coded heatmap)
- Duplicate row count
- Data types, unique values, min/max/mean
- Distribution histograms

### Anomaly Detection
Automatically scans every table and flags:
- Columns with high null rates (>20%)
- Duplicate rows (>5%)
- Statistical outliers (IQR method)
- Zero-variance columns

Each anomaly gets an AI-generated explanation of what caused it and what to do.

### Dynamic Suggested Questions
When you load a dataset, Claude reads the actual schema and generates 5 relevant questions specific to your data — not generic hardcoded ones.

---

## Architecture

```
Your Data (CSV upload or built-in demo)
           │
           ▼
  Databricks + Apache Spark          ← Heavy ETL, Delta Lake storage
  (databricks_notebooks/)            ← Run in Databricks Community Edition
           │
           ▼
        DuckDB                       ← In-process SQL for the live app
           │                            (instant queries, no cluster wait)
           ▼
     Streamlit App
    ┌──────────────────────────────┐
    │  Tab 1: Data Overview        │  ← Profiling, distributions, nulls
    │  Tab 2: Ask Questions        │  ← NL → Claude → SQL → Chart
    │  Tab 3: Anomaly Report       │  ← Auto-detection + AI explanations
    └──────────────────────────────┘
           │
           ▼
    Railway (live URL — always on)
```

**Why two layers?** Databricks clusters take minutes to start — not suitable for a live demo. Databricks handles the heavy Spark/Delta work; DuckDB handles instant queries in the deployed app.

---

## Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Anthropic API key
cp .env.example .env
# Open .env and paste your key from console.anthropic.com

# 3. Start the app
streamlit run app/main.py
```

---

## Databricks Notebooks

Run these in order inside **Databricks Community Edition** to see the full Spark + Delta Lake pipeline:

| Notebook | What it does |
|---|---|
| `01_ingest_ecommerce.py` | Generate synthetic e-commerce data, write to Delta Lake |
| `02_spark_profiling.py` | Spark-powered data profiling, SQL analytics at scale |
| `03_anomaly_detection.py` | Distributed anomaly detection using PySpark |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Big Data Processing | Apache Spark / PySpark |
| Data Storage | Delta Lake (ACID, time travel) |
| Notebook Environment | Databricks Community Edition |
| In-App Query Engine | DuckDB |
| AI / NL→SQL | Claude API (claude-sonnet-4-6) |
| Frontend | Streamlit |
| Charts | Plotly |

---

## Project Structure

```
DataCopilot/
├── app/
│   ├── main.py               # Streamlit app — 3 tabs
│   ├── nl_to_sql.py          # Claude API: NL→SQL + suggestions + insights
│   ├── anomaly_detector.py   # IQR, null, duplicate detection
│   ├── data_profiler.py      # Column-level profiling
│   ├── db_manager.py         # DuckDB query layer
│   └── demo_data.py          # Synthetic e-commerce dataset generator
├── databricks_notebooks/
│   ├── 01_ingest_ecommerce.py
│   ├── 02_spark_profiling.py
│   └── 03_anomaly_detection.py
└── requirements.txt
```

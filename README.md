# DataCopilot — AI-Powered Analytics on Databricks

> Natural language querying + automated anomaly detection, powered by Spark, Delta Lake, and Claude AI.

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## What it does

| Feature | Tech |
|---|---|
| Ingest CSVs into Delta Lake | Apache Spark + Databricks |
| Profile data quality | PySpark + DuckDB |
| Ask questions in plain English | Claude AI → SQL → DuckDB |
| Detect anomalies automatically | IQR, null analysis, duplicate detection |
| AI explanations for each anomaly | Claude API |

## Architecture

```
CSV / Demo Data
      ↓
Databricks (Spark ETL → Delta Lake)   ← notebooks/
      ↓
DuckDB (in-process query layer)        ← fast, no cluster needed for demo
      ↓
Streamlit App                          ← app/
  ├── Data Overview (profiling)
  ├── Ask Questions (NL → SQL via Claude)
  └── Anomaly Report (AI-explained issues)
      ↓
Deploy: Streamlit Cloud (free)
```

## Quick Start

```bash
cd DataCopilot
pip install -r requirements.txt

# Add your Anthropic API key
cp .env.example .env
# Edit .env with your key from console.anthropic.com

# Run the app
streamlit run app/main.py
```

## Databricks Notebooks

Run in order inside Databricks Community Edition:

1. `databricks_notebooks/01_ingest_ecommerce.py` — Generate data + write to Delta Lake
2. `databricks_notebooks/02_spark_profiling.py` — Spark-powered data profiling + SQL analytics
3. `databricks_notebooks/03_anomaly_detection.py` — Scale anomaly detection with Spark

## Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app/main.py`
5. Add `ANTHROPIC_API_KEY` in Secrets
6. Deploy

## Resume Bullet

> Built DataCopilot, a Databricks-based AI analytics platform using Apache Spark, Delta Lake, and Claude AI, enabling natural language querying over large-scale e-commerce datasets and automated anomaly detection with AI-generated explanations; deployed live on Streamlit Cloud.

## Tech Stack

- **Databricks Community Edition** — cluster management, notebook environment
- **Apache Spark / PySpark** — distributed ETL and profiling
- **Delta Lake** — ACID transactions, time travel, schema evolution
- **DuckDB** — in-process SQL for the deployed app
- **Claude API (claude-sonnet-4-6)** — NL→SQL generation and anomaly explanation
- **Streamlit** — frontend and deployment
- **Plotly** — interactive charts

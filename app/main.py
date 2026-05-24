import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from anomaly_detector import AnomalyDetector
from data_profiler import profile_table
from db_manager import DBManager
from demo_data import generate_demo_tables
from nl_to_sql import NLToSQL

load_dotenv()

st.set_page_config(
    page_title="DataCopilot",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🔍</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state init ──────────────────────────────────────────────────────
def _init_state():
    if "db" not in st.session_state:
        st.session_state.db = DBManager()
    if "tables_loaded" not in st.session_state:
        st.session_state.tables_loaded = False
    if "nl_history" not in st.session_state:
        st.session_state.nl_history = []


# ── Sidebar ─────────────────────────────────────────────────────────────────
def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## DataCopilot")
        st.caption("Spark · Delta Lake · Claude AI")
        st.divider()

        st.subheader("Data Source")

        source = st.radio("Choose", ["Demo E-Commerce Dataset", "Upload your own CSV(s)"], label_visibility="collapsed")

        if source == "Demo E-Commerce Dataset":
            if st.button("Load Demo Dataset", use_container_width=True):
                with st.spinner("Generating 5,000 synthetic orders..."):
                    tables = generate_demo_tables()
                    for name, df in tables.items():
                        st.session_state.db.register_dataframe(name, df)
                    st.session_state.tables_loaded = True
                st.success("Loaded: customers, products, orders, order_items")
        else:
            files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
            if files:
                for f in files:
                    df = pd.read_csv(f)
                    name = Path(f.name).stem
                    st.session_state.db.register_dataframe(name, df)
                    st.session_state.tables_loaded = True
                st.success(f"Loaded {len(files)} table(s)")

        if st.session_state.tables_loaded:
            st.divider()
            st.subheader("Loaded Tables")
            for t in st.session_state.db.get_tables():
                count = st.session_state.db.get_row_count(t)
                st.write(f"**{t}** — {count:,} rows")

        st.divider()
        st.caption("Built with Databricks · Spark · Delta Lake · Claude API")

    return os.getenv("ANTHROPIC_API_KEY", "")


# ── Tab 1: Data Overview ────────────────────────────────────────────────────
def _tab_overview():
    st.header("Data Overview")
    tables = st.session_state.db.get_tables()
    selected = st.selectbox("Select table", tables)
    if not selected:
        return

    df = st.session_state.db.get_full_table(selected)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Null cells", f"{int(df.isnull().sum().sum()):,}")
    col4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.subheader("Column Profile")
    profile = profile_table(df)
    st.dataframe(
        profile.style.background_gradient(subset=["Null %"], cmap="YlOrRd", vmin=0, vmax=100),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    # Quick distribution charts for numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        st.subheader("Distributions")
        cols_to_show = numeric_cols[:4]
        chart_cols = st.columns(len(cols_to_show))
        for i, col in enumerate(cols_to_show):
            fig = px.histogram(df, x=col, nbins=30, title=col, height=250)
            fig.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
            chart_cols[i].plotly_chart(fig, use_container_width=True)


# ── Tab 2: NL → SQL ─────────────────────────────────────────────────────────
def _tab_nl_sql(api_key: str):
    st.header("Ask Questions (Natural Language → SQL)")

    if not api_key:
        st.warning("Enter your Anthropic API key in the sidebar to use this feature.")
        return

    # Suggested questions
    st.caption("Try one of these or type your own:")
    suggestions = [
        "What are the top 5 product categories by total revenue?",
        "Show monthly order count trends for 2024",
        "Which states have the most customers?",
        "What percentage of orders were cancelled by category?",
        "Who are the top 10 customers by total spending?",
    ]
    cols = st.columns(3)
    prefill = ""
    for i, s in enumerate(suggestions[:6]):
        if cols[i % 3].button(s, key=f"sug_{i}", use_container_width=True):
            prefill = s

    question = st.text_input("Your question", value=prefill, placeholder="e.g. What was average order value last month?")

    if st.button("Run", type="primary", disabled=not question):
        nl = NLToSQL(api_key)
        schema = st.session_state.db.get_schema()

        with st.spinner("Generating SQL..."):
            try:
                sql = nl.generate_sql(question, schema)
            except Exception as e:
                st.error(f"SQL generation failed: {e}")
                return

        st.code(sql, language="sql")

        with st.spinner("Running query..."):
            try:
                result_df = st.session_state.db.execute_query(sql)
            except Exception as e:
                st.error(f"Query execution failed: {e}")
                return

        st.dataframe(result_df, use_container_width=True)

        # Auto-chart
        _auto_chart(result_df)

        # AI insight
        summary = result_df.head(5).to_string(index=False)
        with st.spinner("Generating insight..."):
            try:
                insight = nl.explain_result(question, sql, summary)
                st.info(f"AI Insight: {insight}")
            except Exception:
                pass

        st.session_state.nl_history.append({"question": question, "sql": sql, "rows": len(result_df)})

    # History
    if st.session_state.nl_history:
        with st.expander("Query History"):
            for item in reversed(st.session_state.nl_history[-10:]):
                st.write(f"**Q:** {item['question']} — {item['rows']} rows")
                st.code(item["sql"], language="sql")


def _auto_chart(df: pd.DataFrame):
    if df.empty or len(df.columns) < 2:
        return
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    if not num_cols:
        return

    y_col = num_cols[0]
    x_col = cat_cols[0] if cat_cols else df.columns[0]

    if len(df) <= 30:
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", height=350)
    else:
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}", height=350, markers=True)

    fig.update_layout(margin=dict(t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: Anomaly Report ────────────────────────────────────────────────────
def _tab_anomalies(api_key: str):
    st.header("Anomaly Report")

    tables = st.session_state.db.get_tables()
    detector = AnomalyDetector()

    run_ai = api_key != ""
    nl = NLToSQL(api_key) if run_ai else None

    all_anomalies = []
    for table in tables:
        df = st.session_state.db.get_full_table(table)
        anomalies = detector.detect(table, df)
        all_anomalies.extend(anomalies)

    if not all_anomalies:
        st.success("No anomalies detected across all tables.")
        return

    severity_order = {"high": 0, "medium": 1, "low": 2}
    all_anomalies.sort(key=lambda a: severity_order[a.severity])

    high = [a for a in all_anomalies if a.severity == "high"]
    medium = [a for a in all_anomalies if a.severity == "medium"]
    low = [a for a in all_anomalies if a.severity == "low"]

    col1, col2, col3 = st.columns(3)
    col1.metric("High Severity", len(high), delta=None)
    col2.metric("Medium Severity", len(medium))
    col3.metric("Low Severity", len(low))

    _render_anomaly_group("High Severity", high, "red", nl)
    _render_anomaly_group("Medium Severity", medium, "orange", nl)
    _render_anomaly_group("Low Severity", low, "blue", nl)


def _render_anomaly_group(label: str, anomalies: list, color: str, nl):
    if not anomalies:
        return
    st.subheader(label)
    for a in anomalies:
        with st.expander(f"[{a.table}] {a.column} — {a.anomaly_type}"):
            st.write(f"**Details:** {a.description}")
            if nl:
                with st.spinner("Getting AI explanation..."):
                    try:
                        explanation = nl.explain_anomaly(
                            f"Table: {a.table}, Column: {a.column}, Issue: {a.anomaly_type}, Detail: {a.description}"
                        )
                        st.info(f"AI: {explanation}")
                    except Exception:
                        pass


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    _init_state()
    api_key = _render_sidebar()

    st.title("DataCopilot")
    st.caption("AI-Powered Analytics · Databricks + Spark + Delta Lake + Claude")

    if not st.session_state.tables_loaded:
        st.info("Load a dataset from the sidebar to get started.")
        st.markdown(
            """
**What this does:**
- Ingest CSV data (or use the built-in e-commerce demo)
- Profile data quality automatically (nulls, duplicates, distributions)
- Ask questions in plain English — Claude generates and runs the SQL
- Detect data anomalies with AI-generated explanations

**Backed by:** Databricks notebooks for Spark ETL + Delta Lake storage *(see `/databricks_notebooks/`)*
"""
        )
        return

    tab1, tab2, tab3 = st.tabs(["Data Overview", "Ask Questions", "Anomaly Report"])
    with tab1:
        _tab_overview()
    with tab2:
        _tab_nl_sql(api_key)
    with tab3:
        _tab_anomalies(api_key)


if __name__ == "__main__":
    main()

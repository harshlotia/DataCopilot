import os
import re
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from anomaly_detector import AnomalyDetector
from data_profiler import profile_table
from db_manager import DBManager
from demo_data import generate_demo_tables, generate_stock_tables, generate_hospital_tables, generate_hr_tables
from nl_to_sql import NLToSQL

load_dotenv()

st.set_page_config(
    page_title="DataCopilot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1e293b;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f1f5f9;
    padding: 4px;
    border-radius: 10px;
    border-bottom: none;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 24px;
    font-weight: 500;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1e293b !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e5e7eb;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    margin-bottom: 8px;
}

/* Code blocks */
.stCodeBlock {
    border-radius: 8px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
}

/* Info / success boxes */
.stAlert {
    border-radius: 10px;
}

/* Question suggestion buttons */
div[data-testid="column"] .stButton > button {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #374151;
    font-size: 0.85rem;
    text-align: left;
    height: auto;
    white-space: normal;
    padding: 10px 14px;
    line-height: 1.4;
}
div[data-testid="column"] .stButton > button:hover {
    background: #eef2ff;
    border-color: #6366f1;
    color: #4f46e5;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "db": DBManager(),
        "tables_loaded": False,
        "nl_history": [],
        "nl_question": "",
        "nl_auto_run": False,
        "nl_last_sql": "",
        "nl_last_result": None,
        "nl_last_insight": "",
        "nl_suggestions": [],
        "nl_suggestions_schema": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── File loader (multi-format) ────────────────────────────────────────────────
def _load_file(f) -> dict[str, pd.DataFrame]:
    name = Path(f.name).stem.replace(" ", "_").replace("-", "_")
    ext = Path(f.name).suffix.lower()
    tables = {}

    if ext == ".csv":
        tables[name] = pd.read_csv(f)
    elif ext in (".xlsx", ".xls"):
        xls = pd.ExcelFile(f)
        for sheet in xls.sheet_names:
            sheet_name = sheet.replace(" ", "_").replace("-", "_")
            tables[f"{name}_{sheet_name}"] = xls.parse(sheet)
    elif ext == ".parquet":
        tables[name] = pd.read_parquet(f)
    elif ext in (".db", ".sqlite", ".sqlite3"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        conn = sqlite3.connect(tmp_path)
        sheet_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
        for t in sheet_names:
            tables[t] = pd.read_sql(f"SELECT * FROM {t}", conn)
        conn.close()
        os.unlink(tmp_path)

    return tables


# ── Smart KPI cards ───────────────────────────────────────────────────────────
def _render_kpis():
    tables = st.session_state.db.get_tables()
    if not tables:
        return

    kpis = []

    for table in tables:
        df = st.session_state.db.get_full_table(table)

        # Total rows
        kpis.append(("Total Rows", f"{len(df):,}", f"across {len(tables)} table(s)"))

        # Revenue / amount columns
        for col in df.select_dtypes(include="number").columns:
            col_lower = col.lower()
            if any(k in col_lower for k in ["revenue", "amount", "total", "sales", "price", "spend"]):
                val = df[col].sum()
                label = col.replace("_", " ").title()
                kpis.append((f"Total {label}", f"${val:,.0f}" if val > 100 else f"{val:,.2f}", table))
                break

        # Date range
        for col in df.columns:
            if df[col].dtype in ["datetime64[ns]", "datetime64[us]"] or "date" in col.lower():
                try:
                    dates = pd.to_datetime(df[col], errors="coerce").dropna()
                    if len(dates) > 0:
                        kpis.append(("Date Range", f"{dates.min().strftime('%b %Y')} – {dates.max().strftime('%b %Y')}", col))
                        break
                except Exception:
                    pass

        # Unique customers / users
        for col in df.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in ["customer", "user", "client", "member"]) and "id" in col_lower:
                kpis.append((f"Unique {col.replace('_', ' ').title()}", f"{df[col].nunique():,}", table))
                break

    # Deduplicate and cap at 4
    seen = set()
    unique_kpis = []
    for k in kpis:
        if k[0] not in seen:
            seen.add(k[0])
            unique_kpis.append(k)
    unique_kpis = unique_kpis[:4]

    if unique_kpis:
        cols = st.columns(len(unique_kpis))
        for i, (label, value, hint) in enumerate(unique_kpis):
            cols[i].metric(label, value, hint)
        st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## DataCopilot")
        st.divider()

        st.markdown("**Demo Datasets**")
        demo_choice = st.selectbox(
            "demo",
            ["E-Commerce", "Stock Market", "Hospital / Healthcare", "HR / Employees"],
            label_visibility="collapsed",
        )
        demo_map = {
            "E-Commerce":           ("5,000 orders · 4 tables",   generate_demo_tables),
            "Stock Market":         ("12 tickers · 2 years daily", generate_stock_tables),
            "Hospital / Healthcare":("8,000 visits · 4 tables",   generate_hospital_tables),
            "HR / Employees":       ("1,500 employees · 3 tables", generate_hr_tables),
        }
        hint, loader = demo_map[demo_choice]
        st.caption(hint)
        if st.button("Load Dataset", use_container_width=True):
            with st.spinner(f"Generating {demo_choice} data..."):
                st.session_state.db = DBManager()
                tables = loader()
                for name, df in tables.items():
                    st.session_state.db.register_dataframe(name, df)
                st.session_state.tables_loaded = True
                st.session_state.nl_suggestions = []
                st.session_state.nl_suggestions_schema = ""
                st.session_state.nl_last_sql = ""
                st.session_state.nl_last_result = None
                st.session_state.nl_last_insight = ""
            st.success(f"Loaded {len(tables)} tables")

        st.divider()
        st.markdown("**Upload your own file**")
        st.caption("CSV · Excel · Parquet · SQLite")
        files = st.file_uploader(
            "Upload files",
            type=["csv", "xlsx", "xls", "parquet", "db", "sqlite", "sqlite3"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if files:
            for f in files:
                loaded = _load_file(f)
                for name, df in loaded.items():
                    st.session_state.db.register_dataframe(name, df)
            st.session_state.tables_loaded = True
            st.success(f"Loaded {len(files)} file(s)")

        if st.session_state.tables_loaded:
            st.divider()
            st.markdown("**Loaded Tables**")
            for t in st.session_state.db.get_tables():
                count = st.session_state.db.get_row_count(t)
                st.markdown(f"<small>📋 **{t}** — {count:,} rows</small>", unsafe_allow_html=True)


    return os.getenv("ANTHROPIC_API_KEY", "")


# ── Landing page ──────────────────────────────────────────────────────────────
def _landing():
    st.markdown("""
<div style='padding: 2rem 0 1rem 0;'>
    <h1 style='font-size: 2.4rem; font-weight: 800; color: #1e293b; margin-bottom: 0.5rem;'>
        Ask your data anything.
    </h1>
    <p style='font-size: 1.1rem; color: #64748b; max-width: 600px; line-height: 1.6;'>
        Upload any dataset and ask questions in plain English.
        DataCopilot writes the SQL, runs it, and explains the result — instantly.
    </p>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:20px;'>
    <div style='font-size:1.5rem; margin-bottom:8px;'>💬</div>
    <div style='font-weight:600; color:#1e293b; margin-bottom:6px;'>Natural Language → SQL</div>
    <div style='font-size:0.85rem; color:#64748b;'>Ask "What are my top customers by revenue?" and get a chart instantly.</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:20px;'>
    <div style='font-size:1.5rem; margin-bottom:8px;'>🔍</div>
    <div style='font-weight:600; color:#1e293b; margin-bottom:6px;'>Auto Data Profiling</div>
    <div style='font-size:0.85rem; color:#64748b;'>Instant quality report — nulls, duplicates, distributions, outliers.</div>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:20px;'>
    <div style='font-size:1.5rem; margin-bottom:8px;'>⚠️</div>
    <div style='font-weight:600; color:#1e293b; margin-bottom:6px;'>Anomaly Detection</div>
    <div style='font-size:0.85rem; color:#64748b;'>AI finds and explains data quality issues before they cause problems.</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div style='background:#eef2ff; border:1px solid #c7d2fe; border-radius:12px; padding:20px 24px;'>
    <div style='font-weight:600; color:#4338ca; margin-bottom:12px;'>Supported formats</div>
    <div style='display:flex; gap:10px; flex-wrap:wrap;'>
        <span style='background:white; border:1px solid #c7d2fe; color:#4338ca; padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight:500;'>CSV</span>
        <span style='background:white; border:1px solid #c7d2fe; color:#4338ca; padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight:500;'>Excel (.xlsx)</span>
        <span style='background:white; border:1px solid #c7d2fe; color:#4338ca; padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight:500;'>Parquet</span>
        <span style='background:white; border:1px solid #c7d2fe; color:#4338ca; padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight:500;'>SQLite</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Tab 1: Data Overview ──────────────────────────────────────────────────────
def _tab_overview():
    tables = st.session_state.db.get_tables()
    selected = st.selectbox("Select table", tables)
    if not selected:
        return

    df = st.session_state.db.get_full_table(selected)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Null cells", f"{int(df.isnull().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Column Profile**")
    profile = profile_table(df)
    st.dataframe(
        profile.style.background_gradient(subset=["Null %"], cmap="YlOrRd", vmin=0, vmax=100),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Sample Data**")
    st.dataframe(df.head(10), use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Distributions**")
        cols_to_show = numeric_cols[:4]
        chart_cols = st.columns(len(cols_to_show))
        for i, col in enumerate(cols_to_show):
            fig = px.histogram(df, x=col, nbins=30, title=col, height=240,
                               color_discrete_sequence=["#6366f1"])
            fig.update_layout(showlegend=False, margin=dict(t=36, b=10, l=10, r=10),
                              plot_bgcolor="white", paper_bgcolor="white")
            chart_cols[i].plotly_chart(fig, use_container_width=True)


# ── Tab 2: Ask Questions ──────────────────────────────────────────────────────
def _tab_nl_sql(api_key: str):
    if not api_key:
        st.warning("No Anthropic API key configured.")
        return

    schema = st.session_state.db.get_schema()
    if schema != st.session_state.nl_suggestions_schema:
        with st.spinner("Generating suggested questions for your data..."):
            try:
                st.session_state.nl_suggestions = NLToSQL(api_key).generate_suggestions(schema)
                st.session_state.nl_suggestions_schema = schema
            except Exception:
                st.session_state.nl_suggestions = []

    suggestions = st.session_state.nl_suggestions
    if suggestions:
        st.markdown("<small style='color:#64748b; font-weight:500;'>SUGGESTED QUESTIONS</small>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, s in enumerate(suggestions):
            if cols[i % 3].button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state.nl_question = s
                st.session_state.nl_auto_run = True

    st.markdown("<br>", unsafe_allow_html=True)
    typed = st.text_input(
        "Your question",
        value=st.session_state.nl_question,
        placeholder="e.g. Which department had the highest spend last quarter?",
    )
    if typed != st.session_state.nl_question:
        st.session_state.nl_question = typed
        st.session_state.nl_last_sql = ""
        st.session_state.nl_last_result = None
        st.session_state.nl_last_insight = ""

    run_clicked = st.button("Run", type="primary", disabled=not st.session_state.nl_question)

    if (run_clicked or st.session_state.nl_auto_run) and st.session_state.nl_question:
        st.session_state.nl_auto_run = False
        question = st.session_state.nl_question
        nl = NLToSQL(api_key)
        schema = st.session_state.db.get_schema()

        with st.spinner("Writing SQL..."):
            try:
                sql = nl.generate_sql(question, schema)
                st.session_state.nl_last_sql = sql
            except Exception as e:
                st.error(f"SQL generation failed: {e}")
                return

        with st.spinner("Running query..."):
            try:
                result_df = st.session_state.db.execute_query(sql)
                st.session_state.nl_last_result = result_df
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.session_state.nl_last_result = None
                return

        with st.spinner("Generating insight..."):
            try:
                summary = result_df.head(5).to_string(index=False)
                st.session_state.nl_last_insight = nl.explain_result(question, sql, summary)
            except Exception:
                st.session_state.nl_last_insight = ""

        st.session_state.nl_history.append({
            "question": question, "sql": sql, "rows": len(result_df)
        })

    if st.session_state.nl_last_sql:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Generated SQL", expanded=True):
            st.code(st.session_state.nl_last_sql, language="sql")

    if st.session_state.nl_last_result is not None:
        result_df = st.session_state.nl_last_result
        st.dataframe(result_df, use_container_width=True)
        _auto_chart(result_df)

    if st.session_state.nl_last_insight:
        clean = re.sub(r"^#+\s*.+\n?", "", st.session_state.nl_last_insight, flags=re.MULTILINE)
        clean = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", clean).replace("`", "").strip()
        if clean:
            st.markdown(
                f"<p style='font-size:0.9rem; color:#475569; margin-top:8px;'>💡 {clean}</p>",
                unsafe_allow_html=True,
            )

    if st.session_state.nl_history:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Query History"):
            for item in reversed(st.session_state.nl_history[-10:]):
                st.markdown(f"**Q:** {item['question']} — {item['rows']} rows")
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
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}",
                     height=360, color_discrete_sequence=["#6366f1"])
    else:
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}",
                      height=360, markers=True, color_discrete_sequence=["#6366f1"])

    fig.update_layout(
        margin=dict(t=50, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: Anomaly Report ─────────────────────────────────────────────────────
def _tab_anomalies(api_key: str):
    tables = st.session_state.db.get_tables()
    detector = AnomalyDetector()
    nl = NLToSQL(api_key) if api_key else None

    all_anomalies = []
    for table in tables:
        df = st.session_state.db.get_full_table(table)
        all_anomalies.extend(detector.detect(table, df))

    if not all_anomalies:
        st.success("No anomalies detected across all tables.")
        return

    all_anomalies.sort(key=lambda a: {"high": 0, "medium": 1, "low": 2}[a.severity])
    high = [a for a in all_anomalies if a.severity == "high"]
    medium = [a for a in all_anomalies if a.severity == "medium"]
    low = [a for a in all_anomalies if a.severity == "low"]

    c1, c2, c3 = st.columns(3)
    c1.metric("High Severity", len(high))
    c2.metric("Medium Severity", len(medium))
    c3.metric("Low Severity", len(low))

    st.markdown("<br>", unsafe_allow_html=True)
    for label, group, color in [("High Severity", high, "🔴"), ("Medium Severity", medium, "🟡"), ("Low Severity", low, "🔵")]:
        if not group:
            continue
        st.markdown(f"**{color} {label}**")
        for a in group:
            with st.expander(f"{a.table} · {a.column} — {a.anomaly_type}"):
                st.markdown(f"**{a.description}**")
                if nl:
                    with st.spinner("AI explanation..."):
                        try:
                            explanation = nl.explain_anomaly(
                                f"Table: {a.table}, Column: {a.column}, Issue: {a.anomaly_type}, Detail: {a.description}"
                            )
                            clean = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", explanation).replace("`", "").strip()
                            st.markdown(f"<p style='font-size:0.9rem; color:#475569;'>💡 {clean}</p>", unsafe_allow_html=True)
                        except Exception:
                            pass


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    _init_state()
    api_key = _render_sidebar()

    if not st.session_state.tables_loaded:
        _landing()
        return

    st.markdown(
        "<h2 style='font-size:1.6rem; font-weight:700; color:#1e293b; margin-bottom:4px;'>DataCopilot</h2>",
        unsafe_allow_html=True,
    )

    _render_kpis()

    tab1, tab2, tab3 = st.tabs(["  Data Overview  ", "  Ask Questions  ", "  Anomaly Report  "])
    with tab1:
        _tab_overview()
    with tab2:
        _tab_nl_sql(api_key)
    with tab3:
        _tab_anomalies(api_key)


if __name__ == "__main__":
    main()

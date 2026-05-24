# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Spark-Powered Data Profiling
# MAGIC
# MAGIC **Purpose:** Use Apache Spark to profile each Delta table at scale.
# MAGIC Generates null rates, cardinality, value distributions, and summary statistics.
# MAGIC
# MAGIC Run after `01_ingest_ecommerce.py`.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType
import json

DELTA_BASE = "dbfs:/user/hive/warehouse/datacopilot"
TABLES = ["customers", "products", "orders", "order_items"]


def profile_spark_table(table_name: str):
    df = spark.read.format("delta").load(f"{DELTA_BASE}/{table_name}")
    total_rows = df.count()
    print(f"\n{'='*60}")
    print(f"Table: {table_name} | Rows: {total_rows:,} | Cols: {len(df.columns)}")
    print(f"{'='*60}")

    # Null rates per column
    null_exprs = [
        F.round(F.sum(F.col(c).isNull().cast("int")) / F.lit(total_rows) * 100, 2).alias(c)
        for c in df.columns
    ]
    null_df = df.select(null_exprs)
    print("\nNull rates (%):")
    null_df.show(truncate=False)

    # Distinct counts
    distinct_exprs = [F.countDistinct(c).alias(c) for c in df.columns]
    distinct_df = df.select(distinct_exprs)
    print("Distinct value counts:")
    distinct_df.show(truncate=False)

    # Numeric summary stats
    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    if numeric_cols:
        print("Numeric column statistics:")
        df.select(numeric_cols).describe().show(truncate=False)

    return null_df, distinct_df


# COMMAND ----------

for table in TABLES:
    profile_spark_table(table)

# COMMAND ----------

# MAGIC %md ## Revenue Analytics with Spark SQL

# COMMAND ----------

spark.sql("""
SELECT
    oi.category,
    COUNT(DISTINCT o.order_id) AS total_orders,
    SUM(oi.line_total) AS total_revenue,
    AVG(oi.line_total) AS avg_order_value,
    ROUND(SUM(oi.line_total) / SUM(SUM(oi.line_total)) OVER () * 100, 2) AS revenue_share_pct
FROM datacopilot.orders o
JOIN datacopilot.order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'delivered'
GROUP BY oi.category
ORDER BY total_revenue DESC
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md ## Monthly Order Trend

# COMMAND ----------

spark.sql("""
SELECT
    DATE_TRUNC('month', CAST(order_date AS DATE)) AS month,
    COUNT(*) AS total_orders,
    SUM(total_amount) AS monthly_revenue,
    AVG(total_amount) AS avg_order_value
FROM datacopilot.orders
GROUP BY 1
ORDER BY 1
""").show(30, truncate=False)

# COMMAND ----------

# MAGIC %md ## Customer Segmentation by State + Loyalty

# COMMAND ----------

spark.sql("""
SELECT
    c.state,
    c.loyalty_tier,
    COUNT(DISTINCT c.customer_id) AS customer_count,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(AVG(o.total_amount), 2) AS avg_order_value
FROM datacopilot.customers c
LEFT JOIN datacopilot.orders o ON c.customer_id = o.customer_id
GROUP BY c.state, c.loyalty_tier
ORDER BY customer_count DESC
LIMIT 20
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Profiling Results to Delta
# MAGIC
# MAGIC Export profiling results so the Streamlit app can display Databricks-computed stats.

# COMMAND ----------

revenue_by_category = spark.sql("""
SELECT oi.category, SUM(oi.line_total) AS total_revenue, COUNT(*) AS total_orders
FROM datacopilot.orders o
JOIN datacopilot.order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'delivered'
GROUP BY oi.category ORDER BY total_revenue DESC
""")

revenue_by_category.write.format("delta").mode("overwrite").save(f"{DELTA_BASE}/profiling/revenue_by_category")
print("Profiling results saved to Delta Lake.")

# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Spark Anomaly Detection
# MAGIC
# MAGIC **Purpose:** Detect data quality issues at scale using Spark.
# MAGIC - Null spike detection
# MAGIC - Duplicate row detection
# MAGIC - Statistical outlier detection (IQR method)
# MAGIC - Schema drift simulation
# MAGIC
# MAGIC Run after `01_ingest_ecommerce.py`.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType

DELTA_BASE = "dbfs:/user/hive/warehouse/datacopilot"
TABLES = ["customers", "products", "orders", "order_items"]

anomaly_report = []

# COMMAND ----------

# MAGIC %md ## 1. Null Rate Detection

# COMMAND ----------

for table_name in TABLES:
    df = spark.read.format("delta").load(f"{DELTA_BASE}/{table_name}")
    total = df.count()

    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull()).count()
        null_pct = round(null_count / total * 100, 2)
        if null_pct >= 20:
            severity = "HIGH" if null_pct >= 50 else "MEDIUM"
            anomaly_report.append({
                "table": table_name,
                "column": col_name,
                "anomaly_type": "High Null Rate",
                "severity": severity,
                "value_pct": null_pct,
                "description": f"{null_pct}% of {total:,} rows have null values"
            })
            print(f"[{severity}] {table_name}.{col_name}: {null_pct}% null")

print(f"\nNull check complete. Flagged: {len([a for a in anomaly_report if a['anomaly_type'] == 'High Null Rate'])} columns")

# COMMAND ----------

# MAGIC %md ## 2. Duplicate Row Detection

# COMMAND ----------

for table_name in TABLES:
    df = spark.read.format("delta").load(f"{DELTA_BASE}/{table_name}")
    total = df.count()
    distinct = df.distinct().count()
    dup_count = total - distinct
    dup_pct = round(dup_count / total * 100, 2)

    if dup_pct >= 5:
        anomaly_report.append({
            "table": table_name,
            "column": "(entire row)",
            "anomaly_type": "Duplicate Rows",
            "severity": "MEDIUM",
            "value_pct": dup_pct,
            "description": f"{dup_count:,} duplicate rows ({dup_pct}% of {total:,})"
        })
        print(f"[MEDIUM] {table_name}: {dup_pct}% duplicate rows")
    else:
        print(f"[OK] {table_name}: {dup_pct}% duplicates (below threshold)")

# COMMAND ----------

# MAGIC %md ## 3. Outlier Detection (IQR Method) via Spark

# COMMAND ----------

IQR_MULTIPLIER = 3.0

for table_name in TABLES:
    df = spark.read.format("delta").load(f"{DELTA_BASE}/{table_name}")
    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]

    for col_name in numeric_cols:
        quants = df.select(
            F.percentile_approx(col_name, 0.25).alias("q1"),
            F.percentile_approx(col_name, 0.75).alias("q3"),
            F.count(col_name).alias("total_non_null")
        ).collect()[0]

        q1, q3, total_non_null = quants["q1"], quants["q3"], quants["total_non_null"]
        if q1 is None or q3 is None or total_non_null < 20:
            continue

        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr

        outlier_count = df.filter(
            (F.col(col_name) < lower) | (F.col(col_name) > upper)
        ).count()
        outlier_pct = round(outlier_count / total_non_null * 100, 2)

        if outlier_pct >= 1:
            anomaly_report.append({
                "table": table_name,
                "column": col_name,
                "anomaly_type": "Outliers Detected",
                "severity": "LOW",
                "value_pct": outlier_pct,
                "description": f"{outlier_pct}% outliers. Expected range: [{lower:.2f}, {upper:.2f}]"
            })
            print(f"[LOW] {table_name}.{col_name}: {outlier_pct}% outliers")

# COMMAND ----------

# MAGIC %md ## 4. Full Anomaly Report

# COMMAND ----------

import pandas as pd

report_df = pd.DataFrame(anomaly_report)
if not report_df.empty:
    print(f"\nTotal anomalies detected: {len(report_df)}")
    print(report_df.sort_values("severity").to_string(index=False))
else:
    print("No anomalies detected.")

# COMMAND ----------

# MAGIC %md ## 5. Save Anomaly Report to Delta

# COMMAND ----------

if anomaly_report:
    report_spark_df = spark.createDataFrame(pd.DataFrame(anomaly_report))
    report_spark_df.write.format("delta").mode("overwrite").save(f"{DELTA_BASE}/anomaly_report")
    print(f"Anomaly report saved: {len(anomaly_report)} anomalies → {DELTA_BASE}/anomaly_report")

# COMMAND ----------

# MAGIC %md ## 6. Schema Drift Simulation
# MAGIC
# MAGIC Simulate detecting a schema change between two versions of the orders table.

# COMMAND ----------

# Simulate a "new batch" with an extra column — schema drift
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

original_schema = spark.read.format("delta").load(f"{DELTA_BASE}/orders").schema
original_cols = set(f.name for f in original_schema.fields)

# Simulate incoming batch with added column
new_batch_cols = original_cols | {"discount_code", "referral_source"}  # pretend these appeared

added = new_batch_cols - original_cols
removed = original_cols - new_batch_cols

if added:
    print(f"Schema Drift Detected — New columns added: {added}")
if removed:
    print(f"Schema Drift Detected — Columns removed: {removed}")
if not added and not removed:
    print("No schema drift detected.")

# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest E-Commerce Data into Delta Lake
# MAGIC
# MAGIC **Purpose:** Load raw CSV files into Databricks Delta Lake using Apache Spark.
# MAGIC
# MAGIC **Run this notebook first.** Upload the Kaggle Brazilian E-Commerce (Olist) dataset
# MAGIC to DBFS before running, or use the synthetic data generator below.

# COMMAND ----------

# MAGIC %md ## Option A — Generate Synthetic Data (no download needed)

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

spark = SparkSession.builder.appName("DataCopilot-Ingest").getOrCreate()

RNG = np.random.default_rng(42)

CATEGORIES = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Beauty", "Toys", "Food"]
STATUSES = ["delivered", "shipped", "processing", "cancelled", "returned"]
STATUS_WEIGHTS = [0.65, 0.15, 0.10, 0.07, 0.03]
STATES = ["CA", "TX", "NY", "FL", "WA", "IL", "PA", "OH", "GA", "NC"]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "bank_transfer"]

N_CUSTOMERS, N_ORDERS, N_PRODUCTS = 1_000, 5_000, 200

# Customers
customer_ids = [f"C{i:04d}" for i in range(1, N_CUSTOMERS + 1)]
reg_dates = pd.date_range("2022-01-01", "2024-12-31", periods=N_CUSTOMERS)
customers_pd = pd.DataFrame({
    "customer_id": customer_ids,
    "name": [f"Customer_{i}" for i in range(1, N_CUSTOMERS + 1)],
    "state": RNG.choice(STATES, N_CUSTOMERS).tolist(),
    "city": [f"City_{RNG.integers(1, 50)}" for _ in range(N_CUSTOMERS)],
    "registration_date": reg_dates.strftime("%Y-%m-%d").tolist(),
    "loyalty_tier": RNG.choice(["Bronze", "Silver", "Gold", "Platinum"], N_CUSTOMERS, p=[0.5, 0.3, 0.15, 0.05]).tolist(),
})

# Products
products_pd = pd.DataFrame({
    "product_id": [f"P{i:04d}" for i in range(1, N_PRODUCTS + 1)],
    "product_name": [f"Product_{i}" for i in range(1, N_PRODUCTS + 1)],
    "category": RNG.choice(CATEGORIES, N_PRODUCTS).tolist(),
    "base_price": RNG.uniform(5, 500, N_PRODUCTS).round(2).tolist(),
    "weight_kg": RNG.uniform(0.1, 20, N_PRODUCTS).round(2).tolist(),
})

# Orders
order_ids = [f"O{i:06d}" for i in range(1, N_ORDERS + 1)]
order_dates = pd.date_range("2023-01-01", "2024-12-31", periods=N_ORDERS)
statuses = RNG.choice(STATUSES, N_ORDERS, p=STATUS_WEIGHTS).tolist()

orders_pd = pd.DataFrame({
    "order_id": order_ids,
    "customer_id": RNG.choice(customer_ids, N_ORDERS).tolist(),
    "order_date": order_dates.strftime("%Y-%m-%d").tolist(),
    "status": statuses,
    "payment_method": RNG.choice(PAYMENT_METHODS, N_ORDERS).tolist(),
    "total_amount": RNG.uniform(10, 2000, N_ORDERS).round(2).tolist(),
})

# Order items
item_rows = []
for order_id in order_ids:
    n_items = int(RNG.integers(1, 5))
    selected = products_pd.sample(n=n_items)
    for _, prod in selected.iterrows():
        qty = int(RNG.integers(1, 6))
        price = round(float(prod["base_price"]) * float(RNG.uniform(0.85, 1.15)), 2)
        item_rows.append({
            "order_id": order_id,
            "product_id": prod["product_id"],
            "category": prod["category"],
            "quantity": qty,
            "unit_price": price,
            "line_total": round(price * qty, 2),
        })
order_items_pd = pd.DataFrame(item_rows)

print(f"Customers: {len(customers_pd):,}")
print(f"Products: {len(products_pd):,}")
print(f"Orders: {len(orders_pd):,}")
print(f"Order items: {len(order_items_pd):,}")

# COMMAND ----------

# MAGIC %md ## Convert to Spark DataFrames

# COMMAND ----------

customers_sdf = spark.createDataFrame(customers_pd)
products_sdf = spark.createDataFrame(products_pd)
orders_sdf = spark.createDataFrame(orders_pd)
order_items_sdf = spark.createDataFrame(order_items_pd)

# COMMAND ----------

# MAGIC %md ## Write to Delta Lake

# COMMAND ----------

DELTA_BASE = "dbfs:/user/hive/warehouse/datacopilot"

customers_sdf.write.format("delta").mode("overwrite").save(f"{DELTA_BASE}/customers")
products_sdf.write.format("delta").mode("overwrite").save(f"{DELTA_BASE}/products")
orders_sdf.write.format("delta").mode("overwrite").save(f"{DELTA_BASE}/orders")
order_items_sdf.write.format("delta").mode("overwrite").save(f"{DELTA_BASE}/order_items")

print("Delta tables written successfully.")

# COMMAND ----------

# MAGIC %md ## Register as SQL Tables

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS datacopilot")
spark.sql(f"CREATE TABLE IF NOT EXISTS datacopilot.customers USING DELTA LOCATION '{DELTA_BASE}/customers'")
spark.sql(f"CREATE TABLE IF NOT EXISTS datacopilot.products USING DELTA LOCATION '{DELTA_BASE}/products'")
spark.sql(f"CREATE TABLE IF NOT EXISTS datacopilot.orders USING DELTA LOCATION '{DELTA_BASE}/orders'")
spark.sql(f"CREATE TABLE IF NOT EXISTS datacopilot.order_items USING DELTA LOCATION '{DELTA_BASE}/order_items'")

spark.sql("SHOW TABLES IN datacopilot").show()

"""Generates synthetic e-commerce demo data — no download needed."""
import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)

CATEGORIES = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Beauty", "Toys", "Food"]
STATUSES = ["delivered", "shipped", "processing", "cancelled", "returned"]
STATUS_WEIGHTS = [0.65, 0.15, 0.10, 0.07, 0.03]
STATES = ["CA", "TX", "NY", "FL", "WA", "IL", "PA", "OH", "GA", "NC"]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "bank_transfer"]

N_CUSTOMERS = 1_000
N_ORDERS = 5_000
N_PRODUCTS = 200


def _make_customers() -> pd.DataFrame:
    customer_ids = [f"C{i:04d}" for i in range(1, N_CUSTOMERS + 1)]
    reg_dates = pd.date_range("2022-01-01", "2024-12-31", periods=N_CUSTOMERS)
    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "name": [f"Customer_{i}" for i in range(1, N_CUSTOMERS + 1)],
            "state": RNG.choice(STATES, N_CUSTOMERS),
            "city": [f"City_{RNG.integers(1, 50)}" for _ in range(N_CUSTOMERS)],
            "registration_date": RNG.choice(reg_dates, N_CUSTOMERS),
            "loyalty_tier": RNG.choice(["Bronze", "Silver", "Gold", "Platinum"], N_CUSTOMERS, p=[0.5, 0.3, 0.15, 0.05]),
        }
    )


def _make_products() -> pd.DataFrame:
    product_ids = [f"P{i:04d}" for i in range(1, N_PRODUCTS + 1)]
    return pd.DataFrame(
        {
            "product_id": product_ids,
            "product_name": [f"Product_{i}" for i in range(1, N_PRODUCTS + 1)],
            "category": RNG.choice(CATEGORIES, N_PRODUCTS),
            "base_price": RNG.uniform(5, 500, N_PRODUCTS).round(2),
            "weight_kg": RNG.uniform(0.1, 20, N_PRODUCTS).round(2),
        }
    )


def _make_orders(customers: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    order_ids = [f"O{i:06d}" for i in range(1, N_ORDERS + 1)]
    order_dates = pd.date_range("2023-01-01", "2024-12-31", periods=N_ORDERS)

    customer_ids = RNG.choice(customers["customer_id"].values, N_ORDERS)
    statuses = RNG.choice(STATUSES, N_ORDERS, p=STATUS_WEIGHTS)
    payment_methods = RNG.choice(PAYMENT_METHODS, N_ORDERS)

    delivery_days = RNG.integers(2, 30, N_ORDERS)
    delivered_dates = [
        (pd.Timestamp(order_dates[i]) + pd.Timedelta(days=int(delivery_days[i]))).date()
        if statuses[i] == "delivered"
        else None
        for i in range(N_ORDERS)
    ]

    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_ids,
            "order_date": order_dates.date,
            "status": statuses,
            "payment_method": payment_methods,
            "delivery_date": delivered_dates,
        }
    )

    # order items — 1-4 items per order
    item_rows = []
    for order_id in order_ids:
        n_items = RNG.integers(1, 5)
        selected_products = products.sample(n=int(n_items), random_state=int(RNG.integers(0, 9999)))
        for _, prod in selected_products.iterrows():
            qty = int(RNG.integers(1, 6))
            price = round(float(prod["base_price"]) * RNG.uniform(0.85, 1.15), 2)
            item_rows.append(
                {
                    "order_id": order_id,
                    "product_id": prod["product_id"],
                    "category": prod["category"],
                    "quantity": qty,
                    "unit_price": price,
                    "line_total": round(price * qty, 2),
                }
            )
    order_items = pd.DataFrame(item_rows)

    # Back-fill total_amount into orders
    totals = order_items.groupby("order_id")["line_total"].sum().reset_index()
    totals.columns = ["order_id", "total_amount"]
    orders = orders.merge(totals, on="order_id", how="left")

    return orders, order_items


def generate_demo_tables() -> dict[str, pd.DataFrame]:
    customers = _make_customers()
    products = _make_products()
    orders, order_items = _make_orders(customers, products)
    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": order_items,
    }

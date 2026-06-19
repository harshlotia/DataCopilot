"""Synthetic demo datasets — no download needed."""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


# ── E-Commerce ────────────────────────────────────────────────────────────────
CATEGORIES = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Beauty", "Toys", "Food"]
STATUSES = ["delivered", "shipped", "processing", "cancelled", "returned"]
STATUS_WEIGHTS = [0.65, 0.15, 0.10, 0.07, 0.03]
STATES = ["CA", "TX", "NY", "FL", "WA", "IL", "PA", "OH", "GA", "NC"]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "bank_transfer"]
N_CUSTOMERS, N_ORDERS, N_PRODUCTS = 1_000, 5_000, 200


def _make_customers() -> pd.DataFrame:
    customer_ids = [f"C{i:04d}" for i in range(1, N_CUSTOMERS + 1)]
    reg_dates = pd.date_range("2022-01-01", "2024-12-31", periods=N_CUSTOMERS)
    return pd.DataFrame({
        "customer_id": customer_ids,
        "name": [f"Customer_{i}" for i in range(1, N_CUSTOMERS + 1)],
        "state": RNG.choice(STATES, N_CUSTOMERS),
        "city": [f"City_{RNG.integers(1, 50)}" for _ in range(N_CUSTOMERS)],
        "registration_date": RNG.choice(reg_dates, N_CUSTOMERS),
        "loyalty_tier": RNG.choice(["Bronze", "Silver", "Gold", "Platinum"], N_CUSTOMERS, p=[0.5, 0.3, 0.15, 0.05]),
    })


def _make_products() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": [f"P{i:04d}" for i in range(1, N_PRODUCTS + 1)],
        "product_name": [f"Product_{i}" for i in range(1, N_PRODUCTS + 1)],
        "category": RNG.choice(CATEGORIES, N_PRODUCTS),
        "base_price": RNG.uniform(5, 500, N_PRODUCTS).round(2),
        "weight_kg": RNG.uniform(0.1, 20, N_PRODUCTS).round(2),
    })


def _make_orders(customers, products):
    order_ids = [f"O{i:06d}" for i in range(1, N_ORDERS + 1)]
    order_dates = pd.date_range("2023-01-01", "2024-12-31", periods=N_ORDERS)
    customer_ids = RNG.choice(customers["customer_id"].values, N_ORDERS)
    statuses = RNG.choice(STATUSES, N_ORDERS, p=STATUS_WEIGHTS)
    payment_methods = RNG.choice(PAYMENT_METHODS, N_ORDERS)
    delivery_days = RNG.integers(2, 30, N_ORDERS)
    delivered_dates = [
        (pd.Timestamp(order_dates[i]) + pd.Timedelta(days=int(delivery_days[i]))).date()
        if statuses[i] == "delivered" else None
        for i in range(N_ORDERS)
    ]
    orders = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": customer_ids,
        "order_date": order_dates.date,
        "status": statuses,
        "payment_method": payment_methods,
        "delivery_date": delivered_dates,
    })
    item_rows = []
    for order_id in order_ids:
        n_items = RNG.integers(1, 5)
        selected = products.sample(n=int(n_items), random_state=int(RNG.integers(0, 9999)))
        for _, prod in selected.iterrows():
            qty = int(RNG.integers(1, 6))
            price = round(float(prod["base_price"]) * RNG.uniform(0.85, 1.15), 2)
            item_rows.append({
                "order_id": order_id,
                "product_id": prod["product_id"],
                "category": prod["category"],
                "quantity": qty,
                "unit_price": price,
                "line_total": round(price * qty, 2),
            })
    order_items = pd.DataFrame(item_rows)
    totals = order_items.groupby("order_id")["line_total"].sum().reset_index()
    totals.columns = ["order_id", "total_amount"]
    orders = orders.merge(totals, on="order_id", how="left")
    return orders, order_items


def generate_demo_tables() -> dict[str, pd.DataFrame]:
    customers = _make_customers()
    products = _make_products()
    orders, order_items = _make_orders(customers, products)
    return {"customers": customers, "products": products, "orders": orders, "order_items": order_items}


# ── Stock Market ──────────────────────────────────────────────────────────────
TICKERS = {
    "AAPL": ("Apple Inc.", "Technology", 175.0),
    "MSFT": ("Microsoft Corp.", "Technology", 380.0),
    "GOOGL": ("Alphabet Inc.", "Technology", 140.0),
    "AMZN": ("Amazon.com Inc.", "Consumer Cyclical", 178.0),
    "TSLA": ("Tesla Inc.", "Consumer Cyclical", 250.0),
    "NVDA": ("NVIDIA Corp.", "Technology", 800.0),
    "META": ("Meta Platforms", "Technology", 490.0),
    "JPM": ("JPMorgan Chase", "Financial Services", 195.0),
    "JNJ": ("Johnson & Johnson", "Healthcare", 155.0),
    "V": ("Visa Inc.", "Financial Services", 275.0),
    "WMT": ("Walmart Inc.", "Consumer Defensive", 170.0),
    "XOM": ("Exxon Mobil", "Energy", 110.0),
}


def generate_stock_tables() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(99)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")  # business days only

    price_rows = []
    for ticker, (company, sector, start_price) in TICKERS.items():
        price = start_price
        for date in dates:
            change_pct = rng.normal(0.0003, 0.018)
            price = max(price * (1 + change_pct), 1.0)
            daily_range = price * rng.uniform(0.005, 0.025)
            high = round(price + daily_range, 2)
            low = round(price - daily_range, 2)
            close = round(price, 2)
            open_ = round(low + rng.uniform(0, high - low), 2)
            volume = int(rng.integers(5_000_000, 80_000_000))
            price_rows.append({
                "ticker": ticker,
                "company": company,
                "sector": sector,
                "date": date.date(),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "market_cap_b": round(close * rng.uniform(5, 50), 1),
            })

    stock_prices = pd.DataFrame(price_rows)

    company_info = pd.DataFrame([
        {"ticker": t, "company": v[0], "sector": v[1], "employees": int(rng.integers(10_000, 200_000)),
         "founded_year": int(rng.integers(1960, 2005)), "headquarters": rng.choice(["CA", "WA", "NY", "TX"])}
        for t, v in TICKERS.items()
    ])

    return {"stock_prices": stock_prices, "company_info": company_info}


# ── Hospital / Healthcare ─────────────────────────────────────────────────────
DEPARTMENTS = ["Cardiology", "Orthopedics", "Neurology", "Oncology", "Pediatrics",
               "Emergency", "Radiology", "General Surgery", "Psychiatry", "Dermatology"]
DIAGNOSES = {
    "Cardiology":      ["Heart Failure", "Arrhythmia", "Hypertension", "Coronary Artery Disease"],
    "Orthopedics":     ["Fracture", "Knee Replacement", "Arthritis", "Spinal Stenosis"],
    "Neurology":       ["Stroke", "Epilepsy", "Migraine", "Multiple Sclerosis"],
    "Oncology":        ["Breast Cancer", "Lung Cancer", "Leukemia", "Colon Cancer"],
    "Pediatrics":      ["Asthma", "Ear Infection", "RSV", "Appendicitis"],
    "Emergency":       ["Trauma", "Chest Pain", "Laceration", "Overdose"],
    "Radiology":       ["CT Scan", "MRI", "X-Ray", "Ultrasound"],
    "General Surgery": ["Appendectomy", "Gallbladder Removal", "Hernia Repair", "Bowel Resection"],
    "Psychiatry":      ["Depression", "Anxiety Disorder", "Bipolar Disorder", "PTSD"],
    "Dermatology":     ["Psoriasis", "Eczema", "Skin Cancer", "Acne"],
}
N_PATIENTS = 2_000
N_VISITS = 8_000


def generate_hospital_tables() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(77)

    patients = pd.DataFrame({
        "patient_id": [f"PT{i:05d}" for i in range(1, N_PATIENTS + 1)],
        "age": rng.integers(1, 95, N_PATIENTS),
        "gender": rng.choice(["Male", "Female", "Other"], N_PATIENTS, p=[0.49, 0.49, 0.02]),
        "blood_type": rng.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"], N_PATIENTS),
        "state": rng.choice(STATES, N_PATIENTS),
        "insurance": rng.choice(["Medicare", "Medicaid", "Private", "Uninsured"], N_PATIENTS, p=[0.3, 0.2, 0.4, 0.1]),
        "registration_date": rng.choice(pd.date_range("2020-01-01", "2024-12-31", periods=N_PATIENTS), N_PATIENTS),
    })

    visit_rows = []
    for i in range(N_VISITS):
        dept = rng.choice(DEPARTMENTS)
        admit = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 730)))
        los = int(rng.integers(1, 21))
        discharge = admit + pd.Timedelta(days=los)
        bill = round(float(rng.uniform(500, 80_000)), 2)
        visit_rows.append({
            "visit_id": f"V{i+1:06d}",
            "patient_id": f"PT{rng.integers(1, N_PATIENTS+1):05d}",
            "department": dept,
            "diagnosis": rng.choice(DIAGNOSES[dept]),
            "admission_date": admit.date(),
            "discharge_date": discharge.date(),
            "length_of_stay_days": los,
            "bill_amount": bill,
            "outcome": rng.choice(["Recovered", "Transferred", "Deceased", "Ongoing"], p=[0.80, 0.10, 0.03, 0.07]),
            "doctor_id": f"DR{rng.integers(1, 51):03d}",
        })
    visits = pd.DataFrame(visit_rows)

    doctors = pd.DataFrame({
        "doctor_id": [f"DR{i:03d}" for i in range(1, 51)],
        "name": [f"Dr. {rng.choice(['Smith','Johnson','Williams','Brown','Jones','Garcia','Miller','Davis'])} {i}" for i in range(1, 51)],
        "department": rng.choice(DEPARTMENTS, 50),
        "years_experience": rng.integers(1, 35, 50),
        "patient_rating": rng.uniform(3.5, 5.0, 50).round(1),
        "patients_per_month": rng.integers(20, 120, 50),
    })

    dept_stats = pd.DataFrame({
        "department": DEPARTMENTS,
        "num_beds": rng.integers(10, 80, len(DEPARTMENTS)),
        "avg_wait_time_mins": rng.integers(10, 180, len(DEPARTMENTS)),
        "annual_budget_m": rng.uniform(2, 50, len(DEPARTMENTS)).round(1),
    })

    return {"patients": patients, "visits": visits, "doctors": doctors, "departments": dept_stats}


# ── HR / Employees ────────────────────────────────────────────────────────────
HR_DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations", "Product", "Legal", "Support", "Design"]
ROLES = {
    "Engineering": ["Software Engineer", "Senior Engineer", "Staff Engineer", "Engineering Manager"],
    "Sales":       ["Sales Rep", "Account Executive", "Sales Manager", "VP Sales"],
    "Marketing":   ["Marketing Analyst", "Content Manager", "Growth Manager", "CMO"],
    "Finance":     ["Financial Analyst", "Senior Analyst", "Finance Manager", "CFO"],
    "HR":          ["HR Coordinator", "HR Business Partner", "HR Manager", "CHRO"],
    "Operations":  ["Operations Analyst", "Operations Manager", "Director of Operations", "COO"],
    "Product":     ["Product Manager", "Senior PM", "Principal PM", "VP Product"],
    "Legal":       ["Legal Counsel", "Senior Counsel", "Associate GC", "General Counsel"],
    "Support":     ["Support Specialist", "Support Lead", "Support Manager", "Director Support"],
    "Design":      ["UX Designer", "Senior Designer", "Lead Designer", "Head of Design"],
}
BASE_SALARIES = {
    "Engineering": (90_000, 220_000), "Sales": (55_000, 180_000), "Marketing": (60_000, 160_000),
    "Finance": (70_000, 170_000), "HR": (55_000, 130_000), "Operations": (60_000, 150_000),
    "Product": (85_000, 200_000), "Legal": (80_000, 200_000), "Support": (45_000, 110_000),
    "Design": (70_000, 160_000),
}
N_EMPLOYEES = 1_500


def generate_hr_tables() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(55)

    depts = rng.choice(HR_DEPARTMENTS, N_EMPLOYEES, p=[0.25, 0.15, 0.10, 0.08, 0.07, 0.08, 0.10, 0.04, 0.08, 0.05])
    roles = [rng.choice(ROLES[d]) for d in depts]
    hire_dates = pd.date_range("2015-01-01", "2024-12-31", periods=N_EMPLOYEES)
    salaries = [
        int(rng.integers(BASE_SALARIES[d][0], BASE_SALARIES[d][1]))
        for d in depts
    ]
    performance_ratings = rng.choice([1, 2, 3, 4, 5], N_EMPLOYEES, p=[0.05, 0.10, 0.35, 0.35, 0.15])

    employees = pd.DataFrame({
        "employee_id": [f"E{i:04d}" for i in range(1, N_EMPLOYEES + 1)],
        "department": depts,
        "role": roles,
        "gender": rng.choice(["Male", "Female", "Non-binary"], N_EMPLOYEES, p=[0.52, 0.44, 0.04]),
        "age": rng.integers(22, 62, N_EMPLOYEES),
        "state": rng.choice(STATES, N_EMPLOYEES),
        "hire_date": rng.choice(hire_dates, N_EMPLOYEES),
        "salary": salaries,
        "performance_rating": performance_ratings,
        "is_remote": rng.choice([True, False], N_EMPLOYEES, p=[0.45, 0.55]),
        "years_at_company": rng.integers(0, 10, N_EMPLOYEES),
        "bonus_pct": rng.uniform(0, 20, N_EMPLOYEES).round(1),
    })

    dept_summary = pd.DataFrame({
        "department": HR_DEPARTMENTS,
        "headcount": [int((depts == d).sum()) for d in HR_DEPARTMENTS],
        "avg_salary": [int(np.mean([salaries[i] for i, dep in enumerate(depts) if dep == d])) for d in HR_DEPARTMENTS],
        "avg_performance": [round(float(np.mean([performance_ratings[i] for i, dep in enumerate(depts) if dep == d])), 2) for d in HR_DEPARTMENTS],
        "annual_budget_m": rng.uniform(1, 30, len(HR_DEPARTMENTS)).round(1),
    })

    reviews = pd.DataFrame({
        "employee_id": rng.choice([f"E{i:04d}" for i in range(1, N_EMPLOYEES + 1)], 3_000),
        "review_year": rng.choice([2021, 2022, 2023, 2024], 3_000),
        "rating": rng.choice([1, 2, 3, 4, 5], 3_000, p=[0.05, 0.10, 0.35, 0.35, 0.15]),
        "bonus_paid": rng.uniform(0, 25_000, 3_000).round(2),
        "promoted": rng.choice([True, False], 3_000, p=[0.12, 0.88]),
    })

    return {"employees": employees, "department_summary": dept_summary, "performance_reviews": reviews}

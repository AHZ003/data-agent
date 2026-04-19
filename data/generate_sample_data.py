"""Generate sample datasets for testing and demos."""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_superstore_sales():
    """Generate a Superstore-style sales dataset."""
    n = 2000
    categories = ["Technology", "Furniture", "Office Supplies"]
    sub_categories = {
        "Technology": ["Phones", "Computers", "Accessories", "Copiers"],
        "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
        "Office Supplies": ["Paper", "Binders", "Art", "Envelopes"],
    }
    regions = ["East", "West", "Central", "South"]
    segments = ["Consumer", "Corporate", "Home Office"]

    dates = pd.date_range("2021-01-01", "2024-12-31", periods=n)
    cats = np.random.choice(categories, n)
    sub_cats = [np.random.choice(sub_categories[c]) for c in cats]

    df = pd.DataFrame({
        "Order Date": dates,
        "Region": np.random.choice(regions, n),
        "Segment": np.random.choice(segments, n),
        "Category": cats,
        "Sub-Category": sub_cats,
        "Sales": np.round(np.random.lognormal(5, 1.2, n), 2),
        "Quantity": np.random.randint(1, 14, n),
        "Discount": np.round(np.random.choice([0, 0, 0, 0.1, 0.15, 0.2, 0.3], n), 2),
        "Profit": np.round(np.random.normal(50, 100, n), 2),
    })
    df["Order ID"] = [f"ORD-{i:05d}" for i in range(n)]
    df = df[["Order ID", "Order Date", "Region", "Segment", "Category",
             "Sub-Category", "Sales", "Quantity", "Discount", "Profit"]]
    df.to_csv(os.path.join(DATA_DIR, "superstore_sales.csv"), index=False)
    print(f"Generated superstore_sales.csv ({n} rows)")


def generate_ecommerce():
    """Generate an e-commerce transactions dataset."""
    n = 1500
    products = ["Laptop", "Phone", "Tablet", "Headphones", "Keyboard",
                "Mouse", "Monitor", "Webcam", "Charger", "Case"]
    channels = ["Web", "Mobile App", "Social Media", "Email"]
    payment = ["Credit Card", "PayPal", "Debit Card", "Apple Pay"]

    df = pd.DataFrame({
        "Transaction Date": pd.date_range("2023-01-01", "2024-12-31", periods=n),
        "Customer ID": [f"CUST-{np.random.randint(1, 500):04d}" for _ in range(n)],
        "Product": np.random.choice(products, n),
        "Channel": np.random.choice(channels, n),
        "Payment Method": np.random.choice(payment, n),
        "Amount": np.round(np.random.lognormal(4, 1, n), 2),
        "Quantity": np.random.randint(1, 5, n),
        "Returned": np.random.choice([True, False], n, p=[0.08, 0.92]),
    })
    df.to_csv(os.path.join(DATA_DIR, "ecommerce_transactions.csv"), index=False)
    print(f"Generated ecommerce_transactions.csv ({n} rows)")


def generate_employee_data():
    """Generate an employee dataset."""
    n = 500
    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
    titles = ["Analyst", "Senior Analyst", "Manager", "Senior Manager", "Director", "VP"]

    df = pd.DataFrame({
        "Employee ID": [f"EMP-{i:04d}" for i in range(n)],
        "Hire Date": pd.date_range("2015-01-01", "2024-06-30", periods=n),
        "Department": np.random.choice(departments, n),
        "Title": np.random.choice(titles, n),
        "Salary": np.round(np.random.normal(85000, 25000, n), -2),
        "Performance Score": np.round(np.random.normal(3.5, 0.8, n).clip(1, 5), 1),
        "Years Experience": np.random.randint(0, 25, n),
        "Satisfaction Score": np.round(np.random.normal(3.8, 0.7, n).clip(1, 5), 1),
        "Left Company": np.random.choice([True, False], n, p=[0.15, 0.85]),
    })
    df.to_csv(os.path.join(DATA_DIR, "employee_data.csv"), index=False)
    print(f"Generated employee_data.csv ({n} rows)")


if __name__ == "__main__":
    generate_superstore_sales()
    generate_ecommerce()
    generate_employee_data()
    print("\nAll sample datasets generated!")

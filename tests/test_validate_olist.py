from __future__ import annotations

import pandas as pd
import pytest

from scripts.validate_olist import build_olist_examples


def _write_tiny_olist_dataset(folder, n: int = 12) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    purchase = pd.date_range("2017-01-01", periods=n, freq="D")
    orders = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "customer_id": [f"c{i}" for i in range(n)],
        "order_status": ["delivered"] * n,
        "order_purchase_timestamp": purchase.astype(str),
        "order_delivered_customer_date": (purchase + pd.to_timedelta([i % 4 for i in range(n)], unit="D")).astype(str),
        "order_estimated_delivery_date": (purchase + pd.to_timedelta(2, unit="D")).astype(str),
    })
    items = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "price": [10.0 + i for i in range(n)],
        "freight_value": [2.0] * n,
        "seller_id": [f"s{i % 3}" for i in range(n)],
        "product_id": [f"p{i % 2}" for i in range(n)],
    })
    customers = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "customer_state": ["SP" if i % 2 else "RJ" for i in range(n)],
    })
    products = pd.DataFrame({
        "product_id": ["p0", "p1"],
        "product_category_name": ["home", "books"],
    })
    orders.to_csv(folder / "olist_orders_dataset.csv", index=False)
    items.to_csv(folder / "olist_order_items_dataset.csv", index=False)
    customers.to_csv(folder / "olist_customers_dataset.csv", index=False)
    products.to_csv(folder / "olist_products_dataset.csv", index=False)


def test_olist_examples_exclude_outcome_columns_from_features(tmp_path):
    _write_tiny_olist_dataset(tmp_path)
    result = build_olist_examples(tmp_path)

    assert len(result) == 12
    assert set(result["late_delivery"].unique()) == {0, 1}
    assert "order_delivered_customer_date" not in result.columns
    assert "order_estimated_delivery_date" not in result.columns
    assert "order_status" not in result.columns
    assert {"promised_lead_days", "total_price", "customer_state"} <= set(result.columns)


def test_olist_examples_ignore_non_delivered_orders(tmp_path):
    _write_tiny_olist_dataset(tmp_path)
    orders_path = tmp_path / "olist_orders_dataset.csv"
    orders = pd.read_csv(orders_path)
    orders.loc[0, "order_status"] = "canceled"
    orders.to_csv(orders_path, index=False)

    result = build_olist_examples(tmp_path)
    assert len(result) == 11
    assert "o0" not in set(result["order_id"])


def test_olist_examples_explain_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download the Olist dataset"):
        build_olist_examples(tmp_path)

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.validate_olist import (
    _classification_metrics,
    build_olist_examples,
    run_olist_validation,
)


def _write_tiny_olist_dataset(folder, n: int = 12) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    purchase = pd.date_range("2017-01-01", periods=n, freq="D")
    orders = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "customer_id": [f"c{i}" for i in range(n)],
        "order_status": ["delivered"] * n,
        "order_purchase_timestamp": purchase.astype(str),
        "order_delivered_customer_date": (
            purchase + pd.to_timedelta([i % 4 for i in range(n)], unit="D")
        ).astype(str),
        "order_estimated_delivery_date": (
            purchase + pd.to_timedelta(2, unit="D")
        ).astype(str),
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


def test_full_validation_pipeline_runs_on_synthetic_fixture(tmp_path):
    _write_tiny_olist_dataset(tmp_path, n=300)

    report = run_olist_validation(tmp_path)

    assert report["metadata"]["n_usable_orders"] == 300
    assert report["metadata"]["train_rows"] == 180
    assert report["metadata"]["validation_rows"] == 60
    assert report["metadata"]["test_rows"] == 60
    assert {row["model"] for row in report["models"]} == {
        "dummy_most_frequent",
        "logistic_regression",
        "random_forest",
    }
    assert all("f1" in row and "brier_score" in row for row in report["models"])


def test_single_class_test_split_reports_undefined_ranking_metrics():
    metrics = _classification_metrics(pd.Series([1, 1, 1]), np.array([0.2, 0.8, 0.9]), 0.5)

    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None
    assert metrics["recall"] == pytest.approx(2 / 3)

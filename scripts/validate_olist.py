"""External validation on the real Olist Brazilian e-commerce dataset.

The Olist data is user-supplied and is not redistributed here. Download it from
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce and place the CSVs
in data/real_world/ (see docs/real_world_validation.md).

Only features plausibly available at purchase time are used. Delivery dates
are used solely to construct the target and never as model features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "real_world"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "real_world" / "olist_validation.json"

REQUIRED_ORDER_COLUMNS = {
    "order_id",
    "order_status",
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
}
REQUIRED_ITEM_COLUMNS = {"order_id", "price", "freight_value", "seller_id", "product_id"}
REQUIRED_CUSTOMER_COLUMNS = {"customer_id", "customer_state"}
REQUIRED_PRODUCT_COLUMNS = {"product_id", "product_category_name"}


def _read_csv(data_dir: Path, filename: str, required: bool = True) -> pd.DataFrame | None:
    path = data_dir / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Missing {path}. Download the Olist dataset from "
                "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce "
                "and extract the CSV files into this directory."
            )
        return None
    return pd.read_csv(path)


def _require_columns(frame: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing required columns: {', '.join(missing)}")


def build_olist_examples(data_dir: str | Path) -> pd.DataFrame:
    """Build order-level features and a late-delivery target from Olist CSVs."""
    root = Path(data_dir)
    orders = _read_csv(root, "olist_orders_dataset.csv")
    items = _read_csv(root, "olist_order_items_dataset.csv")
    customers = _read_csv(root, "olist_customers_dataset.csv")
    products = _read_csv(root, "olist_products_dataset.csv")

    assert orders is not None and items is not None and customers is not None and products is not None
    _require_columns(orders, REQUIRED_ORDER_COLUMNS, "olist_orders_dataset.csv")
    _require_columns(items, REQUIRED_ITEM_COLUMNS, "olist_order_items_dataset.csv")
    _require_columns(customers, REQUIRED_CUSTOMER_COLUMNS, "olist_customers_dataset.csv")
    _require_columns(products, REQUIRED_PRODUCT_COLUMNS, "olist_products_dataset.csv")

    orders = orders.loc[orders["order_status"].eq("delivered")].copy()
    for col in ("order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"):
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    orders = orders.dropna(
        subset=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
    )
    if orders.empty:
        raise ValueError("No delivered orders with valid purchase, actual delivery, and estimated delivery dates.")

    # Target only: actual outcome columns are never passed to the feature matrix.
    orders["late_delivery"] = (
        orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]
    ).astype("int8")

    product_lookup = products[["product_id", "product_category_name"]].drop_duplicates("product_id")
    item_rows = items.merge(product_lookup, on="product_id", how="left", validate="many_to_one")
    item_rows["product_category_name"] = item_rows["product_category_name"].fillna("unknown")
    item_agg = item_rows.groupby("order_id", as_index=False).agg(
        item_count=("order_id", "size"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        seller_count=("seller_id", "nunique"),
        category_count=("product_category_name", "nunique"),
        dominant_category=("product_category_name", lambda values: values.mode().iloc[0] if not values.mode().empty else "unknown"),
    )

    customer_lookup = customers[["customer_id", "customer_state"]].drop_duplicates("customer_id")
    examples = orders.merge(item_agg, on="order_id", how="inner", validate="one_to_one")
    examples = examples.merge(customer_lookup, on="customer_id", how="left", validate="many_to_one")

    purchase = examples["order_purchase_timestamp"]
    estimated = examples["order_estimated_delivery_date"]
    examples["purchase_year"] = purchase.dt.year
    examples["purchase_month"] = purchase.dt.month
    examples["purchase_dayofweek"] = purchase.dt.dayofweek
    examples["purchase_hour"] = purchase.dt.hour
    examples["promised_lead_days"] = (estimated - purchase).dt.total_seconds() / 86400.0
    examples["customer_state"] = examples["customer_state"].fillna("unknown")

    feature_columns = [
        "purchase_year", "purchase_month", "purchase_dayofweek", "purchase_hour",
        "promised_lead_days", "item_count", "total_price", "total_freight",
        "seller_count", "category_count", "dominant_category", "customer_state",
    ]
    result = examples[["order_id", "order_purchase_timestamp", "late_delivery", *feature_columns]].copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=["order_purchase_timestamp", "late_delivery"])
    if result["late_delivery"].nunique() != 2:
        raise ValueError("The usable Olist sample must contain both on-time and late-delivery orders.")
    return result.sort_values("order_purchase_timestamp").reset_index(drop=True)


def _make_pipeline(model: Any, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def _classification_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype("int8")
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def run_olist_validation(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    """Run a chronological train/validation/test evaluation on real Olist data."""
    examples = build_olist_examples(data_dir)
    n = len(examples)
    train_end, val_end = int(n * 0.60), int(n * 0.80)
    if train_end < 100 or val_end - train_end < 30 or n - val_end < 30:
        raise ValueError(f"Not enough examples for chronological 60/20/20 split: {n} usable orders.")

    feature_columns = [c for c in examples.columns if c not in {"order_id", "order_purchase_timestamp", "late_delivery"}]
    numeric_features = [c for c in feature_columns if pd.api.types.is_numeric_dtype(examples[c])]
    categorical_features = [c for c in feature_columns if c not in numeric_features]
    X = examples[feature_columns]
    y = examples["late_delivery"]
    X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": _make_pipeline(
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=2026),
            numeric_features, categorical_features,
        ),
        "random_forest": _make_pipeline(
            RandomForestClassifier(
                n_estimators=250, max_depth=16, min_samples_leaf=8,
                class_weight="balanced_subsample", n_jobs=-1, random_state=2026,
            ),
            numeric_features, categorical_features,
        ),
    }

    results: list[dict[str, Any]] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        # Threshold selection uses validation data only; the test set stays untouched.
        thresholds = np.linspace(0.1, 0.9, 33)
        val_f1 = [f1_score(y_val, val_probs >= threshold, zero_division=0) for threshold in thresholds]
        threshold = float(thresholds[int(np.argmax(val_f1))])
        test_probs = model.predict_proba(X_test)[:, 1]
        metrics = _classification_metrics(y_test, test_probs, threshold)
        metrics["model"] = name
        results.append(metrics)

    results.sort(key=lambda row: row["model"] != "dummy_most_frequent")
    results.sort(key=lambda row: row["f1"], reverse=True)
    return {
        "metadata": {
            "benchmark": "Olist Brazilian E-Commerce real-world proxy validation",
            "dataset_source": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
            "dataset_license": "CC BY-NC-SA 4.0; review license before reuse or redistribution",
            "dataset_period": "2016-2018",
            "dataset_type": "real anonymized commercial order data; not a supply-chain dataset",
            "target_definition": "delivered_customer_date > estimated_delivery_date",
            "feature_policy": "purchase-time features only; actual delivery date used only for target",
            "split_strategy": "chronological 60% train / 20% validation / 20% test",
            "n_usable_orders": int(n),
            "n_features": int(len(feature_columns)),
            "positive_rate": float(y.mean()),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "train_period": [str(examples.iloc[0]["order_purchase_timestamp"]), str(examples.iloc[train_end - 1]["order_purchase_timestamp"])],
            "validation_period": [str(examples.iloc[train_end]["order_purchase_timestamp"]), str(examples.iloc[val_end - 1]["order_purchase_timestamp"])],
            "test_period": [str(examples.iloc[val_end]["order_purchase_timestamp"]), str(examples.iloc[-1]["order_purchase_timestamp"])],
            "limitations": [
                "This evaluates delivery-delay risk on e-commerce orders, not industrial supply-chain disruption directly.",
                "The historical dataset is from 2016-2018 and may not represent current logistics.",
                "Results are observational and do not establish causal effects.",
                "Only delivered orders with complete delivery timestamps can be labelled.",
            ],
        },
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing Olist CSV files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report output path")
    args = parser.parse_args()

    report = run_olist_validation(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()

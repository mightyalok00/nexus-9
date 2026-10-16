"""Data validation and anti-leakage checks for FutureBench."""

from typing import List, Tuple
import pandas as pd
from nexus9.config import settings
from nexus9.exceptions import DataLeakageError, DataValidationError

EXPECTED_COLUMNS = [
    "scenario_id",
    "environment_regime",
    "economic_pressure",
    "weather_severity",
    "supplier_reliability",
    "demand_shock",
    "policy_strength",
    "capacity",
    "demand",
    "supplier_delay_days",
    "available_capacity",
    "inventory_gap",
    "delivery_delay_days",
    "operating_cost",
    "service_level",
    "disruption_risk",
]

def validate_schema(df: pd.DataFrame) -> bool:
    """Verify all expected columns exist and data types are plausible."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns in dataset: {missing}")
    return True

def assert_no_data_leakage(
    feature_names: List[str],
    leakage_columns: List[str] | None = None,
) -> None:
    """Enforce the critical dataset design rule:
    Early warning models must NOT include delivery_delay_days, inventory_gap,
    or service_level, as disruption_risk is mathematically derived from them.
    Also forbids scenario_id.
    """
    if leakage_columns is None:
        leakage_columns = settings.features.leakage_columns

    forbidden = set(leakage_columns) | {settings.features.id_column, settings.features.target_column}
    violating = [col for col in feature_names if col in forbidden]

    if violating:
        raise DataLeakageError(
            f"Data leakage detected! Features contain forbidden outcome-defining columns: {violating}."
            " Early warning models must exclude delivery_delay_days, inventory_gap, service_level, and scenario_id."
        )

def inspect_missingness(df: pd.DataFrame) -> pd.Series:
    """Return missing value counts across features."""
    return df.isnull().sum()

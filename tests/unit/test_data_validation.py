"""Unit tests for schema validation and anti-leakage checks."""

import pytest
import pandas as pd
from nexus9.data.validation import validate_schema, assert_no_data_leakage
from nexus9.exceptions import DataLeakageError

def test_validate_schema_passes_on_valid(raw_dataset):
    assert validate_schema(raw_dataset) is True

def test_assert_no_data_leakage_raises_on_leakage():
    # If feature list contains delivery_delay_days, must raise DataLeakageError
    with pytest.raises(DataLeakageError):
        assert_no_data_leakage(["economic_pressure", "delivery_delay_days", "weather_severity"])

def test_assert_no_data_leakage_raises_on_scenario_id():
    with pytest.raises(DataLeakageError):
        assert_no_data_leakage(["scenario_id", "economic_pressure"])

def test_assert_no_data_leakage_passes_on_clean():
    clean_features = ["economic_pressure", "weather_severity", "capacity", "demand"]
    assert_no_data_leakage(clean_features) # Should not raise

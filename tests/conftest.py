"""Shared pytest fixtures for unit and integration tests."""

import pytest
import pandas as pd
import numpy as np
from nexus9.data.loader import load_futurebench_data
from nexus9.data.splitting import create_train_val_test_splits

@pytest.fixture(scope="session")
def raw_dataset():
    return load_futurebench_data()

@pytest.fixture(scope="session")
def split_data(raw_dataset):
    return create_train_val_test_splits(raw_dataset)

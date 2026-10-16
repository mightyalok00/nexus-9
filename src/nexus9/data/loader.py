"""Data loader for FutureBench dataset."""

from pathlib import Path
from typing import Optional
import pandas as pd
from nexus9.config import PROJECT_ROOT, settings

def load_futurebench_data(
    file_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Load the FutureBench raw scenario dataset.

    Args:
        file_path: Optional path to the CSV file. Defaults to config raw_data path.

    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    if file_path is None:
        file_path = PROJECT_ROOT / settings.paths.raw_data

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"FutureBench dataset not found at {path}")

    df = pd.read_csv(path)
    return df

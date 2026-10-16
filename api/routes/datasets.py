"""Dataset inspection and summary endpoints."""

from fastapi import APIRouter, HTTPException
from nexus9.data.loader import load_futurebench_data
from nexus9.data.validation import inspect_missingness

router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.get("/summary")
def get_dataset_summary():
    try:
        df = load_futurebench_data()
        return {
            "total_rows": len(df),
            "columns": list(df.columns),
            "missing_values": inspect_missingness(df).to_dict(),
            "target_distribution": df["disruption_risk"].value_counts(normalize=True).to_dict(),
            "regimes": df["environment_regime"].value_counts().to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

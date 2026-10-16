"""Model Arena inference and metadata endpoints."""

from fastapi import APIRouter, HTTPException
import pandas as pd
from api.schemas import PredictionRequest, PredictionResponse
from nexus9.data.loader import load_futurebench_data
from nexus9.data.splitting import create_train_val_test_splits
from nexus9.ml_lab.registry import get_model, list_registered_models

router = APIRouter(prefix="/models", tags=["Models"])

# Cache trained models in memory for API fast response
TRAINED_MODELS = {}

def get_or_train_model(model_name: str):
    if model_name not in TRAINED_MODELS:
        df = load_futurebench_data()
        X_train, _, _, y_train, _, _ = create_train_val_test_splits(df)
        pipe = get_model(model_name)
        pipe.fit(X_train, y_train)
        TRAINED_MODELS[model_name] = pipe
    return TRAINED_MODELS[model_name]

@router.get("/")
def list_models():
    return {"registered_models": list_registered_models()}

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        model = get_or_train_model(request.model_name)
        df_instances = pd.DataFrame(request.instances)
        preds = model.predict(df_instances).tolist()

        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_instances)[:, 1].tolist()

        return PredictionResponse(
            model_name=request.model_name,
            predictions=preds,
            probabilities=probs,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

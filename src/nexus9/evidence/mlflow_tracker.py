"""MLflow experiment logging and artifact tracking integration."""

from pathlib import Path
from typing import Any, Dict, Optional

class Nexus9MLflowTracker:
    """Logs hyperparameters, metrics, and models to MLflow."""

    def __init__(self, experiment_name: str = "NEXUS-9_Model_Arena"):
        self.experiment_name = experiment_name
        self.tracking_uri = "file:./mlruns"

    def log_model_run(
        self,
        model_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Log run metadata cleanly with MLflow (graceful fallback if offline)."""
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=model_name):
                mlflow.log_params(parameters)
                mlflow.log_metrics(metrics)
            return {"status": "LOGGED_TO_MLFLOW", "experiment": self.experiment_name}
        except Exception as e:
            return {"status": "LOCAL_FALLBACK", "message": str(e)}

"""
app/schemas.py
--------------
Pydantic data schemas for AAVAIL Revenue Projection API.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class TrainRequest(BaseModel):
    raw_dir: Optional[str] = Field(default="data/raw", description="Directory containing raw JSON/CSV files")
    confidence_level: Optional[float] = Field(default=0.90, ge=0.50, le=0.99, description="Confidence interval coverage level")

class TrainResponse(BaseModel):
    status: str
    model_artifact: str
    training_time_sec: float
    ingestion_summary: Dict[str, Any]
    backtest_results: Dict[str, Any]

class PredictRequest(BaseModel):
    country: str = Field(default="united_kingdom", description="Target country code (e.g. united_kingdom, united_states, singapore, all)")
    target_date: Optional[str] = Field(default=None, description="Projection target date in YYYY-MM-DD format (optional)")
    days_ahead: Optional[int] = Field(default=30, ge=1, le=90, description="Forecast horizon in days (default 30)")

class PredictResponse(BaseModel):
    country: str
    target_date: str
    days_ahead: int
    predicted_revenue_30d: float = Field(description="Point estimate for total revenue over next 30 days")
    lower_bound: float = Field(description="Lower bound of confidence interval")
    upper_bound: float = Field(description="Upper bound of confidence interval")
    confidence_level: float = Field(description="Confidence level of interval (e.g. 0.90)")
    baseline_naive_30d: float = Field(description="Naive baseline forecast for comparison")
    model_architecture: str
    runtime_ms: float

class DriftResponse(BaseModel):
    drift_detected: bool
    drift_severity: str
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    logged_predictions_count: int
    recent_preds_mean: float
    reference_target_mean: float
    action_required: str

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    total_predictions_logged: int

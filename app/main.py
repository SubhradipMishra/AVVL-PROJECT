"""
app/main.py
-----------
FastAPI Production Microservice for AAVAIL Monthly Revenue Projection.
Endpoints:
- POST /train: Retrains model on data/raw, evaluates time-based backtest, updates artifacts.
- POST /predict: Returns 30-day country revenue point estimate + calibrated confidence interval.
- GET /logs: Retrieves recent structured prediction audit logs.
- GET /drift: Performs real-time distribution drift assessment (PSI + KS test).
- GET /eda: Returns summary statistics and country heterogeneity tests.
- GET /health: Health status and metadata.
"""

import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any

from app.schemas import (
    TrainRequest,
    TrainResponse,
    PredictRequest,
    PredictResponse,
    DriftResponse,
    HealthResponse
)
from src.pipeline import train_pipeline, predict_pipeline, MODEL_PATH, PROCESSED_DATA_PATH
from src.logger import get_recent_prediction_logs, logger
from src.monitoring import assess_drift
from src.eda import run_eda

app = FastAPI(
    title="AAVAIL Monthly Revenue Projection Service",
    description="Enterprise API providing country-level 30-day revenue point estimates and calibrated prediction intervals.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", tags=["General"])
def root():
    return {
        "service": "AAVAIL Revenue Projection API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": ["/train", "/predict", "/logs", "/drift", "/eda", "/health"]
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    model_exists = os.path.exists(MODEL_PATH)
    logs = get_recent_prediction_logs(limit=1000)
    return {
        "status": "HEALTHY",
        "service": "aavail-revenue-projection",
        "version": "1.0.0",
        "model_loaded": model_exists,
        "total_predictions_logged": len(logs)
    }

@app.post("/train", response_model=TrainResponse, tags=["Model Operations"])
def trigger_training(req: TrainRequest = TrainRequest()):
    """Triggers end-to-end ingestion, feature engineering, time-based backtesting, and model serialization."""
    try:
        res = train_pipeline(
            raw_dir=req.raw_dir or "data/raw",
            confidence_level=req.confidence_level or 0.90
        )
        return res
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training pipeline error: {str(e)}")

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def get_prediction(req: PredictRequest):
    """
    Computes a 30-day ahead revenue projection and calibrated confidence interval for a specified country.
    """
    try:
        res = predict_pipeline(
            country=req.country,
            target_date=req.target_date,
            days_ahead=req.days_ahead or 30
        )
        return res
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=400, detail=str(fnf))
    except Exception as e:
        logger.error(f"Prediction failed for country {req.country}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/logs", tags=["Auditing"])
def get_logs(limit: int = Query(50, ge=1, le=500), country: Optional[str] = Query(None)):
    """Retrieves recent structured prediction audit logs."""
    logs = get_recent_prediction_logs(limit=limit, country=country)
    return {
        "count": len(logs),
        "filter_country": country,
        "logs": logs
    }

@app.get("/drift", response_model=DriftResponse, tags=["Monitoring"])
def get_drift_status():
    """Evaluates prediction distribution drift against historical training baseline using PSI and KS-test."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise HTTPException(status_code=400, detail="Processed historical data not found. Run training first.")
    
    ref_df = pd.read_csv(PROCESSED_DATA_PATH)
    drift_report = assess_drift(reference_data=ref_df)
    
    if drift_report.get("status") in ["INSUFFICIENT_DATA", "INSUFFICIENT_LOGS"]:
        raise HTTPException(status_code=422, detail=drift_report["message"])
        
    return drift_report

@app.get("/eda", tags=["Analytics"])
def get_eda_summary():
    """Returns exploratory data analysis metrics, top 10 countries, and statistical tests."""
    try:
        summary = run_eda()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EDA error: {str(e)}")

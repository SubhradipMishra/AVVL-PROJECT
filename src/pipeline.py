"""
src/pipeline.py
---------------
End-to-end Training and Inference Pipeline Orchestration for AAVAIL.
Provides unified functions to:
1. Ingest raw data -> Build features -> Backtest -> Train Production Model -> Save Artifacts
2. Single-call Inference with country feature lookup and prediction logging
"""

import os
import time
import json
import logging
import pandas as pd
from typing import Dict, Any, Optional

from src.ingestion import run_ingestion_pipeline
from src.features import build_feature_dataset, extract_latest_feature_vector, engineer_country_features
from src.models import MultiCountryModelManager
from src.evaluate import run_time_based_backtest
from src.logger import log_training_run, log_prediction, logger

MODEL_PATH = os.getenv("MODEL_PATH", "models/aavail_model.joblib")
PROCESSED_DATA_PATH = "data/processed/daily_country_revenue.csv"

def train_pipeline(
    raw_dir: str = "data/raw",
    model_output_path: str = MODEL_PATH,
    confidence_level: float = 0.90
) -> Dict[str, Any]:
    """Executes full training pipeline, generates metrics, and saves model artifacts."""
    start_time = time.time()
    logger.info("Starting AAVAIL training pipeline...")

    # 1. Ingest data
    daily_df, ingest_stats = run_ingestion_pipeline(raw_dir=raw_dir)

    # 2. Build feature table
    feat_df = build_feature_dataset(daily_df, target_horizon=30)

    # 3. Rigorous Time-based backtesting
    backtest_report = run_time_based_backtest(feat_df, test_days=90)

    # 4. Train final production model on full valid dataset
    logger.info("Training production MultiCountryModelManager...")
    manager = MultiCountryModelManager(confidence_level=confidence_level)
    manager.fit(feat_df)
    manager.save(model_output_path)

    elapsed_time = time.time() - start_time

    # 5. Log training session
    train_start = ingest_stats["date_range_start"]
    train_end = ingest_stats["date_range_end"]
    best_metrics = backtest_report["models"]["dedicated_per_country_gb"]

    log_training_run(
        model_name="MultiCountryModelManager_GB",
        train_start=train_start,
        train_end=train_end,
        metrics=best_metrics,
        best_params={"confidence_level": confidence_level, "model_path": model_output_path},
        runtime_sec=elapsed_time
    )

    return {
        "status": "SUCCESS",
        "model_artifact": model_output_path,
        "training_time_sec": round(elapsed_time, 2),
        "ingestion_summary": ingest_stats,
        "backtest_results": backtest_report
    }

def predict_pipeline(
    country: str,
    target_date: Optional[str] = None,
    days_ahead: int = 30,
    model_path: str = MODEL_PATH,
    data_path: str = PROCESSED_DATA_PATH
) -> Dict[str, Any]:
    """Executes real-time inference for a given country with latency tracking and audit logging."""
    start_time = time.time()
    country_clean = country.lower().strip()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please trigger training first.")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed time series data not found at {data_path}.")

    manager = MultiCountryModelManager.load(model_path)
    daily_df = pd.read_csv(data_path)
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    country_sub = daily_df[daily_df["country"] == country_clean]
    if country_sub.empty:
        # If unknown country, use 'all' global aggregate baseline scaled by small factor or fall back
        country_sub = daily_df[daily_df["country"] == "all"]

    # Extract latest feature row
    feat_row = extract_latest_feature_vector(country_sub, target_horizon=days_ahead)
    pred_res = manager.predict_country(country_clean, feat_row, use_per_country=True)

    elapsed_ms = (time.time() - start_time) * 1000

    # Determine reference target date
    if not target_date:
        last_date = country_sub["date"].max()
        target_date = str((last_date + pd.Timedelta(days=days_ahead)).date())

    # Log prediction
    log_prediction(
        country=country_clean,
        target_date=target_date,
        days_ahead=days_ahead,
        predicted_revenue=pred_res["predicted_revenue_30d"],
        lower_bound=pred_res["lower_bound"],
        upper_bound=pred_res["upper_bound"],
        confidence_level=pred_res["confidence_level"],
        model_version="1.0.0",
        runtime_ms=elapsed_ms,
        metadata={"architecture": pred_res["model_architecture"]}
    )

    pred_res["target_date"] = target_date
    pred_res["days_ahead"] = days_ahead
    pred_res["runtime_ms"] = round(elapsed_ms, 2)
    return pred_res

if __name__ == "__main__":
    t_res = train_pipeline()
    print("Train Pipeline Result:", t_res["status"])
    p_res = predict_pipeline("united_kingdom")
    print("UK Prediction:", p_res)

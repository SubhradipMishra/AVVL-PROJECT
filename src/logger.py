"""
src/logger.py
-------------
Production structured logger and prediction auditor for AAVAIL Revenue Projection.
Logs model training metrics, runtime latencies, inputs/outputs, and drift metrics.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, Optional, List

LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Application logger configuration
app_log_file = os.path.join(LOG_DIR, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(app_log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("aavail_revenue")

PREDICTION_LOG_FILE = os.path.join(LOG_DIR, "predictions.jsonl")
TRAINING_LOG_FILE = os.path.join(LOG_DIR, "training.jsonl")

def log_prediction(
    country: str,
    target_date: str,
    days_ahead: int,
    predicted_revenue: float,
    lower_bound: float,
    upper_bound: float,
    confidence_level: float,
    model_version: str,
    runtime_ms: float,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Records every inference call in structured JSONL format for audit and drift tracking."""
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "country": country,
        "target_date": target_date,
        "days_ahead": days_ahead,
        "predicted_revenue": round(predicted_revenue, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "confidence_level": confidence_level,
        "model_version": model_version,
        "runtime_ms": round(runtime_ms, 3),
        "metadata": metadata or {}
    }
    
    with open(PREDICTION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        
    logger.info(f"PREDICT: country={country} target={target_date} revenue={predicted_revenue:.2f} [{lower_bound:.2f}, {upper_bound:.2f}] latency={runtime_ms:.2f}ms")
    return record

def log_training_run(
    model_name: str,
    train_start: str,
    train_end: str,
    metrics: Dict[str, float],
    best_params: Optional[Dict[str, Any]] = None,
    runtime_sec: float = 0.0
) -> Dict[str, Any]:
    """Records model training session details and backtest validation scores."""
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_name": model_name,
        "train_start": train_start,
        "train_end": train_end,
        "metrics": metrics,
        "best_params": best_params or {},
        "runtime_sec": round(runtime_sec, 3)
    }
    with open(TRAINING_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        
    logger.info(f"TRAIN: model={model_name} metrics={metrics} runtime={runtime_sec:.2f}s")
    return record

def get_recent_prediction_logs(limit: int = 50, country: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves recent prediction log records with optional country filtering."""
    if not os.path.exists(PREDICTION_LOG_FILE):
        return []
    
    records = []
    with open(PREDICTION_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if country and rec.get("country", "").lower() != country.lower():
                    continue
                records.append(rec)
            except Exception:
                pass
    return records[-limit:]

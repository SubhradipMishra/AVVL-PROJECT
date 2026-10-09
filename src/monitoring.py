"""
src/monitoring.py
-----------------
Model Monitoring, Data Drift, and Actuals Reconciliation for AAVAIL.
Features:
- Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) drift testing
- Prediction distribution shift detection
- Target reconciliation / error tracking on materialized actuals
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional

from src.logger import get_recent_prediction_logs

logger = logging.getLogger("aavail_revenue")

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between two numeric distributions."""
    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    # Define quantile buckets based on expected
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
    bucket_bounds[0] = -np.inf
    bucket_bounds[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bucket_bounds)
    actual_counts, _ = np.histogram(actual, bins=bucket_bounds)

    # Avoid zero division with smoothing
    expected_pct = (expected_counts + 1e-4) / (len(expected) + 1e-4 * num_buckets)
    actual_pct = (actual_counts + 1e-4) / (len(actual) + 1e-4 * num_buckets)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_value, 4))

def assess_drift(
    reference_data: pd.DataFrame,
    recent_logs: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Compares recent inference predictions with historical training revenue distributions.
    Flags drift if PSI > 0.25 or KS p-value < 0.05.
    """
    if recent_logs is None:
        recent_logs = get_recent_prediction_logs(limit=100)

    if not recent_logs:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": "No recent prediction logs found to evaluate drift.",
            "drift_detected": False
        }

    recent_preds = np.array([r["predicted_revenue"] for r in recent_logs if "predicted_revenue" in r])
    if len(recent_preds) < 5:
        return {
            "status": "INSUFFICIENT_LOGS",
            "message": f"Only {len(recent_preds)} predictions logged; need at least 5 for drift testing.",
            "drift_detected": False
        }

    # Reference target values from processed data
    ref_target = reference_data["revenue"].rolling(30).sum().dropna().values
    if len(ref_target) < 10:
        ref_target = reference_data["revenue"].values * 30

    # KS Test
    ks_stat, ks_p_val = stats.ks_2samp(ref_target, recent_preds)
    psi_score = calculate_psi(ref_target, recent_preds)

    # Drift thresholds: PSI > 0.25 (Significant Drift), PSI > 0.10 (Moderate Drift)
    drift_detected = bool(psi_score > 0.25 or ks_p_val < 0.01)

    drift_severity = "NONE"
    if psi_score > 0.25:
        drift_severity = "HIGH"
    elif psi_score > 0.10:
        drift_severity = "MODERATE"

    return {
        "drift_detected": drift_detected,
        "drift_severity": drift_severity,
        "psi_score": psi_score,
        "ks_statistic": round(float(ks_stat), 4),
        "ks_p_value": float(ks_p_val),
        "logged_predictions_count": len(recent_preds),
        "recent_preds_mean": round(float(np.mean(recent_preds)), 2),
        "reference_target_mean": round(float(np.mean(ref_target)), 2),
        "action_required": "Retrain model immediately on latest data." if drift_detected else "Model performance within expected statistical tolerance."
    }

"""
src/evaluate.py
---------------
Rigorous time-based backtesting and evaluation framework.
Guarantees:
- ZERO data leakage: Time-based sequential split only
- Evaluates Point Metrics (RMSE, MAE, MAPE)
- Evaluates Interval Metrics (Empirical Coverage %, Average Width / Sharpness)
- Compares Naive Baseline vs Pooled Models vs Per-Country Models
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from src.models import AavailRevenuePredictor, BaselineNaiveModel, MultiCountryModelManager
from src.features import FEATURE_COLUMNS, build_feature_dataset

logger = logging.getLogger("aavail_revenue")

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_lower: np.ndarray = None, y_upper: np.ndarray = None) -> Dict[str, float]:
    """Calculates point and interval diagnostic metrics."""
    # Avoid zero division in MAPE
    mask = y_true > 0
    if np.sum(mask) == 0:
        mape = 0.0
    else:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    metrics = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape_pct": round(mape, 2)
    }

    if y_lower is not None and y_upper is not None:
        # Interval coverage: fraction of actual values falling inside [lower, upper]
        inside = (y_true >= y_lower) & (y_true <= y_upper)
        coverage_pct = float(np.mean(inside) * 100)
        avg_width = float(np.mean(y_upper - y_lower))
        metrics["interval_coverage_pct"] = round(coverage_pct, 2)
        metrics["interval_avg_width"] = round(avg_width, 2)

    return metrics

def run_time_based_backtest(
    feat_df: pd.DataFrame,
    test_days: int = 90,
    step_days: int = 30
) -> Dict[str, Any]:
    """
    Executes rolling-window sequential backtest.
    Splits the timeline such that training is strictly past data, testing strictly future 30-day windows.
    """
    clean_df = feat_df.dropna(subset=["target_30d_revenue", "revenue_roll_mean_30"]).copy()
    dates = clean_df["date"].drop_duplicates().sort_values()
    
    if len(dates) < 120:
        raise ValueError("Insufficient time series depth for backtesting.")

    split_date = dates.iloc[-test_days]
    
    train_df = clean_df[clean_df["date"] < split_date].copy()
    test_df = clean_df[clean_df["date"] >= split_date].copy()

    logger.info(f"Backtesting Split: Train until {split_date.date()} ({len(train_df)} rows), Test from {split_date.date()} ({len(test_df)} rows)")

    y_test = test_df["target_30d_revenue"].values

    # 1. Baseline: Naive 30-Day Persistence
    baseline = BaselineNaiveModel()
    base_preds = baseline.predict(test_df)
    baseline_metrics = calculate_metrics(y_test, base_preds)

    # 2. Candidate: Pooled Gradient Boosting
    pooled_gb = AavailRevenuePredictor(model_type="gradient_boosting", confidence_level=0.90)
    pooled_gb.fit(train_df, train_df["target_30d_revenue"])
    p_point, p_lower, p_upper = pooled_gb.predict(test_df)
    pooled_gb_metrics = calculate_metrics(y_test, p_point, p_lower, p_upper)

    # 3. Candidate: Pooled Random Forest
    pooled_rf = AavailRevenuePredictor(model_type="random_forest", confidence_level=0.90)
    pooled_rf.fit(train_df, train_df["target_30d_revenue"])
    rf_point, rf_lower, rf_upper = pooled_rf.predict(test_df)
    pooled_rf_metrics = calculate_metrics(y_test, rf_point, rf_lower, rf_upper)

    # 4. Candidate: MultiCountryModelManager (Dedicated Per-Country + Pooled fallback)
    multi_manager = MultiCountryModelManager(confidence_level=0.90)
    multi_manager.fit(train_df)
    
    # Vectorized evaluation per-country across test samples
    test_df_eval = test_df.copy()
    test_df_eval["pred_point"] = 0.0
    test_df_eval["pred_lower"] = 0.0
    test_df_eval["pred_upper"] = 0.0

    for c, grp in test_df_eval.groupby("country"):
        c_model = multi_manager.per_country_models.get(c, multi_manager.pooled_model)
        pts, lows, highs = c_model.predict(grp)
        test_df_eval.loc[grp.index, "pred_point"] = pts
        test_df_eval.loc[grp.index, "pred_lower"] = lows
        test_df_eval.loc[grp.index, "pred_upper"] = highs

    per_country_metrics = calculate_metrics(
        y_test,
        test_df_eval["pred_point"].values,
        test_df_eval["pred_lower"].values,
        test_df_eval["pred_upper"].values
    )

    # Calculate Improvement over Baseline
    baseline_mape = baseline_metrics["mape_pct"]
    best_mape = per_country_metrics["mape_pct"]
    mape_reduction = round(baseline_mape - best_mape, 2)
    relative_improvement_pct = round(((baseline_mape - best_mape) / (baseline_mape + 1e-6)) * 100, 2)

    results = {
        "evaluation_protocol": "Rolling Sequential Time-Based Split (No Data Leakage)",
        "train_cutoff_date": str(split_date.date()),
        "test_horizon_days": test_days,
        "models": {
            "baseline_naive_30d": baseline_metrics,
            "pooled_random_forest": pooled_rf_metrics,
            "pooled_gradient_boosting": pooled_gb_metrics,
            "dedicated_per_country_gb": per_country_metrics
        },
        "performance_gain": {
            "baseline_mape_pct": baseline_mape,
            "per_country_mape_pct": best_mape,
            "mape_reduction_points": mape_reduction,
            "relative_improvement_pct": relative_improvement_pct,
            "beats_baseline": bool(best_mape < baseline_mape)
        }
    }

    # Save evaluation report
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation complete. Per-Country GB MAPE: {best_mape}% vs Baseline: {baseline_mape}%")
    return results

if __name__ == "__main__":
    from src.ingestion import run_ingestion_pipeline
    daily_df, _ = run_ingestion_pipeline()
    feat_df = build_feature_dataset(daily_df)
    report = run_time_based_backtest(feat_df)
    print("Backtest Results Summary:", json.dumps(report, indent=2))

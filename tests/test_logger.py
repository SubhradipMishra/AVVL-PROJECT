"""
tests/test_logger.py
--------------------
Unit tests for structured logging, prediction auditing, and training logging.
"""

import os
import json
import pytest
from src.logger import log_prediction, log_training_run, get_recent_prediction_logs, PREDICTION_LOG_FILE, TRAINING_LOG_FILE

def test_log_prediction_and_retrieve():
    rec = log_prediction(
        country="singapore",
        target_date="2020-05-01",
        days_ahead=30,
        predicted_revenue=15420.50,
        lower_bound=13200.00,
        upper_bound=17800.00,
        confidence_level=0.90,
        model_version="1.0.0",
        runtime_ms=12.4
    )
    
    assert rec["country"] == "singapore"
    assert rec["predicted_revenue"] == 15420.50
    assert rec["lower_bound"] == 13200.00
    
    recent = get_recent_prediction_logs(limit=10, country="singapore")
    assert len(recent) >= 1
    assert recent[-1]["country"] == "singapore"

def test_log_training_run():
    rec = log_training_run(
        model_name="TestModel",
        train_start="2018-01-01",
        train_end="2019-12-31",
        metrics={"mape_pct": 5.4, "rmse": 120.5},
        best_params={"alpha": 0.9},
        runtime_sec=3.2
    )
    assert rec["model_name"] == "TestModel"
    assert rec["metrics"]["mape_pct"] == 5.4
    assert os.path.exists(TRAINING_LOG_FILE)

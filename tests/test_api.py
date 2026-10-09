"""
tests/test_api.py
-----------------
Unit and integration tests for FastAPI service endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "model_loaded" in data

def test_logs_endpoint():
    response = client.get("/logs?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "logs" in data
    assert "count" in data

def test_train_and_predict_flow():
    # Trigger train
    train_res = client.post("/train", json={"raw_dir": "data/raw", "confidence_level": 0.90})
    assert train_res.status_code == 200
    train_data = train_res.json()
    assert train_data["status"] == "SUCCESS"
    assert "backtest_results" in train_data

    # Test predict
    pred_res = client.post("/predict", json={
        "country": "united_kingdom",
        "target_date": "2020-02-01",
        "days_ahead": 30
    })
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert pred_data["country"] == "united_kingdom"
    assert pred_data["predicted_revenue_30d"] > 0
    assert pred_data["lower_bound"] <= pred_data["predicted_revenue_30d"] <= pred_data["upper_bound"]
    assert pred_data["confidence_level"] == 0.90
    assert "runtime_ms" in pred_data

def test_eda_endpoint():
    response = client.get("/eda")
    assert response.status_code == 200
    data = response.json()
    assert "top_10_countries_by_revenue" in data
    assert "statistical_tests" in data

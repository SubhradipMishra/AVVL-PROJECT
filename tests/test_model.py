"""
tests/test_model.py
-------------------
Unit tests for baseline models, Gradient Boosting quantile predictors, and MultiCountryModelManager.
"""

import pytest
import numpy as np
import pandas as pd
from src.features import engineer_country_features, build_feature_dataset
from src.models import BaselineNaiveModel, AavailRevenuePredictor, MultiCountryModelManager

@pytest.fixture
def synthetic_time_series():
    dates = pd.date_range("2019-01-01", periods=150, freq="D")
    np.random.seed(42)
    revenue = 100.0 + (np.arange(150) * 0.5) + np.random.normal(0, 10, 150)
    revenue = np.maximum(10.0, revenue)
    
    df = pd.DataFrame({
        "country": "united_kingdom",
        "date": dates,
        "revenue": revenue,
        "purchases": (revenue / 10).astype(int),
        "unique_customers": (revenue / 20).astype(int),
        "views": (revenue / 5).astype(int),
        "services_count": 3
    })
    return df

def test_baseline_naive_model(synthetic_time_series):
    feat_df = engineer_country_features(synthetic_time_series, target_horizon=30)
    baseline = BaselineNaiveModel()
    preds = baseline.predict(feat_df)
    
    assert len(preds) == len(feat_df)
    assert not np.isnan(preds[-1])
    assert preds[-1] > 0

def test_gradient_boosting_intervals(synthetic_time_series):
    feat_df = engineer_country_features(synthetic_time_series, target_horizon=30)
    valid_df = feat_df.dropna(subset=["target_30d_revenue", "revenue_roll_mean_30"])
    
    model = AavailRevenuePredictor(model_type="gradient_boosting", confidence_level=0.90)
    model.fit(valid_df, valid_df["target_30d_revenue"])
    
    point, lower, upper = model.predict(valid_df.iloc[[-1]])
    
    assert len(point) == 1
    assert lower[0] >= 0.0
    assert lower[0] <= point[0]
    assert point[0] <= upper[0]

def test_multi_country_manager(synthetic_time_series):
    # Create multi country dataframe
    df_uk = synthetic_time_series.copy()
    df_us = synthetic_time_series.copy()
    df_us["country"] = "united_states"
    df_us["revenue"] *= 1.5
    
    combined = pd.concat([df_uk, df_us], ignore_index=True)
    feat_df = build_feature_dataset(combined, target_horizon=30)
    
    manager = MultiCountryModelManager(confidence_level=0.90)
    manager.fit(feat_df)
    
    uk_latest = feat_df[feat_df["country"] == "united_kingdom"].iloc[[-1]]
    res = manager.predict_country("united_kingdom", uk_latest)
    
    assert res["country"] == "united_kingdom"
    assert res["predicted_revenue_30d"] > 0
    assert res["lower_bound"] <= res["predicted_revenue_30d"] <= res["upper_bound"]
    assert "model_architecture" in res

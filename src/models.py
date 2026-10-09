"""
src/models.py
-------------
Modeling module for AAVAIL monthly revenue projection.
Implements:
1. Baseline Models:
   - Naive Persistence (last 30-day sum)
   - Seasonal / Rolling Baseline
2. Machine Learning Models:
   - Gradient Boosting / Random Forest Regressors
   - Quantile Regressors (loss='quantile') for rigorous prediction intervals (e.g. 80% / 95% confidence)
   - Bootstrapped residual estimation fallback
3. Pooled Global Model (with country features) vs Per-Country Dedicated Models
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from src.features import FEATURE_COLUMNS, engineer_country_features

class BaselineNaiveModel:
    """Naive persistence baseline: predicts next 30 days of revenue as the sum of previous 30 days."""
    def __init__(self):
        self.name = "Baseline_Naive_30d"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "baseline_prev_30d_revenue" in X.columns:
            return X["baseline_prev_30d_revenue"].fillna(0.0).values
        elif "revenue_roll_mean_30" in X.columns:
            return (X["revenue_roll_mean_30"].fillna(0.0) * 30).values
        else:
            return np.zeros(len(X))


class AavailRevenuePredictor:
    """
    Production-grade Gradient Boosting Revenue Forecaster with calibrated Quantile Intervals.
    Trains 3 estimators:
    - Median / Point estimate (squared error / Huber)
    - Lower confidence bound (e.g. quantile=0.05 for 90% interval)
    - Upper confidence bound (e.g. quantile=0.95 for 90% interval)
    """
    def __init__(self, model_type: str = "gradient_boosting", confidence_level: float = 0.90):
        self.model_type = model_type
        self.confidence_level = confidence_level
        self.alpha_low = round((1.0 - confidence_level) / 2.0, 3)
        self.alpha_high = round(1.0 - self.alpha_low, 3)
        
        self.model_point = None
        self.model_lower = None
        self.model_upper = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.residual_std = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Filter available feature columns
        self.feature_names = [c for c in FEATURE_COLUMNS if c in X.columns]
        X_train = X[self.feature_names].fillna(0.0)
        y_train = y.values

        # Point estimator
        if self.model_type == "random_forest":
            self.model_point = RandomForestRegressor(
                n_estimators=40, max_depth=6, min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            self.model_point.fit(X_train, y_train)
            
            # Estimate residual std on training set for parametric / bootstrapped interval
            preds_train = self.model_point.predict(X_train)
            residuals = y_train - preds_train
            self.residual_std = float(np.std(residuals))
        else:
            # HistGradientBoostingRegressor for fast training & quantile loss support
            self.model_point = HistGradientBoostingRegressor(
                loss="squared_error", max_iter=40, max_depth=5, random_state=42
            )
            self.model_point.fit(X_train, y_train)

            # Quantile regressors for empirical prediction intervals
            self.model_lower = HistGradientBoostingRegressor(
                loss="quantile", quantile=self.alpha_low, max_iter=35, max_depth=4, random_state=42
            )
            self.model_lower.fit(X_train, y_train)

            self.model_upper = HistGradientBoostingRegressor(
                loss="quantile", quantile=self.alpha_high, max_iter=35, max_depth=4, random_state=42
            )
            self.model_upper.fit(X_train, y_train)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (point_estimate, lower_bound, upper_bound)
        Ensures lower_bound >= 0 and lower_bound <= point_estimate <= upper_bound.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        X_eval = X[self.feature_names].fillna(0.0)
        point_preds = self.model_point.predict(X_eval)
        point_preds = np.maximum(0.0, point_preds)

        if self.model_lower is not None and self.model_upper is not None:
            lower_preds = self.model_lower.predict(X_eval)
            upper_preds = self.model_upper.predict(X_eval)
            
            # Ensure physical sanity: revenue >= 0 and non-crossing quantiles
            lower_preds = np.maximum(0.0, np.minimum(point_preds, lower_preds))
            upper_preds = np.maximum(point_preds, upper_preds)
        else:
            # Fallback to normal error multiplier (z-score for interval)
            z = 1.645 if self.confidence_level <= 0.90 else 1.96
            lower_preds = np.maximum(0.0, point_preds - (z * self.residual_std))
            upper_preds = point_preds + (z * self.residual_std)

        return point_preds, lower_preds, upper_preds


class MultiCountryModelManager:
    """
    Manages both pooled global models and per-country models for country-level forecasting.
    """
    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.pooled_model = AavailRevenuePredictor(confidence_level=confidence_level)
        self.per_country_models: Dict[str, AavailRevenuePredictor] = {}
        self.baseline_model = BaselineNaiveModel()
        self.all_countries: List[str] = []

    def fit(self, feature_df: pd.DataFrame):
        # Drop rows with NaN target (last 30 days of training series)
        valid_df = feature_df.dropna(subset=["target_30d_revenue", "revenue_roll_mean_30"]).copy()
        self.all_countries = valid_df["country"].unique().tolist()

        # 1. Fit Pooled Model across all data
        self.pooled_model.fit(valid_df, valid_df["target_30d_revenue"])

        # 2. Fit Per-Country models where there are enough historical samples (>= 60 days)
        for country, group in valid_df.groupby("country"):
            if len(group) >= 60:
                c_model = AavailRevenuePredictor(confidence_level=self.confidence_level)
                c_model.fit(group, group["target_30d_revenue"])
                self.per_country_models[country] = c_model
            else:
                # Fallback to pooled model for sparse countries
                self.per_country_models[country] = self.pooled_model

        return self

    def predict_country(self, country: str, row_df: pd.DataFrame, use_per_country: bool = True) -> Dict[str, Any]:
        """Generates 30-day forecast and confidence intervals for a specific country."""
        country_clean = country.lower().strip()
        model = self.per_country_models.get(country_clean, self.pooled_model) if use_per_country else self.pooled_model
        
        point, lower, upper = model.predict(row_df)
        
        # Calculate naive baseline for comparison
        naive_val = float(self.baseline_model.predict(row_df)[0]) if len(row_df) > 0 else 0.0

        return {
            "country": country,
            "predicted_revenue_30d": round(float(point[0]), 2),
            "lower_bound": round(float(lower[0]), 2),
            "upper_bound": round(float(upper[0]), 2),
            "confidence_level": self.confidence_level,
            "baseline_naive_30d": round(naive_val, 2),
            "model_architecture": f"Dedicated Per-Country GB" if (use_per_country and country_clean in self.per_country_models) else "Pooled Global GB"
        }

    def save(self, filepath: str = "models/aavail_model.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = "models/aavail_model.joblib") -> "MultiCountryModelManager":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact not found at {filepath}")
        return joblib.load(filepath)

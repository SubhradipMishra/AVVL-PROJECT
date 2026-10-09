"""
src/features.py
--------------
Feature engineering for country-level time series revenue projection.
Computes:
1. Target: 30-day cumulative forward revenue (the Manager Hill target)
2. Lag features: 1, 7, 14, 21, 30-day past observations for revenue, purchases, customers, views
3. Rolling window statistics: 7, 14, 30-day rolling means, stds, mins, maxs
4. Customer engagement ratios: rev_per_customer, views_per_customer, purchases_per_customer
5. Calendar & Seasonality: day_of_week, day_of_month, month, is_weekend, cyclical sin/cos encodings
6. Country identifiers for pooled models
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

FEATURE_COLUMNS = [
    # Lags
    "revenue_lag_1", "revenue_lag_7", "revenue_lag_14", "revenue_lag_21", "revenue_lag_30",
    "purchases_lag_1", "purchases_lag_7", "purchases_lag_30",
    "unique_customers_lag_1", "unique_customers_lag_7", "unique_customers_lag_30",
    "views_lag_1", "views_lag_7", "views_lag_30",
    # Rolling Statistics
    "revenue_roll_mean_7", "revenue_roll_std_7", "revenue_roll_min_7", "revenue_roll_max_7",
    "revenue_roll_mean_14", "revenue_roll_std_14",
    "revenue_roll_mean_30", "revenue_roll_std_30", "revenue_roll_min_30", "revenue_roll_max_30",
    "purchases_roll_mean_7", "purchases_roll_mean_30",
    "unique_customers_roll_mean_7", "unique_customers_roll_mean_30",
    "views_roll_mean_7", "views_roll_mean_30",
    # Ratios
    "rev_per_customer_30d", "views_per_customer_30d", "purchases_per_customer_30d",
    # Calendar & Cyclical
    "day_of_week", "day_of_month", "month", "is_weekend",
    "dow_sin", "dow_cos", "month_sin", "month_cos"
]

def engineer_country_features(country_df: pd.DataFrame, target_horizon: int = 30) -> pd.DataFrame:
    """Builds historical lag, rolling, calendar, and cumulative forward target features for a single country."""
    df = country_df.sort_values("date").copy()
    
    # 1. Target: Cumulative forward revenue over the next `target_horizon` days (shifted back so today predicts next 30 days)
    # Target at index t = sum of revenue from (t+1) to (t+30)
    # Using rolling on reversed series or shift
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=target_horizon)
    df["target_30d_revenue"] = df["revenue"].rolling(window=indexer).sum().shift(-1)
    
    # Baseline benchmark: Previous 30-day rolling sum (Naive persistence baseline)
    df["baseline_prev_30d_revenue"] = df["revenue"].rolling(window=target_horizon).sum()
    
    # 2. Lag features
    for lag in [1, 7, 14, 21, 30]:
        df[f"revenue_lag_{lag}"] = df["revenue"].shift(lag)
        if lag in [1, 7, 30]:
            df[f"purchases_lag_{lag}"] = df["purchases"].shift(lag)
            df[f"unique_customers_lag_{lag}"] = df["unique_customers"].shift(lag)
            df[f"views_lag_{lag}"] = df["views"].shift(lag)

    # 3. Rolling window statistics
    for w in [7, 14, 30]:
        df[f"revenue_roll_mean_{w}"] = df["revenue"].rolling(window=w).mean()
        df[f"revenue_roll_std_{w}"] = df["revenue"].rolling(window=w).std()
        df[f"revenue_roll_min_{w}"] = df["revenue"].rolling(window=w).min()
        df[f"revenue_roll_max_{w}"] = df["revenue"].rolling(window=w).max()
        if w in [7, 30]:
            df[f"purchases_roll_mean_{w}"] = df["purchases"].rolling(window=w).mean()
            df[f"unique_customers_roll_mean_{w}"] = df["unique_customers"].rolling(window=w).mean()
            df[f"views_roll_mean_{w}"] = df["views"].rolling(window=w).mean()

    # 4. Customer engagement ratios
    df["rev_per_customer_30d"] = df["revenue_roll_mean_30"] / (df["unique_customers_roll_mean_30"] + 1e-6)
    df["views_per_customer_30d"] = df["views_roll_mean_30"] / (df["unique_customers_roll_mean_30"] + 1e-6)
    df["purchases_per_customer_30d"] = df["purchases_roll_mean_30"] / (df["unique_customers_roll_mean_30"] + 1e-6)

    # 5. Calendar features
    dates = pd.to_datetime(df["date"])
    df["day_of_week"] = dates.dt.dayofweek
    df["day_of_month"] = dates.dt.day
    df["month"] = dates.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical sin/cos encodings for periodic calendar patterns
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df

def build_feature_dataset(daily_df: pd.DataFrame, target_horizon: int = 30) -> pd.DataFrame:
    """Builds features across all countries and combines them into an analytical dataset."""
    country_dfs = []
    for country, group in daily_df.groupby("country"):
        country_feat = engineer_country_features(group, target_horizon=target_horizon)
        country_dfs.append(country_feat)

    full_feat_df = pd.concat(country_dfs, ignore_index=True)
    
    # Sort strictly by country and date
    full_feat_df = full_feat_df.sort_values(by=["country", "date"]).reset_index(drop=True)
    return full_feat_df

def extract_latest_feature_vector(country_df: pd.DataFrame, target_horizon: int = 30) -> pd.DataFrame:
    """Extracts the latest complete feature row for a given country to perform live inference."""
    feat_df = engineer_country_features(country_df, target_horizon=target_horizon)
    # Take the very last day in history
    latest_row = feat_df.iloc[[-1]].copy()
    return latest_row

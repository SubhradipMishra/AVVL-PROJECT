"""
src/eda.py
----------
Exploratory Data Analysis (EDA) module for AAVAIL Revenue Projection.
Performs:
1. Top ~10 country identification and ranking
2. Trend and Day-of-Week Seasonality analysis
3. Outlier detection (IQR method)
4. Statistical hypothesis testing (ANOVA / Kruskal-Wallis) to verify country revenue divergence
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List

logger = logging.getLogger("aavail_revenue")

def get_top_countries(df: pd.DataFrame, top_n: int = 10) -> List[str]:
    """Returns top N countries by total revenue excluding 'all'."""
    filtered = df[df["country"] != "all"]
    top = (
        filtered.groupby("country")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    return top

def analyze_seasonality_and_trend(country_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes trend slope, weekday/weekend multipliers, and autocorrelations."""
    c_df = country_df.sort_values("date").copy()
    c_df["dow"] = pd.to_datetime(c_df["date"]).dt.dayofweek
    c_df["is_weekend"] = c_df["dow"].isin([5, 6])
    
    # Weekday vs Weekend Average
    weekday_mean = float(c_df[~c_df["is_weekend"]]["revenue"].mean())
    weekend_mean = float(c_df[c_df["is_weekend"]]["revenue"].mean())
    weekend_lift = (weekend_mean - weekday_mean) / (weekday_mean + 1e-9)

    # 7-day and 30-day autocorrelation
    rev_series = c_df["revenue"]
    autocorr_7 = float(rev_series.autocorr(lag=7)) if len(rev_series) > 14 else 0.0
    autocorr_30 = float(rev_series.autocorr(lag=30)) if len(rev_series) > 60 else 0.0

    # Trend slope (Ordinary Least Squares on day index)
    x = np.arange(len(c_df))
    y = rev_series.values
    if len(x) > 1 and np.var(x) > 0:
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    else:
        slope, r_value, p_value = 0.0, 0.0, 1.0

    return {
        "mean_daily_revenue": round(float(rev_series.mean()), 2),
        "std_daily_revenue": round(float(rev_series.std()), 2),
        "median_daily_revenue": round(float(rev_series.median()), 2),
        "weekday_mean": round(weekday_mean, 2),
        "weekend_mean": round(weekend_mean, 2),
        "weekend_lift_pct": round(weekend_lift * 100, 2),
        "autocorr_7_day": round(autocorr_7, 3),
        "autocorr_30_day": round(autocorr_30, 3),
        "trend_slope_per_day": round(float(slope), 4),
        "trend_r2": round(float(r_value ** 2), 4)
    }

def detect_outliers(series: pd.Series) -> Dict[str, Any]:
    """Detects upper/lower outliers using the standard 1.5 * IQR rule."""
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    lower_bound = max(0.0, q25 - (1.5 * iqr))
    upper_bound = q75 + (1.5 * iqr)

    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return {
        "q25": round(float(q25), 2),
        "q75": round(float(q75), 2),
        "iqr": round(float(iqr), 2),
        "outlier_count": int(len(outliers)),
        "outlier_pct": round(float(len(outliers) / (len(series) + 1e-9) * 100), 2),
        "upper_threshold": round(float(upper_bound), 2)
    }

def test_country_heterogeneity(df: pd.DataFrame, top_countries: List[str]) -> Dict[str, Any]:
    """Performs One-Way ANOVA and Kruskal-Wallis test to verify if country revenue distributions differ."""
    groups = [
        df[df["country"] == c]["revenue"].values
        for c in top_countries
        if len(df[df["country"] == c]) > 10
    ]
    
    if len(groups) < 2:
        return {"error": "Insufficient country groups for hypothesis testing."}

    f_stat, p_val_anova = stats.f_oneway(*groups)
    kw_stat, p_val_kw = stats.kruskal(*groups)

    countries_significantly_differ = bool(p_val_anova < 0.05 and p_val_kw < 0.05)

    return {
        "anova_f_stat": round(float(f_stat), 3),
        "anova_p_value": float(p_val_anova),
        "kruskal_h_stat": round(float(kw_stat), 3),
        "kruskal_p_value": float(p_val_kw),
        "countries_significantly_differ": countries_significantly_differ,
        "interpretation": "Country distributions are statistically distinct (p < 0.05). Per-country modeling or country-specific features are strongly justified." if countries_significantly_differ else "No statistically significant difference between countries detected."
    }

def run_eda(df_or_path: Any = "data/processed/daily_country_revenue.csv") -> Dict[str, Any]:
    """Runs high-impact, time-boxed EDA across top countries and aggregate."""
    if isinstance(df_or_path, str):
        if not os.path.exists(df_or_path):
            raise FileNotFoundError(f"Processed dataset not found at {df_or_path}")
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    df["date"] = pd.to_datetime(df["date"])
    top_10 = get_top_countries(df, top_n=10)
    target_countries = ["all"] + top_10

    analysis = {
        "total_countries_available": int(df["country"].nunique()),
        "top_10_countries_by_revenue": top_10,
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "country_profiles": {}
    }

    for c in target_countries:
        c_sub = df[df["country"] == c]
        if c_sub.empty:
            continue
        stats_profile = analyze_seasonality_and_trend(c_sub)
        outlier_profile = detect_outliers(c_sub["revenue"])
        stats_profile["outliers"] = outlier_profile
        stats_profile["total_revenue"] = round(float(c_sub["revenue"].sum()), 2)
        stats_profile["total_purchases"] = int(c_sub["purchases"].sum())
        stats_profile["total_views"] = int(c_sub["views"].sum())
        analysis["country_profiles"][c] = stats_profile

    # Statistical hypothesis testing
    analysis["statistical_tests"] = test_country_heterogeneity(df, top_10)

    # Save summary artifact
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    logger.info("EDA completed successfully. Top countries and statistical divergence evaluated.")
    return analysis

if __name__ == "__main__":
    res = run_eda()
    print("EDA Summary Generated. Statistical Test:", res["statistical_tests"])

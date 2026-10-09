"""
src/ingestion.py
----------------
Robust Data Ingestion Pipeline for AAVAIL Revenue Projection.
Features:
- Multi-format ingestion (JSON and CSV across directories)
- Heterogeneous schema mapping and normalization
- Missing value imputation / filtering
- Invoice deduplication
- Gap-free daily per-country time-series aggregation (revenue, purchases, unique_customers, views)
- Global aggregate ('all') synthesis
"""

import os
import glob
import json
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

logger = logging.getLogger("aavail_revenue")

# Standard column mapping dictionary
COLUMN_MAPPINGS = {
    # Invoices
    "invoice": "invoice_id",
    "invoice_num": "invoice_id",
    "inv_id": "invoice_id",
    # Customer
    "customer_ID": "customer_id",
    "cust_id": "customer_id",
    "user_id": "customer_id",
    # Amount / Price
    "amount": "price",
    "revenue_amount": "price",
    "cost": "price",
    # Stream / Service
    "stream_type": "stream_id",
    "service_type": "stream_id",
    "product_id": "stream_id",
    # Views
    "times_streamed": "views",
    "total_views": "views",
    "stream_views": "views",
    # Country
    "country_code": "country",
    "location": "country"
}

def normalize_country_name(name: Any) -> str:
    """Standardizes country strings to clean lower_snake_case format."""
    if pd.isna(name) or not str(name).strip():
        return "unknown"
    s = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    # Clean common country synonyms
    synonyms = {
        "uk": "united_kingdom",
        "great_britain": "united_kingdom",
        "england": "united_kingdom",
        "us": "united_states",
        "usa": "united_states",
        "america": "united_states",
        "deutschland": "germany",
        "sg": "singapore",
        "fr": "france"
    }
    return synonyms.get(s, s)

def load_and_standardize_file(filepath: str) -> pd.DataFrame:
    """Reads a raw JSON or CSV file and normalizes its column schema."""
    if filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        return pd.DataFrame()

    if df.empty:
        return df

    # Apply column renames
    rename_dict = {}
    for col in df.columns:
        if col in COLUMN_MAPPINGS:
            rename_dict[col] = COLUMN_MAPPINGS[col]
        elif col.lower() in COLUMN_MAPPINGS:
            rename_dict[col] = COLUMN_MAPPINGS[col.lower()]
    df = df.rename(columns=rename_dict)

    # Standardize country
    if "country" in df.columns:
        df["country"] = df["country"].apply(normalize_country_name)
    else:
        df["country"] = "unknown"

    # Standardize price
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price"] = np.nan

    # Standardize views
    if "views" in df.columns:
        df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(1).astype(int)
    else:
        df["views"] = 1

    # Standardize date
    if "invoice_date" in df.columns:
        df["date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    elif all(c in df.columns for c in ["year", "month", "day"]):
        df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    # Fill customer_id if missing
    if "customer_id" not in df.columns:
        df["customer_id"] = "ANON_" + df.index.astype(str)
    else:
        df["customer_id"] = df["customer_id"].fillna("ANON")

    # Fill stream_id
    if "stream_id" not in df.columns:
        df["stream_id"] = "standard_stream"
    else:
        df["stream_id"] = df["stream_id"].fillna("standard_stream")

    return df[["invoice_id", "customer_id", "country", "stream_id", "price", "views", "date"]]

def clean_transactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Deduplicates transactions, handles nulls, and discards corrupt/negative amounts."""
    initial_count = len(df)
    
    # Drop rows with null dates or null prices
    clean_df = df.dropna(subset=["date", "price"]).copy()
    null_dropped = initial_count - len(clean_df)

    # Filter out anomalous/negative prices (refund anomalies)
    clean_df = clean_df[clean_df["price"] > 0]
    negative_dropped = initial_count - null_dropped - len(clean_df)

    # Deduplicate invoices
    dedup_subset = ["invoice_id", "customer_id", "country", "date", "price"]
    clean_df = clean_df.drop_duplicates(subset=dedup_subset)
    duplicates_dropped = (initial_count - null_dropped - negative_dropped) - len(clean_df)

    # Truncate date to calendar date (day level)
    clean_df["date"] = clean_df["date"].dt.normalize()

    stats = {
        "initial_records": initial_count,
        "clean_records": len(clean_df),
        "null_rows_dropped": null_dropped,
        "negative_rows_dropped": negative_dropped,
        "duplicate_rows_dropped": duplicates_dropped
    }
    return clean_df, stats

def aggregate_daily_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates clean transactions into a continuous daily per-country dataset."""
    # 1. Per-Country Aggregation
    country_agg = df.groupby(["country", "date"]).agg(
        revenue=("price", "sum"),
        purchases=("invoice_id", "count"),
        unique_customers=("customer_id", "nunique"),
        views=("views", "sum"),
        services_count=("stream_id", "nunique")
    ).reset_index()

    # 2. Global ('all') Aggregation
    all_agg = df.groupby("date").agg(
        revenue=("price", "sum"),
        purchases=("invoice_id", "count"),
        unique_customers=("customer_id", "nunique"),
        views=("views", "sum"),
        services_count=("stream_id", "nunique")
    ).reset_index()
    all_agg["country"] = "all"

    # Combine individual countries and the "all" total
    combined = pd.concat([country_agg, all_agg], ignore_index=True)

    # Ensure gap-free date continuous range for each country
    all_dates = pd.date_range(start=combined["date"].min(), end=combined["date"].max(), freq="D")
    unique_countries = combined["country"].unique()
    
    full_index = pd.MultiIndex.from_product([unique_countries, all_dates], names=["country", "date"])
    full_df = combined.set_index(["country", "date"]).reindex(full_index).reset_index()

    # Fill zero revenue/purchases/views for inactive days
    full_df["revenue"] = full_df["revenue"].fillna(0.0)
    full_df["purchases"] = full_df["purchases"].fillna(0).astype(int)
    full_df["unique_customers"] = full_df["unique_customers"].fillna(0).astype(int)
    full_df["views"] = full_df["views"].fillna(0).astype(int)
    full_df["services_count"] = full_df["services_count"].fillna(0).astype(int)

    # Sort strictly by country and date
    full_df = full_df.sort_values(by=["country", "date"]).reset_index(drop=True)
    return full_df

def run_ingestion_pipeline(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    output_filename: str = "daily_country_revenue.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Full data ingestion workflow from raw files to processed daily country table."""
    os.makedirs(output_dir, exist_ok=True)
    
    raw_files = glob.glob(os.path.join(raw_dir, "*.json")) + glob.glob(os.path.join(raw_dir, "*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No raw JSON or CSV data files found in '{raw_dir}'.")

    logger.info(f"Ingesting {len(raw_files)} raw files from '{raw_dir}'...")
    dfs = []
    for fp in raw_files:
        try:
            df_file = load_and_standardize_file(fp)
            if not df_file.empty:
                dfs.append(df_file)
        except Exception as e:
            logger.warning(f"Error parsing file {fp}: {e}")

    if not dfs:
        raise ValueError("Failed to ingest any valid transaction rows.")

    merged_raw = pd.concat(dfs, ignore_index=True)
    clean_df, clean_stats = clean_transactions(merged_raw)
    daily_df = aggregate_daily_time_series(clean_df)

    output_path = os.path.join(output_dir, output_filename)
    daily_df.to_csv(output_path, index=False)
    
    clean_stats["processed_file_path"] = output_path
    clean_stats["total_daily_rows"] = len(daily_df)
    clean_stats["num_countries"] = int(daily_df["country"].nunique())
    clean_stats["date_range_start"] = str(daily_df["date"].min().date())
    clean_stats["date_range_end"] = str(daily_df["date"].max().date())
    clean_stats["total_revenue"] = float(daily_df[daily_df["country"] == "all"]["revenue"].sum())

    logger.info(f"Ingestion complete: {clean_stats['total_daily_rows']} rows saved to {output_path}")
    return daily_df, clean_stats

if __name__ == "__main__":
    df, stats = run_ingestion_pipeline()
    print("Ingestion Summary:", json.dumps(stats, indent=2))

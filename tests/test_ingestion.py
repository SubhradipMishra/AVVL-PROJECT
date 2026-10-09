"""
tests/test_ingestion.py
-----------------------
Unit tests for data ingestion, deduplication, schema normalization, and daily aggregation.
"""

import os
import tempfile
import json
import pandas as pd
import numpy as np
import pytest

from src.ingestion import (
    normalize_country_name,
    load_and_standardize_file,
    clean_transactions,
    aggregate_daily_time_series
)

def test_normalize_country_name():
    assert normalize_country_name("UK") == "united_kingdom"
    assert normalize_country_name("United States") == "united_states"
    assert normalize_country_name("Deutschland") == "germany"
    assert normalize_country_name("  singapore  ") == "singapore"
    assert normalize_country_name(None) == "unknown"

def test_clean_transactions_dedup_and_nulls():
    sample_data = pd.DataFrame([
        # Valid row 1
        {"invoice_id": "INV1", "customer_id": "C1", "country": "uk", "stream_id": "s1", "price": 9.99, "views": 1, "date": "2020-01-01"},
        # Duplicate of row 1
        {"invoice_id": "INV1", "customer_id": "C1", "country": "uk", "stream_id": "s1", "price": 9.99, "views": 1, "date": "2020-01-01"},
        # Row with null price
        {"invoice_id": "INV2", "customer_id": "C2", "country": "uk", "stream_id": "s1", "price": None, "views": 1, "date": "2020-01-01"},
        # Row with negative price
        {"invoice_id": "INV3", "customer_id": "C3", "country": "uk", "stream_id": "s1", "price": -5.0, "views": 1, "date": "2020-01-01"},
        # Valid row 2
        {"invoice_id": "INV4", "customer_id": "C4", "country": "us", "stream_id": "s2", "price": 14.99, "views": 2, "date": "2020-01-02"}
    ])
    sample_data["date"] = pd.to_datetime(sample_data["date"])
    
    clean_df, stats = clean_transactions(sample_data)
    
    assert len(clean_df) == 2
    assert stats["duplicate_rows_dropped"] == 1
    assert stats["null_rows_dropped"] == 1
    assert stats["negative_rows_dropped"] == 1
    assert "INV1" in clean_df["invoice_id"].values
    assert "INV4" in clean_df["invoice_id"].values

def test_aggregate_daily_time_series_gap_filling():
    sample_df = pd.DataFrame([
        {"invoice_id": "INV1", "customer_id": "C1", "country": "uk", "stream_id": "s1", "price": 10.0, "views": 1, "date": pd.Timestamp("2020-01-01")},
        {"invoice_id": "INV2", "customer_id": "C2", "country": "uk", "stream_id": "s2", "price": 20.0, "views": 2, "date": pd.Timestamp("2020-01-03")}
    ])
    
    daily_df = aggregate_daily_time_series(sample_df)
    
    # Should contain country 'uk' and 'all'
    assert set(daily_df["country"].unique()) == {"uk", "all"}
    
    # Should have continuous days from 2020-01-01 to 2020-01-03 (3 days for uk + 3 days for all = 6 rows)
    assert len(daily_df) == 6
    
    uk_jan2 = daily_df[(daily_df["country"] == "uk") & (daily_df["date"] == "2020-01-02")]
    assert len(uk_jan2) == 1
    assert uk_jan2["revenue"].iloc[0] == 0.0
    assert uk_jan2["purchases"].iloc[0] == 0

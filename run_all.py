"""
run_all.py
----------
Orchestration runner for the entire AAVAIL Revenue Projection pipeline.
Executes:
1. Synthetic raw batch generation (multi-file, heterogeneous schemas, duplicates, missing data)
2. Ingestion pipeline (cleaning, deduplication, daily time-series aggregation)
3. Exploratory Data Analysis & Statistical divergence tests
4. Model training, Quantile Interval calibration, and Rolling-window Backtesting
5. Comprehensive pytest execution across all modules
"""

import sys
import subprocess
import json
import os

def print_header(title: str):
    print("\n" + "="*80)
    print(f"  {title.upper()}")
    print("="*80)

def main():
    # 1. Generate Raw Data
    print_header("Step 1: Generating Raw AAVAIL Multi-Batch Invoice Files")
    from generate_sample_data import generate_aavail_raw_data
    generate_aavail_raw_data(output_dir="data/raw", num_months=24)

    # 2. Ingest and Clean Data
    print_header("Step 2: Data Ingestion & Daily Time-Series Aggregation")
    from src.ingestion import run_ingestion_pipeline
    daily_df, ingest_stats = run_ingestion_pipeline()
    print("Ingestion Summary:")
    print(json.dumps(ingest_stats, indent=2))

    # 3. Exploratory Data Analysis
    print_header("Step 3: Exploratory Data Analysis (EDA)")
    from src.eda import run_eda
    eda_summary = run_eda(daily_df)
    print("Top Countries:", eda_summary["top_10_countries_by_revenue"])
    print("Statistical Heterogeneity Test:", json.dumps(eda_summary["statistical_tests"], indent=2))

    # 4. Training & Backtesting Pipeline
    print_header("Step 4: Model Training, Quantile Intervals & Time-Based Backtest")
    from src.pipeline import train_pipeline, predict_pipeline
    train_result = train_pipeline(confidence_level=0.90)
    print("Backtest Validation Performance:")
    print(json.dumps(train_result["backtest_results"]["models"], indent=2))
    print("Performance Gain over Baseline:", json.dumps(train_result["backtest_results"]["performance_gain"], indent=2))

    # 5. Sample Live Inference
    print_header("Step 5: Sample Country Inference")
    for country in ["united_kingdom", "united_states", "singapore", "all"]:
        pred = predict_pipeline(country=country)
        print(f"[{country.upper()}] 30-day Revenue Projection: ${pred['predicted_revenue_30d']:,.2f} (90% CI: [${pred['lower_bound']:,.2f} - ${pred['upper_bound']:,.2f}]) | Naive Baseline: ${pred['baseline_naive_30d']:,.2f} | Model: {pred['model_architecture']} | Latency: {pred['runtime_ms']}ms")

    # 6. Run Full Test Suite
    print_header("Step 6: Executing Full Pytest Suite")
    ret_code = subprocess.call([sys.executable, "-m", "pytest", "tests/", "-v"])
    
    if ret_code == 0:
        print_header("All Steps & Tests Completed Successfully!")
    else:
        print("\n[WARNING] Some unit tests failed. Check output above.")

if __name__ == "__main__":
    main()

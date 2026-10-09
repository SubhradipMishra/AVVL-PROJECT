# AAVAIL Revenue Projection Case Study: Strategic Plan & Technical Report

---

## Executive Summary

This project implements an end-to-end Machine Learning system for projecting country-level monthly revenue at **AAVAIL**, an international streaming entertainment platform transitioning from bundled subscriptions to an a la carte micro-transaction model.

The system is designed directly from two **Design-Thinking Hill Statements**, delivering both point estimates and calibrated prediction intervals while mitigating critical data leakage, sparsity, and distribution drift risks.

---

## 1. What the Hills Tell You

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             MANAGER HILL                                    │
│  "Target monthly revenue for a single country with prediction intervals"    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Requires
                                       ▼
                     [Prediction Interval Model Framework]
                     • Point estimate: Expected 30-day revenue
                     • Confidence bounds: 90% Calibrated Interval
                     • Unit: Per-Country Model with Pooled Fallback
                                       ▲
                                       │ Feeds from
┌──────────────────────────────────────┴──────────────────────────────────────┐
│                            END-USER HILL                                    │
│   "A la carte subscriptions mean revenue is the sum of micro-purchases"     │
└─────────────────────────────────────────────────────────────────────────────┘
                     • Daily aggregation across multi-batch JSON/CSV
                     • Ingestion cleanup: schema fixes, dedup, missing values
                     • Features: daily revenue, active users, stream views
```

### Manager Hill (The Business Target)
- **Target**: Cumulative revenue for a single country over the next month (30 days).
- **Output**: Point estimate along with empirical lower and upper bounds (e.g. 90% confidence intervals) to allow executive finance and operations teams to quantify upside potential and downside exposure.
- **Unit of Prediction**: Country-level time series. To balance high-volume markets with emerging markets, the system employs **dedicated gradient boosting models for top countries** and an **aggregate pooled model with country embeddings for sparse countries**.

### End-User Hill (The Data Challenge)
- **Granularity Shift**: Revenue is not a uniform recurring subscription fee; it is generated via independent streams, day passes, and tier upgrades.
- **Aggregation**: Invoices must be harmonized and aggregated into daily time series per country.
- **Signal Extraction**: Daily revenue alone is noisy; auxiliary features like active unique customers, purchase counts, stream view volumes, and customer spending velocity provide vital leading indicators.

---

## 2. Six-Phase Time-Boxed Workflow

### Phase 1: Business Opportunity & Success Metrics
- **Primary Success Metric**: 
  - **MAPE (Mean Absolute Percentage Error)**: Normalized error across countries of varying economic scale.
  - **RMSE (Root Mean Squared Error)**: Penalizes large projection errors on top revenue-driving countries.
- **Baseline Benchmark to Beat**: 
  - **30-Day Naive Persistence**: Using the preceding 30-day revenue sum as the projected 30-day revenue.
  - **Target Hurdle**: ML candidate models must achieve at least a **15–30% relative reduction in MAPE** compared to the naive persistence baseline on time-based validation.

### Phase 2: Data Ingestion & Harmonization (`src/ingestion.py`)
- **Heterogeneous Schemas**: Standardizes inconsistent legacy column names (`customer_ID` vs `customer_id`, `amount` vs `price`, `stream_type` vs `stream_id`).
- **Data Hygiene**:
  - Filters out anomalous negative transaction values (dirty refunds).
  - Imputes missing view counts and fills missing customer IDs with anonymous tokens.
  - Deduplicates transaction entries on `(invoice_id, customer_id, country, date, price)`.
- **Continuous Daily Aggregation**: Creates a gap-free continuous timeline for each country and computes a synthetic `"all"` global platform total.

### Phase 3: Exploratory Data Analysis & Country Heterogeneity (`src/eda.py`)
- **Top 10 Countries**: Identifies primary markets (`united_kingdom`, `united_states`, `singapore`, `germany`, `france`, etc.).
- **Seasonality & Trend**: Measures weekend surge ratios (+30% to +40% increase on Saturdays/Sundays) and annual trend slopes.
- **Statistical Divergence Test**: Runs **One-Way ANOVA** and **Kruskal-Wallis tests** across country distributions ($p < 0.001$), confirming statistically significant country differences that validate country-specific modeling.

### Phase 4: Feature Engineering & Modeling (`src/features.py`, `src/models.py`)
- **Feature Pipeline**:
  - Lags: 1, 7, 14, 21, 30 days for revenue, purchases, unique customers, and stream views.
  - Rolling Statistics: 7, 14, 30-day rolling means, standard deviations, minimums, and maximums.
  - Behavioral Ratios: Revenue per customer, views per customer, purchase frequency.
  - Calendar Dynamics: Day of week, day of month, month, weekend indicator, cyclical sin/cos encodings.
- **Model Candidates**:
  - `BaselineNaiveModel`: 30-day rolling persistence.
  - `PooledGradientBoostingModel`: Global gradient boosting on all countries.
  - `PooledRandomForestModel`: Bagged tree ensemble.
  - `MultiCountryModelManager`: Dedicated per-country gradient boosting models with pooled fallback for sparse markets.
- **Interval Calibration**: Quantile Gradient Boosting regressors (`alpha=0.05` and `alpha=0.95`) creating a 90% confidence interval.

### Phase 5: Deployment, Packaging & Observability (`app/main.py`, `src/monitoring.py`)
- **FastAPI Microservice**: High-performance asynchronous REST API (`/train`, `/predict`, `/logs`, `/drift`, `/eda`, `/health`).
- **Audit Logging**: Every inference request logs input arguments, point estimates, confidence intervals, model version, and latency in JSONL format.
- **Drift Monitoring**: Computes **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS) two-sample tests** to detect distribution shift between live predictions and historical baseline.
- **Containerization**: Packaged via `Dockerfile` and `docker-compose.yml`.

### Phase 6: Continuous Improvement & Diagnostics (`src/evaluate.py`)
- Automated model comparison on sequential out-of-time backtests.
- Interval coverage tracking (verifying empirical coverage approaches theoretical 90%).

---

## 3. Risk Matrix & Mitigation

| Risk | Severity | Root Cause | Implemented Technical Mitigation |
|---|---|---|---|
| **Data Leakage** | 🔴 Critical | Random train/test split mixing future dates with past dates | Strict time-based rolling split (`run_time_based_backtest`) where the model is trained strictly on $t < T_{\text{split}}$ and evaluated strictly on $t \ge T_{\text{split}}$. |
| **Sparse Countries** | 🟡 Medium | Small markets have sparse transactions leading to high variance | `MultiCountryModelManager` automatically defaults to the global pooled model with country indicators for countries with $<60$ days of history. |
| **Interval Miscalibration** | 🟡 Medium | Assuming Gaussian residuals on skewed revenue data | Non-parametric Quantile Regression (`loss='quantile'`) directly fitting 5th and 95th percentiles + boundary non-crossing constraints. |
| **Over-investing in EDA** | 🟢 Low | Spending excess time on visualizations instead of deployment | Time-boxed automated EDA script exporting structured JSON diagnostics and hypothesis test summaries in $<2$ seconds. |

---

## 4. Verification & Testing

The project includes unit and integration tests across all pipeline layers:
1. `tests/test_ingestion.py`: Schema harmonization, deduplication, date normalization, and gap-free aggregation.
2. `tests/test_model.py`: Estimators, non-crossing prediction intervals, and multi-country routing.
3. `tests/test_logger.py`: Structured JSONL logging and retrieval.
4. `tests/test_api.py`: FastAPI endpoints (`/health`, `/train`, `/predict`, `/logs`, `/drift`, `/eda`).

To run the complete suite:
```bash
python run_all.py
```

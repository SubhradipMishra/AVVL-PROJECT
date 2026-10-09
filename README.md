# AAVAIL Monthly Revenue Projection System

An enterprise-ready, time-boxed Machine Learning and MLOps system built for the **AAVAIL Revenue Projection Case Study**, addressing the core design-thinking Hill Statements.

---

## 🏔️ The Two Hill Statements

### 1. Manager Hill *(The Business Modeling Objective)*
> **Target**: Project total revenue for a single country over the next month (30 days).  
> **Output**: Point estimate + calibrated confidence measure (prediction intervals: 80% / 90% / 95%), providing actionable risk bounds for financial planning.  
> **Unit of Prediction**: Country-level (Dedicated Per-Country Model vs. Global Pooled Model with country encodings and sparse-country fallback).

### 2. End-User Hill *(The Data Engineering Challenge)*
> **Business Reality**: A la carte subscriptions mean revenue is the sum of thousands of small, independent stream purchases across heterogeneous batch sources.  
> **Solution**: Robust multi-file ingestion pipeline that standardizes column variations, eliminates duplicate invoices, cleans anomalous prices, and aggregates atomic transactions into a continuous daily per-country time series.  
> **Engineered Features**: Daily revenue, purchase volume, unique active customers, total stream views, and service engagement ratios.

---

## 🛠️ Architecture & Workflow (Time-Boxed 6 Phases)

```
                       [Raw Batch Logs (JSON / CSV)]
                                    │
                                    ▼
                      [src/ingestion.py Pipeline]
              (Schema Normalization, Dedup, Continuous Imputation)
                                    │
                                    ▼
                   [data/processed/daily_country_revenue.csv]
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                             ▼
       [src/eda.py]                               [src/features.py]
 (Top 10 Countries, Trend,                  (Lags: 1,7,14,21,30d; Rolling Stats;
 Seasonality, ANOVA/KW Tests)                Customer Ratios; Sin/Cos Calendar)
                                                           │
                                                           ▼
                                                  [src/evaluate.py]
                                            (Rolling Sequential Backtest:
                                             Strict Time Split, NO Leakage)
                                                           │
                                                           ▼
                                                  [src/models.py]
                                         (MultiCountryModelManager:
                                          Per-Country Gradient Boosting +
                                          Quantile Prediction Intervals)
                                                           │
                                                           ▼
                                            [FastAPI Deployment & Audit]
                                         (POST /train, POST /predict,
                                          GET /logs, GET /drift, GET /eda)
```

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run End-to-End Orchestration (Data Gen ➔ Ingest ➔ EDA ➔ Backtest ➔ Test Suite)
```bash
python run_all.py
```

### 3. Launch the API Microservice
```bash
uvicorn app.main:app --reload --port 8000
```
Interactive Swagger API documentation is available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Run Pytest Suite
```bash
pytest tests/ -v
```

### 5. Docker Deployment
```bash
docker build -t aavail-revenue-api .
docker run -p 8000:8000 aavail-revenue-api
# Or via docker-compose:
docker-compose up --build
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/train` | Triggers raw ingestion, feature creation, time-based backtest, and updates serialized model artifact. |
| `POST` | `/predict` | Generates 30-day ahead revenue projection + lower/upper prediction intervals for a country. |
| `GET` | `/logs` | Returns recent structured prediction logs with filtering options. |
| `GET` | `/drift` | Monitors data and prediction distribution drift using Population Stability Index (PSI) and KS-test. |
| `GET` | `/eda` | Returns statistical summaries and ANOVA country heterogeneity test results. |
| `GET` | `/health` | Service health status and model artifact status. |

### Sample Prediction Request
```json
POST /predict
{
  "country": "united_kingdom",
  "target_date": "2020-02-01",
  "days_ahead": 30
}
```

### Sample Response
```json
{
  "country": "united_kingdom",
  "target_date": "2020-02-01",
  "days_ahead": 30,
  "predicted_revenue_30d": 56420.75,
  "lower_bound": 51200.30,
  "upper_bound": 61850.10,
  "confidence_level": 0.90,
  "baseline_naive_30d": 49820.00,
  "model_architecture": "Dedicated Per-Country GB",
  "runtime_ms": 14.8
}
```

---

## 🛡️ Risk Mitigation Strategy

| Risk | Consequence | Implemented Solution |
|---|---|---|
| **Data Leakage** | Optimistic metrics failing in production | Strict sequential time-based splitting (`run_time_based_backtest`); zero random train/test splits. |
| **Sparse Countries** | Overfitting or failure on countries with few transactions | Hybrid modeling: dedicated models for high-volume countries, automatic fallback to global pooled model with country indicators for low-volume countries. |
| **Poorly Calibrated Intervals** | Misleading manager confidence | Quantile Loss Gradient Boosting (`alpha=0.05, 0.95`) combined with sanity boundaries ensuring non-crossing bounds and strictly positive revenue. |
| **Drift in Production** | Model degradation over time | Automated drift monitoring via PSI and Kolmogorov-Smirnov test on live prediction distributions. |

"""
generate_sample_data.py
-----------------------
Generates realistic multi-file raw transaction/invoice logs for the AAVAIL case study.
Simulates a real-world enterprise environment:
- Multiple batch files across months/years (JSON and CSV)
- Heterogeneous column schemas across legacy sources (e.g. 'customer_id' vs 'customer_ID', 'price' vs 'amount', 'stream_id' vs 'views')
- Occasional duplicate invoices, missing fields, negative outliers, and dirty strings
- Realistic time series behavior (trend, weekly seasonality, holiday/weekend surges, country differences)
"""

import os
import json
import random
import datetime
import pandas as pd
import numpy as np

def generate_aavail_raw_data(output_dir: str = "data/raw", num_months: int = 24, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Top countries and relative weightings
    countries_weights = {
        "united_kingdom": 0.28,
        "united_states": 0.22,
        "singapore": 0.12,
        "germany": 0.09,
        "france": 0.08,
        "eire": 0.06,
        "spain": 0.04,
        "netherlands": 0.04,
        "australia": 0.03,
        "belgium": 0.02,
        "switzerland": 0.01,
        "portugal": 0.01
    }
    country_list = list(countries_weights.keys())
    country_probs = list(countries_weights.values())

    # Stream services / products available a la carte
    stream_types = ["aavail_basic_stream", "aavail_premium_stream", "aavail_unlimited", "music_pass", "sports_tier"]
    stream_prices = {
        "aavail_basic_stream": [1.99, 2.99, 4.99],
        "aavail_premium_stream": [6.99, 9.99, 12.99],
        "aavail_unlimited": [14.99, 19.99],
        "music_pass": [0.99, 1.99],
        "sports_tier": [7.99, 14.99]
    }

    start_date = datetime.date(2018, 1, 1)
    
    # We will generate monthly file batches
    all_dates = [start_date + datetime.timedelta(days=i) for i in range(num_months * 30)]
    
    # Group dates by month
    month_groups = {}
    for d in all_dates:
        key = (d.year, d.month)
        if key not in month_groups:
            month_groups[key] = []
        month_groups[key].append(d)

    customer_pool = [f"CUST_{i:05d}" for i in range(1, 4000)]
    file_count = 0
    total_invoices = 0

    for (year, month), dates in month_groups.items():
        records = []
        # Decide schema variation for this batch (legacy vs new system)
        schema_variant = random.choice(["v1_json", "v2_json", "v1_csv"])
        
        for d in dates:
            # Base daily volume with trend and day-of-week seasonality (weekends have +30% volume)
            day_idx = (d - start_date).days
            growth_trend = 1.0 + (0.0008 * day_idx)  # Steady annual growth
            dow_multiplier = 1.35 if d.weekday() >= 5 else 1.0  # Weekend surge
            month_multiplier = 1.25 if d.month in [11, 12] else 1.0 # Q4 holiday surge
            
            num_txns = int(np.random.poisson(lam=120) * growth_trend * dow_multiplier * month_multiplier)

            for _ in range(num_txns):
                country = np.random.choice(country_list, p=country_probs)
                cust = random.choice(customer_pool)
                stype = random.choice(stream_types)
                price = random.choice(stream_prices[stype])
                views = int(np.random.geometric(p=0.4))
                invoice_id = f"INV_{d.strftime('%Y%m%d')}_{random.randint(10000, 99999)}"
                
                # Introduce occasional data dirtiness (<1.5% cases)
                rand_dirty = random.random()
                if rand_dirty < 0.005:
                    price = -price  # Dirty negative refund/anomaly
                elif rand_dirty < 0.01:
                    price = None   # Missing price
                
                rec = {
                    "invoice_id": invoice_id,
                    "customer_id": cust,
                    "country": country,
                    "stream_id": stype,
                    "price": price,
                    "views": views,
                    "year": d.year,
                    "month": d.month,
                    "day": d.day,
                    "invoice_date": d.isoformat()
                }
                records.append(rec)
                
                # Introduce duplicate invoice occasionally (~1% duplication)
                if random.random() < 0.01:
                    records.append(rec.copy())

        total_invoices += len(records)
        file_count += 1
        
        # Save according to schema variant
        filename_base = f"invoices_{year}_{month:02d}"
        
        if schema_variant == "v1_json":
            # Standard JSON
            path = os.path.join(output_dir, f"{filename_base}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        elif schema_variant == "v2_json":
            # JSON with alternate column naming (e.g. customer_ID, amount, country_code)
            renamed_records = []
            for r in records:
                renamed = {
                    "invoice": r["invoice_id"],
                    "customer_ID": r["customer_id"],
                    "country": r["country"].replace("_", " ").title() if random.random() < 0.3 else r["country"],
                    "stream_type": r["stream_id"],
                    "amount": r["price"],
                    "times_streamed": r["views"],
                    "year": r["year"],
                    "month": r["month"],
                    "day": r["day"]
                }
                renamed_records.append(renamed)
            path = os.path.join(output_dir, f"{filename_base}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(renamed_records, f, indent=2)
        else:
            # CSV with header
            df = pd.DataFrame(records)
            if random.random() < 0.5:
                df = df.rename(columns={"invoice_id": "invoice_num", "price": "revenue_amount", "views": "total_views"})
            path = os.path.join(output_dir, f"{filename_base}.csv")
            df.to_csv(path, index=False)

    print(f"[SUCCESS] Generated {file_count} raw batch files with {total_invoices} transactions in '{output_dir}'.")

if __name__ == "__main__":
    generate_aavail_raw_data()

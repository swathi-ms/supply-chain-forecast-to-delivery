# Forecast to Delivery: End-to-End Supply Chain Analytics

An end-to-end supply chain analytics project combining two public datasets to tell one story: **forecast demand → replenish inventory → ship it → deliver on time** — measured end-to-end with a blended Perfect Order Rate KPI.

**Live dashboard:** https://supply-chain-forecast-to-delivery-els8vwfdswogwh5xh8yfgc.streamlit.app/

## Data sources
- [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) (Kaggle) — 73,100 daily records, 5 stores x 20 products, Jan 2022–Jan 2024
- [Smart Logistics Supply Chain Dataset](https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset) (Kaggle) — 1,000 real-time truck shipment records, 10 trucks, 2024

## Key findings
- **43% on-time delivery rate overall**, dropping to **0%** the moment traffic conditions turn "Heavy" — the single strongest delay driver.
- **38% blended Perfect Order Rate** end-to-end — inventory availability isn't the bottleneck (86% stockout-free), on-time delivery is.
- **Data quality catch:** the retail dataset's own "Demand Forecast" column correlates 0.997 with actual sales — flagged as leakage rather than used as a benchmark. A fair baseline (7-day moving average) was built instead, and a Random Forest model beat it by ~5% on WAPE.
- A Random Forest delay-risk classifier predicts shipment delays with 76% accuracy.

## Tech stack
Python (pandas, scikit-learn) for data cleaning, feature engineering, and modeling · Streamlit + Plotly for the interactive dashboard · also built in Excel, Tableau, and Power BI as parallel versions of the same analysis (see `docs/`).

## Project structure
```
phase1_pipeline.py     # Cleans both datasets, engineers KPIs, trains 2 models, builds the end-to-end bridge table
streamlit_app.py       # Interactive dashboard - 6 KPI cards, 4 tabs, 12 charts, live filters
requirements.txt       # Python dependencies
data_raw/               # Place the two downloaded Kaggle CSVs here (not committed - see .gitignore)
data_processed/         # Output of phase1_pipeline.py - streamlit_app.py reads from here (committed so the app runs out of the box)
```

## Run it locally
1. Download the two datasets from Kaggle (links above) into `data_raw/`:
   - `data_raw/retail_store_inventory.csv`
   - `data_raw/smart_logistics_dataset.csv`
2. Install dependencies: `pip install -r requirements.txt`
3. (Optional - `data_processed/` is already included) Regenerate the processed data:
   ```
   python phase1_pipeline.py --retail_csv data_raw/retail_store_inventory.csv --logistics_csv data_raw/smart_logistics_dataset.csv --out_dir data_processed
   ```
4. Launch the dashboard: `streamlit run streamlit_app.py`

## A note on methodology (read before presenting this anywhere)
The two source datasets share no common key. The end-to-end "bridge" table pairs each monthly replenishment event with a randomly sampled logistics shipment to illustrate the Perfect Order Rate concept — a disclosed, simulated link for portfolio storytelling, not a real transactional join. Full detail is in the code comments in `phase1_pipeline.py`.

## Author
Swathi Munikoti

I used Claude as a development tool to help build it, while the analysis, sourcing logic, and interpretation are my own.
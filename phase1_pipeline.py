"""
phase1_pipeline.py
===================
End-to-end data pipeline for the "Forecast to Delivery" supply chain portfolio
project. Cleans both source datasets, engineers KPIs, trains two models
(demand forecast + delivery delay risk), and builds a simulated end-to-end
bridge table with a Perfect Order Rate metric.

USAGE
-----
    python phase1_pipeline.py \
        --retail_csv data_raw/retail_store_inventory.csv \
        --logistics_csv data_raw/smart_logistics_dataset.csv \
        --out_dir data_processed

Source datasets (download from Kaggle first):
  - Retail Store Inventory Forecasting Dataset (anirudhchauhan)
    https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset
  - Smart Logistics Supply Chain Dataset (ziya07)
    https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset

OUTPUT
------
All files are written to --out_dir as CSVs, ready to be consumed by the
Streamlit app (streamlit_app.py), Tableau, Power BI, or Excel.

DATA QUALITY NOTES (read before presenting results anywhere)
--------------------------------------------------------------
1. The retail dataset's own "Demand Forecast" column correlates 0.997 with
   actual "Units Sold" - it is actual sales plus small random noise, not an
   independently generated forecast. It is NOT used as a benchmark here.
   A genuine baseline (trailing 7-day moving average) is used instead.
2. Category and Region are not fixed attributes of a given Store-Product pair
   in the retail dataset - they vary row to row. Aggregations that need a
   single label per Store-Product-Month use the most frequent (mode) value.
3. The retail dataset (73,100 daily rows, 5 stores x 20 products, 2022-2024)
   and the logistics dataset (1,000 shipments, 10 trucks, 2024) share no
   common key. The end-to-end "bridge" table pairs monthly replenishment
   events with a randomly sampled logistics shipment to illustrate a
   Perfect Order Rate KPI. This is a disclosed, simulated link, not a real
   join - say so in any writeup or interview.
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

RANDOM_STATE = 42


def wape(actual, forecast):
    """Weighted Absolute Percentage Error - standard supply-chain forecast
    metric, more robust than MAPE on low-volume days (avoids division blowups
    near zero)."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return np.sum(np.abs(actual - forecast)) / np.sum(actual) * 100


# ---------------------------------------------------------------------------
# STEP 1: Clean both datasets and engineer row-level KPIs
# ---------------------------------------------------------------------------
def clean_retail(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store ID', 'Product ID', 'Date']).reset_index(drop=True)

    df['Revenue'] = df['Units Sold'] * df['Price'] * (1 - df['Discount'] / 100)

    df['Forecast_Error'] = df['Demand Forecast'] - df['Units Sold']
    df['Forecast_Abs_Error'] = df['Forecast_Error'].abs()
    df['Forecast_APE'] = np.where(df['Units Sold'] > 0,
                                   df['Forecast_Abs_Error'] / df['Units Sold'], np.nan)

    df['Stockout_Occurred'] = (df['Units Sold'] >= df['Inventory Level']).astype(int)
    df['Stockout_Risk'] = (df['Demand Forecast'] > df['Inventory Level']).astype(int)

    df['Avg_Daily_Sales_7d'] = (
        df.groupby(['Store ID', 'Product ID'])['Units Sold']
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
    df['Days_of_Supply'] = np.where(df['Avg_Daily_Sales_7d'] > 0,
                                     df['Inventory Level'] / df['Avg_Daily_Sales_7d'], np.nan)
    df['Overstock_Flag'] = (df['Days_of_Supply'] > 30).astype(int)

    df['Price_Gap_vs_Competitor'] = df['Price'] - df['Competitor Pricing']
    df['Priced_Above_Competitor'] = (df['Price_Gap_vs_Competitor'] > 0).astype(int)

    # Fair, non-leaky baseline forecast for later model comparison
    df['Naive_MA7_Forecast'] = (
        df.groupby(['Store ID', 'Product ID'])['Units Sold']
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    df['Naive_MA7_Forecast'] = df['Naive_MA7_Forecast'].fillna(df['Units Sold'].mean())

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    df['DayOfWeek'] = df['Date'].dt.day_name()
    return df


def clean_logistics(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # The literal string "None" gets parsed as NaN by pandas - restore it
    df['Logistics_Delay_Reason'] = df['Logistics_Delay_Reason'].fillna('None')

    df['On_Time_Flag'] = np.where(df['Logistics_Delay'] == 0, 1, 0)
    df['Delivered_Flag'] = (df['Shipment_Status'] == 'Delivered').astype(int)

    df['Date'] = df['Timestamp'].dt.date
    df['Month'] = df['Timestamp'].dt.to_period('M').astype(str)
    df['Hour'] = df['Timestamp'].dt.hour
    return df


# ---------------------------------------------------------------------------
# STEP 2: Aggregated KPI tables (dashboard-ready)
# ---------------------------------------------------------------------------
def build_retail_kpis(retail):
    daily = (
        retail.groupby(['Date', 'Region', 'Category'])
        .agg(Units_Sold=('Units Sold', 'sum'), Units_Ordered=('Units Ordered', 'sum'),
             Revenue=('Revenue', 'sum'), Avg_Inventory=('Inventory Level', 'mean'),
             Forecast_APE=('Forecast_APE', 'mean'), Stockout_Rate=('Stockout_Occurred', 'mean'),
             Overstock_Rate=('Overstock_Flag', 'mean'), Avg_Days_of_Supply=('Days_of_Supply', 'mean'),
             Promo_Share=('Holiday/Promotion', 'mean'))
        .reset_index()
    )

    monthly = (
        retail.groupby(['Month', 'Region', 'Category'])
        .agg(Units_Sold=('Units Sold', 'sum'), Revenue=('Revenue', 'sum'),
             Forecast_MAPE=('Forecast_APE', 'mean'), Stockout_Rate=('Stockout_Occurred', 'mean'),
             Avg_Days_of_Supply=('Days_of_Supply', 'mean'))
        .reset_index()
    )

    monthly_overall = retail.groupby('Month').apply(lambda g: pd.Series({
        'Revenue': g['Revenue'].sum(),
        'Units_Sold': g['Units Sold'].sum(),
        'Forecast_WAPE_Pct': g['Forecast_Abs_Error'].sum() / g['Units Sold'].sum() * 100,
        'Stockout_Rate_Pct': g['Stockout_Occurred'].mean() * 100,
        'Avg_Days_of_Supply': g['Days_of_Supply'].mean(),
    })).reset_index()

    category_summary = (
        retail.groupby('Category')
        .agg(Total_Revenue=('Revenue', 'sum'), Total_Units_Sold=('Units Sold', 'sum'),
             Forecast_MAPE=('Forecast_APE', 'mean'), Stockout_Rate=('Stockout_Occurred', 'mean'),
             Overstock_Rate=('Overstock_Flag', 'mean'), Avg_Days_of_Supply=('Days_of_Supply', 'mean'))
        .reset_index().sort_values('Total_Revenue', ascending=False)
    )

    promo_lift = (
        retail.groupby(['Category', 'Holiday/Promotion'])['Units Sold']
        .mean().unstack().rename(columns={0: 'Avg_Units_No_Promo', 1: 'Avg_Units_With_Promo'})
    )
    promo_lift['Promo_Lift_Pct'] = (
        (promo_lift['Avg_Units_With_Promo'] - promo_lift['Avg_Units_No_Promo'])
        / promo_lift['Avg_Units_No_Promo'] * 100
    )
    promo_lift = promo_lift.reset_index()

    return daily, monthly, monthly_overall, category_summary, promo_lift


def build_logistics_kpis(log):
    asset_kpi = (
        log.groupby('Asset_ID')
        .agg(Shipments=('Asset_ID', 'count'), On_Time_Rate=('On_Time_Flag', 'mean'),
             Avg_Waiting_Time=('Waiting_Time', 'mean'), Avg_Asset_Utilization=('Asset_Utilization', 'mean'),
             Delivered_Rate=('Delivered_Flag', 'mean'))
        .reset_index().sort_values('On_Time_Rate')
    )

    traffic_kpi = (
        log.groupby('Traffic_Status')
        .agg(Shipments=('Traffic_Status', 'count'), On_Time_Rate=('On_Time_Flag', 'mean'),
             Avg_Waiting_Time=('Waiting_Time', 'mean'))
        .reset_index()
    )

    delay_reason_kpi = (
        log[log['Logistics_Delay'] == 1]['Logistics_Delay_Reason']
        .value_counts(normalize=True).mul(100).round(1).reset_index()
    )
    delay_reason_kpi.columns = ['Delay_Reason', 'Share_of_Delays_Pct']

    monthly_otd = (
        log.groupby('Month')
        .agg(Shipments=('Month', 'count'), On_Time_Rate=('On_Time_Flag', 'mean'),
             Avg_Waiting_Time=('Waiting_Time', 'mean'))
        .reset_index()
    )

    return asset_kpi, traffic_kpi, delay_reason_kpi, monthly_otd


# ---------------------------------------------------------------------------
# STEP 3: Models
# ---------------------------------------------------------------------------
def train_forecast_model(retail):
    feature_cols_num = ['Price', 'Discount', 'Competitor Pricing', 'Holiday/Promotion']
    feature_cols_cat = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition',
                         'Seasonality', 'DayOfWeek']
    target = 'Units Sold'

    X = retail[feature_cols_num + feature_cols_cat].copy()
    X['Month_Num'] = pd.to_datetime(retail['Month']).dt.month if False else retail['Date'].dt.month
    feature_cols_num = feature_cols_num + ['Month_Num']
    y = retail[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    preprocess = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), feature_cols_cat)],
                                    remainder='passthrough')
    model = Pipeline([('prep', preprocess),
                       ('rf', RandomForestRegressor(n_estimators=150, max_depth=12,
                                                     random_state=RANDOM_STATE, n_jobs=-1))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    test_idx = X_test.index
    mae_model = mean_absolute_error(y_test, pred)
    wape_model = wape(y_test, pred)

    naive_pred = retail.loc[test_idx, 'Naive_MA7_Forecast']
    mae_naive = mean_absolute_error(y_test, naive_pred)
    wape_naive = wape(y_test, naive_pred)

    comparison = pd.DataFrame({
        'Metric': ['MAE', 'WAPE (%)'],
        'Naive_MovingAvg_Baseline': [mae_naive, wape_naive],
        'RandomForest_Model': [mae_model, wape_model],
    })

    ohe = model.named_steps['prep'].named_transformers_['cat']
    cat_names = ohe.get_feature_names_out(feature_cols_cat)
    all_names = list(cat_names) + feature_cols_num
    importances = model.named_steps['rf'].feature_importances_
    fi = pd.DataFrame({'Feature': all_names, 'Importance': importances}).sort_values(
        'Importance', ascending=False)

    return comparison, fi


def train_delay_model(log):
    feat_num = ['Temperature', 'Humidity', 'Waiting_Time', 'Asset_Utilization',
                'User_Transaction_Amount', 'User_Purchase_Frequency', 'Inventory_Level']
    feat_cat = ['Traffic_Status', 'Asset_ID']
    target = 'Logistics_Delay'

    X = log[feat_num + feat_cat]
    y = log[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                          random_state=RANDOM_STATE, stratify=y)
    preprocess = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), feat_cat)],
                                    remainder='passthrough')
    clf = Pipeline([('prep', preprocess),
                     ('rf', RandomForestClassifier(n_estimators=200, max_depth=8,
                                                    random_state=RANDOM_STATE, n_jobs=-1))])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    metrics = pd.DataFrame({'Metric': ['Accuracy', 'F1_Score'], 'Value': [acc, f1]})

    ohe = clf.named_steps['prep'].named_transformers_['cat']
    cat_names = ohe.get_feature_names_out(feat_cat)
    all_names = list(cat_names) + feat_num
    importances = clf.named_steps['rf'].feature_importances_
    fi = pd.DataFrame({'Feature': all_names, 'Importance': importances}).sort_values(
        'Importance', ascending=False)

    # Score every shipment for dashboard use
    log = log.copy()
    log['Delay_Risk_Score'] = clf.predict_proba(X)[:, 1]

    return metrics, fi, log


# ---------------------------------------------------------------------------
# STEP 4: Simulated end-to-end bridge + Perfect Order Rate
# ---------------------------------------------------------------------------
def build_bridge(retail, log, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    mode_first = lambda s: s.mode().iloc[0]

    replen = (
        retail.groupby(['Month', 'Store ID', 'Product ID'])
        .agg(Category=('Category', mode_first), Region=('Region', mode_first),
             Units_Ordered=('Units Ordered', 'sum'), Stockout_Occurred=('Stockout_Occurred', 'max'),
             Stockout_Risk=('Stockout_Risk', 'max'), Forecast_APE=('Forecast_APE', 'mean'))
        .reset_index()
    )

    sampled_idx = rng.integers(0, len(log), size=len(replen))
    shipment_cols = ['Asset_ID', 'Shipment_Status', 'Traffic_Status', 'Waiting_Time',
                      'Logistics_Delay', 'Logistics_Delay_Reason', 'On_Time_Flag', 'Delay_Risk_Score']
    shipment_cols = [c for c in shipment_cols if c in log.columns]
    shipment_sample = log.iloc[sampled_idx][shipment_cols].reset_index(drop=True)

    bridge = pd.concat([replen, shipment_sample], axis=1)
    bridge = bridge.rename(columns={'On_Time_Flag': 'Delivered_On_Time'})
    bridge['Perfect_Order'] = ((bridge['Stockout_Occurred'] == 0) &
                                (bridge['Delivered_On_Time'] == 1)).astype(int)

    by_category = (
        bridge.groupby('Category')
        .agg(Replenishment_Events=('Perfect_Order', 'count'), Perfect_Order_Rate=('Perfect_Order', 'mean'),
             Stockout_Free_Rate=('Stockout_Occurred', lambda s: 1 - s.mean()),
             On_Time_Rate=('Delivered_On_Time', 'mean'))
        .reset_index()
    )
    for c in ['Perfect_Order_Rate', 'Stockout_Free_Rate', 'On_Time_Rate']:
        by_category[c] *= 100

    return bridge, by_category


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Phase 1 pipeline: clean, engineer, model, bridge.')
    parser.add_argument('--retail_csv', default='data_raw/retail_store_inventory.csv')
    parser.add_argument('--logistics_csv', default='data_raw/smart_logistics_dataset.csv')
    parser.add_argument('--out_dir', default='data_processed')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading and cleaning retail dataset...')
    retail = clean_retail(args.retail_csv)
    print('Loading and cleaning logistics dataset...')
    log = clean_logistics(args.logistics_csv)

    print('Building retail KPI tables...')
    daily, monthly, monthly_overall, category_summary, promo_lift = build_retail_kpis(retail)

    print('Building logistics KPI tables...')
    asset_kpi, traffic_kpi, delay_reason_kpi, monthly_otd = build_logistics_kpis(log)

    print('Training demand forecast model (this can take ~30-60s)...')
    forecast_comparison, forecast_fi = train_forecast_model(retail)

    print('Training delay prediction model...')
    delay_metrics, delay_fi, log = train_delay_model(log)

    print('Building simulated end-to-end bridge and Perfect Order Rate...')
    bridge, perfect_order_by_category = build_bridge(retail, log)

    # Persist everything
    outputs = {
        'retail_cleaned.csv': retail,
        'logistics_cleaned.csv': log,
        'retail_kpi_daily.csv': daily,
        'retail_kpi_monthly.csv': monthly,
        'retail_monthly_overall.csv': monthly_overall,
        'retail_category_summary.csv': category_summary,
        'retail_promo_lift.csv': promo_lift,
        'logistics_asset_kpi.csv': asset_kpi,
        'logistics_traffic_kpi.csv': traffic_kpi,
        'logistics_delay_reason_kpi.csv': delay_reason_kpi,
        'logistics_monthly_otd.csv': monthly_otd,
        'model_forecast_comparison.csv': forecast_comparison,
        'model_forecast_feature_importance.csv': forecast_fi,
        'model_delay_metrics.csv': delay_metrics,
        'model_delay_feature_importance.csv': delay_fi,
        'bridge_perfect_order.csv': bridge,
        'perfect_order_by_category.csv': perfect_order_by_category,
    }
    for filename, df in outputs.items():
        df.to_csv(os.path.join(args.out_dir, filename), index=False)

    print(f'\nDone. {len(outputs)} files written to {args.out_dir}/')
    print('\n--- Headline numbers ---')
    print(f"Model forecast WAPE: {forecast_comparison.loc[1, 'RandomForest_Model']:.1f}% "
          f"(vs {forecast_comparison.loc[1, 'Naive_MovingAvg_Baseline']:.1f}% naive baseline)")
    print(f"Delay model accuracy: {delay_metrics.loc[0, 'Value']:.1%}")
    print(f"Overall on-time delivery rate: {log['On_Time_Flag'].mean():.1%}")
    print(f"Blended Perfect Order Rate: {bridge['Perfect_Order'].mean():.1%}")


if __name__ == '__main__':
    main()

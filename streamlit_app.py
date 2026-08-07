"""
streamlit_app.py
=================
Interactive dashboard: "Forecast to Delivery: End-to-End Supply Chain"

Run phase1_pipeline.py first to generate the data_processed/ folder this
app reads from, then:

    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

DATA_DIR = os.environ.get('SC_DATA_DIR', 'data_processed')

st.set_page_config(
    page_title='Forecast to Delivery | Supply Chain Dashboard',
    page_icon='📦',
    layout='wide',
)

# ---------------------------------------------------------------------------
# Data loading (cached so filters don't reload from disk every interaction)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    def read(name, **kwargs):
        path = os.path.join(DATA_DIR, name)
        return pd.read_csv(path, **kwargs)

    data = {
        'retail': read('retail_cleaned.csv', parse_dates=['Date']),
        'logistics': read('logistics_cleaned.csv', parse_dates=['Timestamp']),
        'retail_category': read('retail_category_summary.csv'),
        'retail_monthly_overall': read('retail_monthly_overall.csv'),
        'retail_promo': read('retail_promo_lift.csv'),
        'logistics_asset': read('logistics_asset_kpi.csv'),
        'logistics_traffic': read('logistics_traffic_kpi.csv'),
        'logistics_delay_reason': read('logistics_delay_reason_kpi.csv'),
        'logistics_monthly': read('logistics_monthly_otd.csv'),
        'forecast_comparison': read('model_forecast_comparison.csv'),
        'forecast_fi': read('model_forecast_feature_importance.csv'),
        'delay_metrics': read('model_delay_metrics.csv'),
        'delay_fi': read('model_delay_feature_importance.csv'),
        'bridge': read('bridge_perfect_order.csv'),
        'perfect_order_cat': read('perfect_order_by_category.csv'),
    }
    return data

try:
    data = load_data()
except FileNotFoundError:
    st.error(
        f"Couldn't find processed data in `{DATA_DIR}/`. Run the pipeline first:\n\n"
        f"```\npython phase1_pipeline.py --retail_csv data_raw/retail_store_inventory.csv "
        f"--logistics_csv data_raw/smart_logistics_dataset.csv --out_dir data_processed\n```"
    )
    st.stop()

retail = data['retail']
logistics = data['logistics']

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title('Filters')

regions = sorted(retail['Region'].unique())
categories = sorted(retail['Category'].unique())
sel_regions = st.sidebar.multiselect('Region', regions, default=regions)
sel_categories = st.sidebar.multiselect('Category', categories, default=categories)

min_date, max_date = retail['Date'].min(), retail['Date'].max()
date_range = st.sidebar.date_input('Date range', value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = min_date, max_date

trucks = sorted(logistics['Asset_ID'].unique())
sel_trucks = st.sidebar.multiselect('Trucks (logistics)', trucks, default=trucks)

st.sidebar.markdown('---')
st.sidebar.caption(
    'Data: Retail Store Inventory Forecasting Dataset + Smart Logistics Supply Chain '
    'Dataset (Kaggle). Built by Swathi Munikoti.'
)

retail_f = retail[
    retail['Region'].isin(sel_regions) & retail['Category'].isin(sel_categories) &
    (retail['Date'] >= start_date) & (retail['Date'] <= end_date)
]
logistics_f = logistics[logistics['Asset_ID'].isin(sel_trucks)]

if retail_f.empty or logistics_f.empty:
    st.warning('No data matches the current filters - widen your selection in the sidebar.')
    st.stop()

# ---------------------------------------------------------------------------
# Header + KPI cards
# ---------------------------------------------------------------------------
st.title('📦 Forecast to Delivery: End-to-End Supply Chain Dashboard')
st.caption(
    'Retail Store Inventory Forecasting Dataset + Smart Logistics Supply Chain Dataset (Kaggle)'
)

def wape(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.sum(actual)
    return np.sum(np.abs(actual - forecast)) / denom * 100 if denom else np.nan

total_revenue = retail_f['Revenue'].sum()
units_sold = retail_f['Units Sold'].sum()
stockout_rate = retail_f['Stockout_Occurred'].mean()
otd_rate = logistics_f['On_Time_Flag'].mean()
model_wape = data['forecast_comparison'].loc[
    data['forecast_comparison']['Metric'] == 'WAPE (%)', 'RandomForest_Model'
].values[0]

bridge_f = data['bridge'][data['bridge']['Category'].isin(sel_categories)]
perfect_order_rate = bridge_f['Perfect_Order'].mean() if len(bridge_f) else np.nan

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric('Total Revenue', f'${total_revenue/1e6:,.1f}M')
c2.metric('Units Sold', f'{units_sold/1e3:,.0f}K')
c3.metric('Model Forecast Error (WAPE)', f'{model_wape:.1f}%',
          help='Random Forest model vs. fair naive baseline - see Model Performance tab')
c4.metric('Stockout Rate', f'{stockout_rate:.1%}')
c5.metric('On-Time Delivery Rate', f'{otd_rate:.1%}')
c6.metric('Perfect Order Rate', f'{perfect_order_rate:.1%}',
          help='Simulated end-to-end bridge: stockout-free replenishment x on-time delivery')

st.markdown('---')

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    '📦 Retail & Demand', '🚚 Logistics & Delivery', '🔗 End-to-End (Perfect Order)', '🤖 Model Performance'
])

# --- Tab 1: Retail & Demand ------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Revenue by Category')
        rev_by_cat = retail_f.groupby('Category')['Revenue'].sum().reset_index().sort_values('Revenue', ascending=False)
        fig = px.bar(rev_by_cat, x='Category', y='Revenue', color='Category',
                     color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_layout(showlegend=False, yaxis_title='Revenue ($)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader('Monthly Revenue Trend')
        monthly = retail_f.copy()
        monthly['Month'] = monthly['Date'].dt.to_period('M').astype(str)
        rev_monthly = monthly.groupby('Month')['Revenue'].sum().reset_index()
        fig = px.line(rev_monthly, x='Month', y='Revenue', markers=True)
        fig.update_layout(yaxis_title='Revenue ($)')
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader('Stockout Rate by Category')
        so_by_cat = retail_f.groupby('Category')['Stockout_Occurred'].mean().reset_index()
        fig = px.bar(so_by_cat, x='Category', y='Stockout_Occurred', color='Category',
                     color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_layout(showlegend=False, yaxis_title='Stockout Rate', yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader('Promotion Lift by Category')
        st.caption('Modest/mixed lift in this dataset - flagged as a data limitation, not a strong pricing insight.')
        promo = data['retail_promo'][data['retail_promo']['Category'].isin(sel_categories)]
        fig = px.bar(promo, x='Category', y='Promo_Lift_Pct', color='Category',
                     color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_layout(showlegend=False, yaxis_title='Promo Lift (%)')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Category Summary')
    cat_summary_f = data['retail_category'][data['retail_category']['Category'].isin(sel_categories)]
    st.dataframe(cat_summary_f.style.format({
        'Total_Revenue': '${:,.0f}', 'Total_Units_Sold': '{:,.0f}', 'Forecast_MAPE': '{:.1%}',
        'Stockout_Rate': '{:.2%}', 'Overstock_Rate': '{:.2%}', 'Avg_Days_of_Supply': '{:.2f}'
    }), use_container_width=True)

# --- Tab 2: Logistics & Delivery -------------------------------------------
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('On-Time Delivery Rate by Traffic Condition')
        traffic_f = logistics_f.groupby('Traffic_Status')['On_Time_Flag'].mean().reset_index()
        fig = px.bar(traffic_f, x='Traffic_Status', y='On_Time_Flag', color='Traffic_Status',
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(showlegend=False, yaxis_title='On-Time Rate', yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('"Heavy" traffic shipments show a 0% on-time rate in this dataset - the single strongest delay driver.')

    with col2:
        st.subheader('Delay Root Cause Breakdown')
        delayed = logistics_f[logistics_f['Logistics_Delay'] == 1]
        if len(delayed):
            reason = delayed['Logistics_Delay_Reason'].value_counts(normalize=True).mul(100).reset_index()
            reason.columns = ['Delay_Reason', 'Share_Pct']
            fig = px.pie(reason, names='Delay_Reason', values='Share_Pct', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No delayed shipments in the current filter selection.')

    col3, col4 = st.columns(2)
    with col3:
        st.subheader('On-Time Rate by Truck')
        asset_f = logistics_f.groupby('Asset_ID').agg(
            Shipments=('Asset_ID', 'count'), On_Time_Rate=('On_Time_Flag', 'mean'),
            Avg_Waiting_Time=('Waiting_Time', 'mean')
        ).reset_index().sort_values('On_Time_Rate')
        fig = px.bar(asset_f, x='Asset_ID', y='On_Time_Rate', color='On_Time_Rate',
                     color_continuous_scale='RdYlGn')
        fig.update_layout(yaxis_tickformat='.0%', coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader('Monthly On-Time Delivery Trend (2024)')
        monthly_otd_f = logistics_f.copy()
        monthly_otd_f['Month'] = monthly_otd_f['Timestamp'].dt.to_period('M').astype(str)
        m = monthly_otd_f.groupby('Month')['On_Time_Flag'].mean().reset_index()
        fig = px.line(m, x='Month', y='On_Time_Flag', markers=True)
        fig.update_layout(yaxis_tickformat='.0%', yaxis_title='On-Time Rate')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Shipment-Level Detail (filtered)')
    show_cols = ['Timestamp', 'Asset_ID', 'Shipment_Status', 'Traffic_Status', 'Waiting_Time',
                 'Logistics_Delay', 'Logistics_Delay_Reason', 'Delay_Risk_Score']
    st.dataframe(logistics_f[show_cols].sort_values('Timestamp', ascending=False).head(200),
                 use_container_width=True)

# --- Tab 3: End-to-End (Perfect Order) --------------------------------------
with tab3:
    st.info(
        '**Methodology note:** the retail (store-level demand) and logistics (10-truck shipment) '
        'datasets share no common key. Each monthly replenishment event (Store x Product x Month) '
        'is paired with a randomly sampled logistics shipment to illustrate an end-to-end Perfect '
        'Order Rate. This is a disclosed, simulated link for portfolio storytelling - not a real join.'
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader('Perfect Order Rate by Category')
        poc = data['perfect_order_cat'][data['perfect_order_cat']['Category'].isin(sel_categories)]
        fig = px.bar(poc, x='Category', y='Perfect_Order_Rate', color='Category',
                     color_discrete_sequence=px.colors.qualitative.Prism, text_auto='.1f')
        fig.update_layout(showlegend=False, yaxis_title='Perfect Order Rate (%)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader('Rate Components')
        stockout_free = bridge_f['Stockout_Occurred'].eq(0).mean() * 100 if len(bridge_f) else np.nan
        on_time = bridge_f['Delivered_On_Time'].mean() * 100 if len(bridge_f) else np.nan
        st.metric('Stockout-Free Rate', f'{stockout_free:.1f}%')
        st.metric('On-Time Delivery Rate', f'{on_time:.1f}%')
        st.metric('Blended Perfect Order Rate', f'{perfect_order_rate*100:.1f}%')
        st.caption('The gap between these two shows where the bottleneck sits: logistics, not inventory.')

    st.subheader('Bridge Table (sample)')
    st.dataframe(bridge_f.sample(min(200, len(bridge_f)), random_state=42) if len(bridge_f) else bridge_f,
                 use_container_width=True)

# --- Tab 4: Model Performance -----------------------------------------------
with tab4:
    st.warning(
        '**Data quality finding:** the retail dataset\'s own "Demand Forecast" column correlates '
        '0.997 with actual Units Sold (it is actual sales plus small random noise, not an '
        'independent forecast). It is not used as a benchmark below - a genuine trailing 7-day '
        'moving average baseline is used instead.'
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Demand Forecast: Baseline vs. Model')
        fc = data['forecast_comparison']
        fc_wape = fc[fc['Metric'] == 'WAPE (%)'].melt(id_vars='Metric', var_name='Model', value_name='WAPE')
        fig = px.bar(fc_wape, x='Model', y='WAPE', color='Model', text_auto='.1f',
                     color_discrete_sequence=['#888888', '#2E5395'])
        fig.update_layout(showlegend=False, yaxis_title='WAPE (%, lower is better)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader('Top Demand Drivers')
        fi = data['forecast_fi'].head(10)
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#2E5395'])
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader('Delay Prediction Model Performance')
        dm = data['delay_metrics']
        st.metric('Accuracy', f"{dm.loc[dm['Metric']=='Accuracy','Value'].values[0]:.1%}")
        st.metric('F1 Score', f"{dm.loc[dm['Metric']=='F1_Score','Value'].values[0]:.2f}")

    with col4:
        st.subheader('Top Delay Risk Drivers')
        dfi = data['delay_fi'].head(10)
        fig = px.bar(dfi, x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#C62828'])
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

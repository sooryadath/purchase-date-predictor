import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="Customer Dashboard", layout="wide")

# Main Title
st.title("🛍️ Customer Purchase Date Predictor & Sales Dashboard")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Bill date'] = pd.to_datetime(df['Bill date'])
    df['Year'] = df['Bill date'].dt.year
    
    # SECTION 1: SALES DASHBOARD
    st.header("📊 Sales Dashboard")
    
    st.sidebar.header("📅 Filter by Date Range")
    min_date = df['Bill date'].min()
    max_date = df['Bill date'].max()
    start_date, end_date = st.sidebar.date_input("Select date range", [min_date, max_date])
    
    # Filter data based on selected range
    filtered_df = df[(df['Bill date'] >= pd.to_datetime(start_date)) & (df['Bill date'] <= pd.to_datetime(end_date))]
    
    # --- Top Customer Info ---
    top_customer_df = filtered_df.groupby('Customer Name')['Bill Qty'].sum().reset_index()
    top_customer_df = top_customer_df.sort_values('Bill Qty', ascending=False)
    top_customer = top_customer_df.iloc[0] if not top_customer_df.empty else {"Customer Name": "N/A", "Bill Qty": 0}
    top_10 = top_customer_df.head(10)

    # Metric cards for top customer summary
    st.subheader("🏆 Top Customer Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
            <div style="font-size:14px; color:gray;">Top Customer</div>
            <div style="font-size:20px; font-weight:bold; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 300px;">
                {top_customer['Customer Name']}
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="font-size:14px; color:gray;">Total Quantity Purchased</div>
            <div style="font-size:24px; font-weight:bold;">
                {int(top_customer['Bill Qty'])}
            </div>
        """, unsafe_allow_html=True)

    # Top 10 Customers Chart
    st.subheader("💯 Top 10 Customers by Quantity")
    fig1 = px.bar(top_10, x='Customer Name', y='Bill Qty', title='Top 10 Customers',
                  color='Bill Qty', color_continuous_scale='Blues')
    fig1.update_layout(xaxis={'categoryorder': 'total descending'}, xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)

    # Customer Group-wise Sales Chart
    st.subheader("👥 Customer Group-wise Sales (Bar Chart)")
    group_sales = filtered_df.groupby('Customer group')['Bill Qty'].sum().reset_index()
    group_sales = group_sales.sort_values('Bill Qty', ascending=False)
    fig3 = px.bar(group_sales, x='Customer group', y='Bill Qty', title='Customer Group-wise Sales',
                  color='Bill Qty', color_continuous_scale='Viridis')
    fig3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

    # -------------------------------
    # SECTION 2: PREDICTION DASHBOARD
    st.header("🔮 Purchase Date Prediction")

    # Clean column names
    df.columns = df.columns.str.strip()
    df['Bill date'] = pd.to_datetime(df['Bill date'])
    df = df.sort_values(['Customer Code', 'Bill date'])

    # Feature Engineering
    df['Next Purchase Date'] = df.groupby('Customer Code')['Bill date'].shift(-1)
    df['Days Until Next Purchase'] = (df['Next Purchase Date'] - df['Bill date']).dt.days
    df['Previous Purchase Date'] = df.groupby('Customer Code')['Bill date'].shift(1)
    df['Days Since Last Purchase'] = (df['Bill date'] - df['Previous Purchase Date']).dt.days
    df['Purchase Count'] = df.groupby('Customer Code').cumcount() + 1

    feature_cols = ['Bill Qty', 'Days Since Last Purchase', 'Purchase Count']
    df_model = df.dropna(subset=feature_cols + ['Days Until Next Purchase'])

    X = df_model[feature_cols]
    y = df_model['Days Until Next Purchase']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    st.subheader("📊 Model Performance")
    st.write(f"Mean Absolute Error (MAE): **{mae:.2f} days**")

    # Predict next 3 purchases for each customer
    latest_txns = df.sort_values('Bill date').groupby('Customer Code').tail(1)
    latest_txns = latest_txns.dropna(subset=feature_cols)
    next_dates = []

    for _, row in latest_txns.iterrows():
        pred_dates = []
        current_date = row['Bill date']
        days_since_last = row['Days Since Last Purchase']
        purchase_count = row['Purchase Count']
        qty = row['Bill Qty']

        for _ in range(3):
            features = pd.DataFrame([{
                'Bill Qty': qty,
                'Days Since Last Purchase': days_since_last,
                'Purchase Count': purchase_count
            }])
            predicted_days = model.predict(features)[0]
            next_purchase_date = current_date + pd.to_timedelta(predicted_days, unit='D')
            pred_dates.append(next_purchase_date)

            days_since_last = predicted_days
            current_date = next_purchase_date
            purchase_count += 1

        next_dates.append(pred_dates)

    latest_txns['Next Purchase Date 1'] = [d[0] for d in next_dates]
    latest_txns['Next Purchase Date 2'] = [d[1] for d in next_dates]
    latest_txns['Next Purchase Date 3'] = [d[2] for d in next_dates]

    # Format dates for display
    date_cols = ['Bill date', 'Next Purchase Date 1', 'Next Purchase Date 2', 'Next Purchase Date 3']
    for col in date_cols:
        latest_txns[col] = pd.to_datetime(latest_txns[col]).dt.strftime('%d/%m/%Y')

    # Select customer to view predictions
    customer_names = latest_txns['Customer Name'].dropna().unique()
    selected_customer = st.selectbox("Select a customer to view predictions", options=customer_names)

    result = latest_txns[latest_txns['Customer Name'] == selected_customer]
    st.subheader("📌 Next Predicted Purchase Dates")
    st.dataframe(result[['Customer Code', 'Customer Name', 'Bill date',
                         'Next Purchase Date 1', 'Next Purchase Date 2', 'Next Purchase Date 3']])

    # Date range filter for all predictions
    st.markdown("## 🔍 Filter Predictions by Date Range")

    today = datetime.today()
    start_date_pred = st.date_input("Start Date", value=today, key="start_pred")
    end_date_pred = st.date_input("End Date", value=today + timedelta(days=30), key="end_pred")

    start_date_pred = pd.Timestamp(start_date_pred)
    end_date_pred = pd.Timestamp(end_date_pred)

    # Convert back to datetime for filtering (dayfirst=True for dd/mm/yyyy)
    for col in date_cols:
        latest_txns[col] = pd.to_datetime(latest_txns[col], errors='coerce', dayfirst=True)

    filtered_preds = latest_txns[
        latest_txns['Next Purchase Date 1'].between(start_date_pred, end_date_pred) |
        latest_txns['Next Purchase Date 2'].between(start_date_pred, end_date_pred) |
        latest_txns['Next Purchase Date 3'].between(start_date_pred, end_date_pred)
    ]

    filtered_preds = filtered_preds[['Customer Code', 'Customer Name', 'Bill date',
                                     'Next Purchase Date 1', 'Next Purchase Date 2', 'Next Purchase Date 3']]

    for col in ['Bill date', 'Next Purchase Date 1', 'Next Purchase Date 2', 'Next Purchase Date 3']:
        filtered_preds[col] = pd.to_datetime(filtered_preds[col], errors='coerce').dt.strftime('%d-%b-%y')

    st.write(f"Showing predicted purchases between {start_date_pred.date()} and {end_date_pred.date()}")
    st.dataframe(filtered_preds)

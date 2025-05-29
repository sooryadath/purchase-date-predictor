import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import plotly.express as px
st.set_page_config(page_title="Customer Dashboard", layout="wide")
st.title("🛍️ Customer Purchase Date Predictor")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Bill date'] = pd.to_datetime(df['Bill date'])
    df['Year'] = df['Bill date'].dt.year
    
    
    st.title("📊 Billing Dashboard")
    
    st.sidebar.header("📅 Filter by Date Range")
    min_date = df['Bill date'].min()
    max_date = df['Bill date'].max()
    start_date, end_date = st.sidebar.date_input("Select date range", [min_date, max_date])
    
    # Filter data based on selected range
    filtered_df = df[(df['Bill date'] >= pd.to_datetime(start_date)) & (df['Bill date'] <= pd.to_datetime(end_date))]
    
    # ---- Top Customer Info ----
    top_customer_df = filtered_df.groupby('Customer Name')['Bill Qty'].sum().reset_index()
    top_customer_df = top_customer_df.sort_values('Bill Qty', ascending=False)
    
    top_customer = top_customer_df.iloc[0] if not top_customer_df.empty else {"Customer Name": "N/A", "Bill Qty": 0}
    top_100 = top_customer_df.head(100)
    
    # ---- ROW 1: Metric Card ----
    st.subheader("🏆 Top Customer Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Top Customer", value=top_customer['Customer Name'])
    with col2:
        st.metric(label="Total Quantity", value=int(top_customer['Bill Qty']))
    
    # ---- ROW 2: Top 10 Customers Chart ----
    st.subheader("💯 Top 10 Customers by Quantity")
    fig1 = px.bar(top_10, x='Customer Name', y='Bill Qty', title='Top 10 Customers', color='Bill Qty',
                  color_continuous_scale='Blues')
    fig1.update_layout(xaxis={'categoryorder': 'total descending'}, xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)
    
    # ---- ROW 3: Year-wise Sales ----
    st.subheader("📅 Year-wise Sales Quantity")
    year_sales = filtered_df.groupby('Year')['Bill Qty'].sum().reset_index()
    fig2 = px.bar(year_sales, x='Year', y='Bill Qty', title='Year-wise Sales Quantity', text='Bill Qty')
    st.plotly_chart(fig2, use_container_width=True)
    
    # ---- ROW 4: Customer Group-wise Sales (with "Others") ----
    st.subheader("👥 Customer Group-wise Sales")
    group_sales = filtered_df.groupby('Customer group')['Bill Qty'].sum().reset_index()
    
    # Group small contributors as 'Others'
    threshold = 100000
    group_sales['Group Name'] = group_sales.apply(
        lambda row: 'Others' if row['Bill Qty'] < threshold else row['Customer group'], axis=1
    )
    group_final = group_sales.groupby('Group Name')['Bill Qty'].sum().reset_index()
    group_final = group_final.sort_values('Bill Qty', ascending=False)
    
    fig3 = px.pie(group_final, names='Group Name', values='Bill Qty', title='Customer Group-wise Sales')
    st.plotly_chart(fig3, use_container_width=True)
   
    
    # Clean column names
    df.columns = df.columns.str.strip()

    # Preprocessing
    df['Bill date'] = pd.to_datetime(df['Bill date'])
    df = df.sort_values(['Customer Code', 'Bill date'])

    # Feature Engineering
    df['Next Purchase Date'] = df.groupby('Customer Code')['Bill date'].shift(-1)
    df['Days Until Next Purchase'] = (df['Next Purchase Date'] - df['Bill date']).dt.days
    df['Previous Purchase Date'] = df.groupby('Customer Code')['Bill date'].shift(1)
    df['Days Since Last Purchase'] = (df['Bill date'] - df['Previous Purchase Date']).dt.days
    df['Purchase Count'] = df.groupby('Customer Code').cumcount() + 1

    # Model-ready data
    feature_cols = ['Bill Qty', 'Days Since Last Purchase', 'Purchase Count']
    df_model = df.dropna(subset=feature_cols + ['Days Until Next Purchase'])

    # Train/test split
    X = df_model[feature_cols]
    y = df_model['Days Until Next Purchase']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Show MAE
    mae = mean_absolute_error(y_test, model.predict(X_test))
    st.write(f"📊 **Model MAE:** {mae:.2f} days")

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

            # Update for next iteration
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

    # Show predictions for selected customer
    customer_names = latest_txns['Customer Name'].dropna().unique()
    selected_customer = st.selectbox("Select a customer to view predictions", options=customer_names)

    result = latest_txns[latest_txns['Customer Name'] == selected_customer]
    st.write("📌 **Next Predicted Purchase Dates for Selected Customer:**")
    st.dataframe(result[['Customer Code', 'Customer Name', 'Bill date',
                         'Next Purchase Date 1', 'Next Purchase Date 2', 'Next Purchase Date 3']])

    # Show all predictions
    # output_df = latest_txns[['Customer Code', 'Customer Name', 'Bill date',
    #                          'Next Purchase Date 1', 'Next Purchase Date 2', 'Next Purchase Date 3']]
    # st.write("📅 **Predicted Purchase Dates for All Customers:**")
    # st.dataframe(output_df)

    # 📅 Date Range Filter for Predicted Purchases
    st.markdown("## 🔍 Filter Predictions by Date Range")

    # Select date range
    today = datetime.today()
    start_date = st.date_input("Start Date", value=today)
    end_date = st.date_input("End Date", value=today + timedelta(days=30))

    # Convert start_date and end_date from datetime.date to pd.Timestamp
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Convert date columns back to datetime for filtering (dayfirst=True to handle dd/mm/yyyy format)
    for col in date_cols:
        latest_txns[col] = pd.to_datetime(latest_txns[col], errors='coerce', dayfirst=True)

    # Filter rows where any of the Next Purchase Dates fall within the selected date range
    filtered_df = latest_txns[
        latest_txns['Next Purchase Date 1'].between(start_date, end_date) |
        latest_txns['Next Purchase Date 2'].between(start_date, end_date) |
        latest_txns['Next Purchase Date 3'].between(start_date, end_date)
    ]

    # Format all date columns to 'dd-mmm-yy' for display
   # Select only required columns
    filtered_df = filtered_df[['Customer Code', 'Customer Name', 'Bill date',
                           'Next Purchase Date 1', 'Next Purchase Date 2', 'Next Purchase Date 3']]

# Format date columns for display
    for col in ['Bill date', 'Next Purchase Date 1', 'Next Purchase Date 2', 'Next Purchase Date 3']:
      filtered_df[col] = pd.to_datetime(filtered_df[col], errors='coerce').dt.strftime('%d-%b-%y')

    st.write(f"Showing purchases between {start_date.date()} and {end_date.date()}")
    st.dataframe(filtered_df)

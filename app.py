import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.title("🛍️ Customer Purchase Date Predictor")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

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

    # ✅ Format dates
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
    output_df = latest_txns[['Customer Code', 'Customer Name', 'Bill date',
                             'Next Purchase Date 1', 'Next Purchase Date 2', 'Next Purchase Date 3']]
    st.write("📅 **Predicted Purchase Dates for All Customers:**")
    st.dataframe(output_df)
     # 📅 Date Range Filter for Predicted Purchases
st.markdown("## 🔍 Filter Predictions by Date Range")

# Convert date columns with correct format
date_cols = ['Next Purchase Date 1', 'Next Purchase Date 2', 'Next Purchase Date 3']
for col in date_cols:
    latest_txns[col] = pd.to_datetime(latest_txns[col], errors='coerce', dayfirst=True)

# Filter based on any "Next Purchase Date" falling within range
filtered_df = latest_txns[
    latest_txns['Next Purchase Date 1'].between(start_date, end_date) |
    latest_txns['Next Purchase Date 2'].between(start_date, end_date) |
    latest_txns['Next Purchase Date 3'].between(start_date, end_date)
]

# Format the filtered date columns to dd-mmm-yy
for col in date_cols:
    filtered_df[col] = filtered_df[col].dt.strftime('%d-%b-%y')

# Display filtered table
st.write(f"{start_date} – {end_date}")
st.dataframe(filtered_df)

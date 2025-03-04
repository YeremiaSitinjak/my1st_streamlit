import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from scipy.stats import zscore

# Function to convert "clock orientation" to degrees
def convert_clock_orientation(val):
    if isinstance(val, str):
        val = val.strip()
        if ":" in val:  # Example: "3:00", "6:30"
            hours, minutes = map(int, val.split(":"))
            return (hours % 12) * 30 + (minutes / 60) * 30  # Convert to degrees
    elif isinstance(val, pd.Timestamp) or isinstance(val, datetime.time):
        return (val.hour % 12) * 30 + (val.minute / 60) * 30  # Convert time to degrees
    elif isinstance(val, (int, float)):
        return val  # Already in degrees
    return np.nan

# Function to preprocess the uploaded Excel file
def preprocess_data(df):
    df.columns = df.columns.str.strip().str.lower()  # Standardize column names

    # Ensure required columns exist
    required_columns = ["log distance [m]", "component/anomaly type", "clock orientation", "depth [%]", "surface location"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Convert "clock orientation" values
    df["clock orientation"] = df["clock orientation"].apply(convert_clock_orientation)

    # Keep only "anomaly" rows
    df = df[df["component/anomaly type"].str.lower() == "anomaly"]

    # Drop missing values in key columns
    df = df.dropna(subset=["log distance [m]", "depth [%]", "clock orientation"])

    return df

# Function to calculate missing value percentage
def calculate_missing_percentage(df):
    return (df.isnull().sum() / len(df)) * 100

# Function to detect anomalies using different methods
def detect_anomalies(df, method):
    df = df.copy()
    features = ["log distance [m]", "depth [%]", "clock orientation"]

    # Standardize features for consistency
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    if method == "Isolation Forest":
        model = IsolationForest(contamination=0.05, random_state=42)
        df["anomaly_score"] = model.fit_predict(X)

    elif method == "Local Outlier Factor":
        model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        df["anomaly_score"] = model.fit_predict(X)

    elif method == "DBSCAN":
        if len(df) < 5:
            st.warning("DBSCAN requires at least 5 data points. Try another method.")
            df["Anomaly"] = "No"
            return df  # Return without running DBSCAN

        min_samples = max(2, min(5, len(df) // 3))  # Reduce min_samples for Streamlit Cloud
        X_dbscan = df[['log distance [m]', 'depth [%]']].dropna()  # Use fewer features
        X_dbscan = StandardScaler().fit_transform(X_dbscan)  # Scale features

        model = DBSCAN(eps=0.8, min_samples=min_samples)  # Reduce eps for smaller clusters
        labels = model.fit_predict(X_dbscan)

        df["anomaly_score"] = -1  # Default to normal
        df.loc[df.index[:len(labels)], "anomaly_score"] = labels  # Assign computed labels
        df["Anomaly"] = df["anomaly_score"].apply(lambda x: "Yes" if x == -1 else "No")
        return df

    elif method == "Z-Score":
        df["anomaly_score"] = np.abs(zscore(X)).max(axis=1) > 2.5  # Mark as anomaly if z-score > 2.5

    df["Anomaly"] = df["anomaly_score"].apply(lambda x: "Yes" if x == -1 else "No")
    return df

# Streamlit App UI
st.title("Pipe Sensor Data Analysis")

# File Upload Section
uploaded_files = st.file_uploader("Upload multiple Excel files", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    st.subheader("Uploaded Files Summary")
    historical_data = {}
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        year = "".join(filter(str.isdigit, file_name))  # Extract year from filename
        df = pd.read_excel(uploaded_file)
        df = preprocess_data(df)
        historical_data[year] = df

# Show missing percentage
        missing_percentage = calculate_missing_percentage(df)
        st.write(f"**Missing Data Percentage for {file_name}**")
        st.write(missing_percentage)

    if len(historical_data) > 1:
        st.subheader("Anomaly Detection")
        years_sorted = sorted(historical_data.keys())
        latest_year = years_sorted[-1]
        df_latest = historical_data[latest_year]

        # User selects anomaly detection method
        method = st.selectbox("Choose Anomaly Detection Method", ["Isolation Forest", "Local Outlier Factor", "DBSCAN", "Z-Score"])

        # Show loading spinner while detecting anomalies
        with st.spinner(f"Running {method}... Please wait."):
            anomalies_df = detect_anomalies(df_latest, method)

        # Show results in two columns
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Anomaly Detection Results")
            st.dataframe(anomalies_df[["log distance [m]", "depth [%]", "clock orientation", "Anomaly"]])
        with col2:
            anomaly_counts = anomalies_df["Anomaly"].value_counts(normalize=True) * 100
            yes_percent = anomaly_counts.get("Yes", 0)
            no_percent = anomaly_counts.get("No", 0)
            st.subheader("Anomaly Breakdown")
            st.write(f"- ✅ No Anomaly: **{no_percent:.2f}%**")
            st.write(f"- ⚠️ Anomaly: **{yes_percent:.2f}%**")

        # Downloadable anomaly file
        csv = anomalies_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Anomaly Report", csv, f"anomalies_{latest_year}.csv", "text/csv")

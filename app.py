import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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
    df.columns = df.columns.str.strip().str.lower()
    required_columns = ["log distance [m]", "component/anomaly type", "clock orientation"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan
    df["clock orientation"] = df["clock orientation"].apply(convert_clock_orientation)
    df = df.dropna(subset=["log distance [m]", "clock orientation"])
    return df

# Function to detect outliers across different years using Isolation Forest
def detect_outliers_across_years(historical_data, contamination):
    results = []
    for year, df in historical_data.items():
        features = ["log distance [m]", "clock orientation"]
        scaler = StandardScaler()
        X = scaler.fit_transform(df[features])

        model = IsolationForest(contamination=contamination, random_state=42)
        outlier_scores = model.fit_predict(X)
        df["Outlier"] = ["Yes" if score == -1 else "No" for score in outlier_scores]
        df["Year"] = year
        results.append(df)

    combined_df = pd.concat(results).reset_index(drop=True)
    return combined_df

# Function to plot the most outlier-heavy data points
def plot_most_outliers(combined_df):
    outlier_counts = combined_df.groupby("Year")["Outlier"].apply(lambda x: (x == "Yes").sum())
    most_outlier_year = outlier_counts.idxmax()
    df_most_outliers = combined_df[combined_df["Year"] == most_outlier_year]

    fig = px.scatter(
        df_most_outliers, x="log distance [m]", y="clock orientation",
        color="Outlier", title=f"Most Outlier Data Points ({most_outlier_year})",
        labels={"log distance [m]": "Distance [m]", "clock orientation": "Clock Orientation"},
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# Function to generate outlier summary table
def generate_outlier_summary(combined_df):
    summary = combined_df.groupby("Year")["Outlier"].value_counts(normalize=True).unstack() * 100
    summary = summary.rename(columns={"Yes": "% Outliers", "No": "% Normal"}).fillna(0)
    return summary

# Streamlit App UI
st.title("Pipe Sensor Data Analysis - Outlier Detection")

uploaded_files = st.file_uploader("Upload multiple Excel files", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    st.subheader("Uploaded Files Summary")
    historical_data = {}

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        year = "".join(filter(str.isdigit, file_name))
        df = pd.read_excel(uploaded_file)
        df = preprocess_data(df)
        historical_data[year] = df
        st.write(f"**File: {file_name} ({year})**")
        st.write(df.head())
    
    if len(historical_data) > 1:
        st.subheader("Outlier Detection Across Years")
        
        # Display Rule of Thumb Table
        st.subheader("📊 Contamination Sensitivity Guide")
        contamination_table = pd.DataFrame({
            "Contamination Value": ["0.01 (1%)", "0.03 (3%)", "0.05 (5%)", "0.10 (10%)"],
            "Effect": [
                "✅ Very strict: Only extreme anomalies are flagged",
                "⚖️ Balanced: Moderate number of anomalies detected",
                "🔍 Detects more anomalies: Some slight variations flagged",
                "❌ Highly sensitive: Even minor deviations are considered anomalies"
            ]
        })
        st.table(contamination_table)
        
        contamination = st.slider("Select Contamination (Outlier Sensitivity)", 0.01, 0.10, 0.03, 0.01)
        
        with st.spinner(f"Running Isolation Forest with contamination={contamination}... Please wait."):
            combined_df = detect_outliers_across_years(historical_data, contamination)
        
        st.subheader("Most Outlier-Heavy Year")
        plot_most_outliers(combined_df)
        
        # Display summary table
        st.subheader("Outlier Summary by Year")
        summary_table = generate_outlier_summary(combined_df)
        st.table(summary_table)

        st.subheader("Download Outlier Report")
        csv = combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Outlier Report", csv, "outliers_report.csv", "text/csv")

import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats

# Set page configuration
st.set_page_config(layout="wide", page_title="Pipe Sensor Analysis", page_icon="🔍")

# Apply caching to data preprocessing for better performance
@st.cache_data
def preprocess_data(df):
    df.columns = df.columns.str.strip().str.lower()
    required_columns = ["log distance [m]", "component/anomaly type", "clock orientation"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan
    df["clock orientation"] = df["clock orientation"].apply(convert_clock_orientation)
    df = df.dropna(subset=["log distance [m]"])
    return df

# Function to convert "clock orientation" to degrees
def convert_clock_orientation(val):
    try:
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
    except Exception as e:
        st.error(f"Error converting clock orientation: {e}")
        return np.nan

# Function to detect outliers across different years using Isolation Forest
@st.cache_data
def detect_outliers_across_years(historical_data, contamination):
    results = []
    for year, df in historical_data.items():
        features = ["log distance [m]", "clock orientation"]
        scaler = StandardScaler()
        X = scaler.fit_transform(df[features])

        model = IsolationForest(contamination=contamination, random_state=42)
        outlier_scores = model.fit_predict(X)
        df["Outlier"] = ["Yes" if score == -1 else "No" for score in outlier_scores]
        df["Outlier_Score"] = model.decision_function(X)  # Save the anomaly score
        df["Year"] = year
        results.append(df)

    combined_df = pd.concat(results).reset_index(drop=True)
    return combined_df

# Function that classifies anomalies based on severity
def classify_anomalies(combined_df):
    # Create anomaly severity classification based on outlier score
    conditions = [
        (combined_df["Outlier"] == "Yes") & (combined_df["Outlier_Score"] < -0.3),
        (combined_df["Outlier"] == "Yes") & (combined_df["Outlier_Score"] >= -0.3)
    ]
    choices = ["Severe", "Moderate"]
    combined_df["Anomaly_Severity"] = np.select(conditions, choices, default="None")
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
        template="plotly_white", hover_data=["component/anomaly type", "Anomaly_Severity"]
    )
    st.plotly_chart(fig, use_container_width=True)

# Function to generate outlier summary table
def generate_outlier_summary(combined_df):
    summary = combined_df.groupby("Year")["Outlier"].value_counts(normalize=True).unstack() * 100
    summary = summary.rename(columns={"Yes": "% Outliers", "No": "% Normal"}).fillna(0)
    return summary

# Add this function to plot outlier trends across years
def plot_outlier_trends(combined_df):
    outlier_trends = combined_df.groupby("Year")["Outlier"].value_counts(normalize=True).unstack().fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=outlier_trends.index, y=outlier_trends["Yes"]*100, 
                             mode='lines+markers', name='Outliers %'))
    fig.update_layout(
        title="Outlier Trends Across Years", 
        xaxis_title="Year", 
        yaxis_title="Percentage of Outliers (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# Add this function to generate a heatmap of anomalies
def plot_anomaly_heatmap(combined_df):
    # Create distance bins and clock orientation bins without using Interval objects
    distance_min = combined_df['log distance [m]'].min()
    distance_max = combined_df['log distance [m]'].max()
    distance_bins = np.linspace(distance_min, distance_max, 21)
    
    orientation_min = combined_df['clock orientation'].min()
    orientation_max = combined_df['clock orientation'].max()
    orientation_bins = np.linspace(orientation_min, orientation_max, 13)
    
    # Get bin indices instead of Interval objects
    combined_df['distance_bin'] = np.digitize(combined_df['log distance [m]'], distance_bins)
    combined_df['orientation_bin'] = np.digitize(combined_df['clock orientation'], orientation_bins)
    
    # Group by distance bin and orientation bin, count outliers
    heatmap_data = (combined_df[combined_df['Outlier'] == 'Yes']
                     .groupby(['distance_bin', 'orientation_bin'])
                     .size()
                     .reset_index(name='count'))
    
    # Create a pivot table using integer bin indices
    pivot_data = heatmap_data.pivot(index='distance_bin', columns='orientation_bin', values='count').fillna(0)
    
    fig = px.imshow(
        pivot_data, 
        labels=dict(x="Clock Orientation Bin", y="Distance Bin", color="Anomaly Count"),
        title="Anomaly Distribution Heatmap",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)


# Improved function to count and compare weld anomalies across years
def weld_error_summary(historical_data):
    weld_counts = {}
    
    # First, let's print the unique values to help debugging
    with st.expander("Debug: Unique anomaly types in data", expanded=False):
        for year, df in historical_data.items():
            st.write(f"Year {year} unique values:")
            st.write(df["component/anomaly type"].unique())
    
    for year, df in historical_data.items():
        # More flexible matching - use contains instead of exact match
        count = df[df["component/anomaly type"].str.lower().str.contains("weld")].shape[0]
        weld_counts[year] = count

    if not weld_counts or all(count == 0 for count in weld_counts.values()):
        st.warning("No weld entries found in any of the datasets. Check if weld entries have a different label.")
        min_year = list(weld_counts.keys())[0] if weld_counts else "N/A"
        min_count = 0
    else:
        min_year = min(weld_counts, key=weld_counts.get)
        min_count = weld_counts[min_year]

    st.subheader("Weld Error Summary")
    result_data = []
    for year, count in weld_counts.items():
        diff = count - min_count
        pct_change = (diff / min_count) * 100 if min_count > 0 else 0
        result_data.append((year, count, diff, pct_change))

    result_df = pd.DataFrame(result_data, columns=["Year", "Weld Count", "+/-", "% Change"])
    st.dataframe(result_df)
    
    # Add weld count visualization if there are welds
    if any(weld_counts.values()):
        fig = px.bar(
            result_df, x="Year", y="Weld Count",
            title="Weld Counts by Year",
            text="Weld Count"
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    return result_df

# Improved function to pair welds by proximity across years
def match_welds_across_years(historical_data, tolerance=0.05):
    years = sorted(historical_data.keys())
    all_pairs = []

    for i in range(len(years)-1):
        year_a = years[i]
        year_b = years[i+1]

        # More flexible matching for welds
        df_a = historical_data[year_a][historical_data[year_a]["component/anomaly type"].str.lower().str.contains("weld")]
        df_b = historical_data[year_b][historical_data[year_b]["component/anomaly type"].str.lower().str.contains("weld")]

        distances_a = df_a["log distance [m]"].values.reshape(-1, 1) if df_a.shape[0] > 0 else np.array([]).reshape(-1, 1)
        distances_b = df_b["log distance [m]"].values.reshape(-1, 1) if df_b.shape[0] > 0 else np.array([]).reshape(-1, 1)

        if distances_a.shape[0] == 0 or distances_b.shape[0] == 0:
            all_pairs.append((f"{year_a}-{year_b}", len(df_a), 0, len(df_a)))
            continue

        # Improved error handling for NearestNeighbors
        try:
            model = NearestNeighbors(radius=tolerance)
            model.fit(distances_b)
            matches = model.radius_neighbors(distances_a, return_distance=False)
            pair_count = sum([1 for match in matches if len(match) > 0])
        except Exception as e:
            st.error(f"Error in matching algorithm: {e}")
            pair_count = 0
            
        all_pairs.append((f"{year_a}-{year_b}", len(df_a), pair_count, len(df_a) - pair_count))

    st.subheader("Weld Matching Results")
    pair_df = pd.DataFrame(all_pairs, columns=["Year Pair", "Original Count", "Matched", "Unmatched"])
    st.dataframe(pair_df)
    
    # Add visualization only if there are welds
    if any(row[1] > 0 for row in all_pairs):
        fig = px.bar(
            pair_df, x="Year Pair", y=["Matched", "Unmatched"],
            title="Matched vs Unmatched Welds Between Consecutive Years",
            barmode="stack"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    return pair_df


# Improved function to pair welds by proximity across years
def match_welds_across_years(historical_data, tolerance=0.05):
    years = sorted(historical_data.keys())
    all_pairs = []

    for i in range(len(years)-1):
        year_a = years[i]
        year_b = years[i+1]

        # More flexible matching for welds
        df_a = historical_data[year_a][historical_data[year_a]["component/anomaly type"].str.lower().str.contains("weld")]
        df_b = historical_data[year_b][historical_data[year_b]["component/anomaly type"].str.lower().str.contains("weld")]

        distances_a = df_a["log distance [m]"].values.reshape(-1, 1) if df_a.shape[0] > 0 else np.array([]).reshape(-1, 1)
        distances_b = df_b["log distance [m]"].values.reshape(-1, 1) if df_b.shape[0] > 0 else np.array([]).reshape(-1, 1)

        if distances_a.shape[0] == 0 or distances_b.shape[0] == 0:
            all_pairs.append((f"{year_a}-{year_b}", len(df_a), 0, len(df_a)))
            continue

        # Improved error handling for NearestNeighbors
        try:
            model = NearestNeighbors(radius=tolerance)
            model.fit(distances_b)
            matches = model.radius_neighbors(distances_a, return_distance=False)
            pair_count = sum([1 for match in matches if len(match) > 0])
        except Exception as e:
            st.error(f"Error in matching algorithm: {e}")
            pair_count = 0
            
        all_pairs.append((f"{year_a}-{year_b}", len(df_a), pair_count, len(df_a) - pair_count))

    st.subheader("Weld Matching Results")
    pair_df = pd.DataFrame(all_pairs, columns=["Year Pair", "Original Count", "Matched", "Unmatched"])
    st.dataframe(pair_df)
    
    # Add visualization only if there are welds
    if any(row[1] > 0 for row in all_pairs):
        fig = px.bar(
            pair_df, x="Year Pair", y=["Matched", "Unmatched"],
            title="Matched vs Unmatched Welds Between Consecutive Years",
            barmode="stack"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    return pair_df


# Function to perform statistical tests between years
def compare_distributions(historical_data):
    years = sorted(historical_data.keys())
    results = []
    
    for i in range(len(years)-1):
        year_a, year_b = years[i], years[i+1]
        # KS test for log distance
        ks_distance, p_distance = stats.ks_2samp(
            historical_data[year_a]["log distance [m]"], 
            historical_data[year_b]["log distance [m]"]
        )
        # KS test for clock orientation
        ks_orient, p_orient = stats.ks_2samp(
            historical_data[year_a]["clock orientation"], 
            historical_data[year_b]["clock orientation"]
        )
        
        results.append({
            "Year Pair": f"{year_a}-{year_b}",
            "KS Stat (Distance)": round(ks_distance, 4),
            "p-value (Distance)": round(p_distance, 4),
            "Different? (Distance)": "Yes" if p_distance < 0.05 else "No",
            "KS Stat (Orientation)": round(ks_orient, 4),
            "p-value (Orientation)": round(p_orient, 4),
            "Different? (Orientation)": "Yes" if p_orient < 0.05 else "No"
        })
    
    results_df = pd.DataFrame(results)
    st.subheader("Statistical Comparison Between Years")
    st.dataframe(results_df)
    
    return results_df

# Streamlit App UI with improved layout
st.title("🔍 Pipe Sensor Data Analysis - Enhanced Version")

# Create sidebar for controls
with st.sidebar:
    st.header("Controls")
    st.info("ℹ️ Upload Excel files containing pipe sensor data from different years.")
    uploaded_files = st.file_uploader("Upload multiple Excel files", 
                                    accept_multiple_files=True, 
                                    type=['xlsx'])
    
    st.markdown("---")
    st.subheader("Outlier Detection Settings")
    contamination = st.slider(
        "Select Contamination (Outlier Sensitivity)", 
        0.01, 0.10, 0.03, 0.01,
        help="Lower values are more strict in identifying outliers."
    )
    
    tolerance = st.slider(
        "Weld Matching Tolerance", 
        0.01, 0.20, 0.05, 0.01,
        help="Maximum distance difference to consider welds as matching between years."
    )

if uploaded_files:
    try:
        historical_data = {}

        # Load and process all files
        with st.spinner("Processing uploaded files..."):
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                year = "".join(filter(str.isdigit, file_name))
                df = pd.read_excel(uploaded_file)
                df = preprocess_data(df)
                historical_data[year] = df
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Overview", 
            "🔎 Outlier Analysis", 
            "🔧 Weld Analysis",
            "📈 Statistical Tests"
        ])
        
        # Tab 1: Data Overview
        with tab1:
            st.header("Uploaded Files Summary")
            
            for year, df in historical_data.items():
                with st.expander(f"File: {year} - {df.shape[0]} records"):
                    st.dataframe(df.head())
                    
                    # Show basic statistics
                    st.subheader("Basic Statistics")
                    st.dataframe(df.describe())
                    
                    # Show distribution of anomaly types
                    anomaly_counts = df["component/anomaly type"].value_counts()
                    fig = px.pie(
                        values=anomaly_counts.values, 
                        names=anomaly_counts.index,
                        title=f"Distribution of Anomaly Types - {year}"
                    )
                    st.plotly_chart(fig)
        
        if len(historical_data) > 1:
            # Tab 2: Outlier Analysis
            with tab2:
                st.header("Outlier Detection Across Years")
                
                # Display Rule of Thumb Table
                with st.expander("📊 Contamination Sensitivity Guide", expanded=False):
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
                
                with st.spinner(f"Running Isolation Forest with contamination={contamination}... Please wait."):
                    combined_df = detect_outliers_across_years(historical_data, contamination)
                    combined_df = classify_anomalies(combined_df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Most Outlier-Heavy Year")
                    plot_most_outliers(combined_df)
                
                with col2:
                    st.subheader("Outlier Trends")
                    plot_outlier_trends(combined_df)
                
                # Display summary table
                st.subheader("Outlier Summary by Year")
                summary_table = generate_outlier_summary(combined_df)
                st.dataframe(summary_table)
                
                # Add anomaly heatmap
                st.subheader("Anomaly Distribution")
                plot_anomaly_heatmap(combined_df)
                
                # Add anomaly severity breakdown
                st.subheader("Anomaly Severity Breakdown")
                severity_counts = combined_df.groupby(["Year", "Anomaly_Severity"]).size().reset_index(name="Count")
                fig = px.bar(
                    severity_counts, 
                    x="Year", 
                    y="Count", 
                    color="Anomaly_Severity",
                    title="Anomaly Severity by Year",
                    barmode="group"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download button for outlier report
                st.subheader("Download Outlier Report")
                csv = combined_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Outlier Report", csv, "outliers_report.csv", "text/csv")
            
            # Tab 3: Weld Analysis
            with tab3:
                st.header("Weld Analysis Across Years")
                
                weld_summary = weld_error_summary(historical_data)
                pairing_results = match_welds_across_years(historical_data, tolerance)
            
            # Tab 4: Statistical Tests
            with tab4:
                st.header("Statistical Analysis")
                
                st.write("""
                This tab shows statistical tests to compare distributions between consecutive years.
                A p-value < 0.05 indicates that the distributions are significantly different.
                """)
                
                compare_distributions(historical_data)
                
                # Add visualization for total records by year
                st.subheader("Total Records by Year")
                records_by_year = {year: df.shape[0] for year, df in historical_data.items()}
                records_df = pd.DataFrame(list(records_by_year.items()), columns=["Year", "Record Count"])
                
                fig = px.bar(
                    records_df,
                    x="Year",
                    y="Record Count",
                    title="Total Records by Year",
                    text="Record Count"
                )
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Please upload at least two Excel files to enable comparative analysis.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your data format and try again.")
else:
    # Welcome screen when no files are uploaded
    st.write("## 👋 Welcome to the Pipe Sensor Data Analysis App!")
    st.write("""
    This application helps analyze pipe sensor data across different years to:
    - Detect and visualize outliers using machine learning
    - Compare weld anomalies across years
    - Perform statistical tests to identify significant changes
    - Generate comprehensive visual reports
    
    ### Getting Started
    1. Upload Excel files containing pipe sensor data from the sidebar
    2. Each file should represent a different year of data
    3. File names should include the year (e.g., "PipeSensor2022.xlsx")
    4. Required columns: "log distance [m]", "component/anomaly type", "clock orientation"
    """)
    
    # Display example data format
    st.subheader("Expected Data Format")
    example_data = pd.DataFrame({
        "log distance [m]": [10.5, 15.2, 20.7, 25.3],
        "component/anomaly type": ["weld", "dent", "weld", "corrosion"],
        "clock orientation": ["3:00", "6:30", "12:00", "9:15"]
    })
    st.dataframe(example_data)

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Initialize session state for segmented analysis
if 'segmented_results' not in st.session_state:
    st.session_state.segmented_results = None
if 'segment_stats' not in st.session_state:
    st.session_state.segment_stats = None
if 'segments' not in st.session_state:
    st.session_state.segments = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'unmatched_welds' not in st.session_state:
    st.session_state.unmatched_welds = None
if 'defect_matching_results' not in st.session_state:
    st.session_state.defect_matching_results = None
if 'defect_matching_plot' not in st.session_state:
    st.session_state.defect_matching_plot = None
if "run_global_isolation" not in st.session_state:
    st.session_state.run_global_isolation = False


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
    numeric_cols = ["width [mm]", "length [mm]", "depth [%]"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["clock orientation"] = df["clock orientation"].apply(convert_clock_orientation)
    df["component/anomaly type"] = df["component/anomaly type"].astype(str)
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

#function to convert mm to degrees based on pipe diameter
def mm_to_deg(width_mm, pipe_diameter_mm):
    """Convert anomaly width in mm to degrees using the pipe diameter."""
    circumference = np.pi * pipe_diameter_mm
    return (width_mm / circumference) * 360 if pipe_diameter_mm > 0 else 0

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
    # Create a single figure with all years combined
    fig = px.scatter(
        combined_df, 
        x="log distance [m]", 
        y="clock orientation",
        color="Year",  # Color by year
        title="Outlier Distribution Across All Years",
        labels={"log distance [m]": "Distance [m]", "clock orientation": "Clock Orientation"},
        template="plotly_white", 
        hover_data=["component/anomaly type", "Anomaly_Severity", "Year"],
        category_orders={"Year": sorted(combined_df["Year"].unique())}
    )
    
    # Configure zoom and layout
    fig.update_layout(
        dragmode='zoom',
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add rectangle shapes for zoomable anomalies
    shapes = []
    for _, row in combined_df.iterrows():
        if row["Outlier"] == "Yes":
            shapes.append(dict(
                type="rect",
                xref="x", yref="y",
                x0=row["log distance [m]"] - 0.5,
                x1=row["log distance [m]"] + 0.5,
                y0=row["clock orientation"] - 5,
                y1=row["clock orientation"] + 5,
                fillcolor="rgba(255, 255, 0, 0.4)",
                line=dict(width=0)
            ))
    
    fig.update_layout(shapes=shapes)
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

    st.subheader("Weld Count Summary")
    result_data = []
    for year, count in weld_counts.items():
        diff = count - min_count
        pct_change = (diff / min_count) * 100 if min_count > 0 else 0
        result_data.append((year, count, diff, pct_change))

    result_df = pd.DataFrame(result_data, columns=["Year", "Weld Count", "+/-", "% Change"])
    st.dataframe(result_df)
    
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

# Function to filter and display data for a specific segment
def filter_and_display_segment(combined_df, year, segment=None):
    """Efficiently filter and display data for a selected segment without reprocessing."""
    df_to_plot = combined_df[combined_df["Year"] == year]
    
    if segment != "All Segments" and segment is not None:
        df_to_plot = df_to_plot[df_to_plot["Segment"] == segment]
    
    title = f"Outliers in {year} - {segment if segment != 'All Segments' else 'All Segments'}"
    
    fig = px.scatter(
        df_to_plot,
        x="log distance [m]",
        y="clock orientation",
        color="Outlier",
        title=title,
        color_discrete_map={"Yes": "blue", "No": "red"},
        hover_data=["Outlier_Score", "component/anomaly type"]
    )
    st.plotly_chart(fig, use_container_width=True,key=f"segment_plot_{year}")


# Function to create weld-based segments
@st.cache_data
def create_weld_based_segments(historical_data):
    """Segment pipe data using welds as natural boundary points."""
    segmentation_results = {}
    
    for year, df in historical_data.items():
        # Find all weld positions
        weld_rows = df[df["component/anomaly type"].str.lower().str.contains("weld")]
        weld_positions = sorted(weld_rows["log distance [m]"].values)
        
        # Add start and end points if needed
        min_dist = df["log distance [m]"].min()
        max_dist = df["log distance [m]"].max()
        
        if len(weld_positions) == 0:
            segments = [(min_dist, max_dist, "Full-Pipe")]
        else:
            segments = []
            # Add segment before first weld if needed
            if weld_positions[0] > min_dist + 1:
                segments.append((min_dist, weld_positions[0], "Pre-Weld"))
                
            # Add segments between welds
            for i in range(len(weld_positions)-1):
                start = weld_positions[i]
                end = weld_positions[i+1]
                segments.append((start, end, f"W{i+1}-W{i+2}"))
                
            # Add segment after last weld if needed
            if weld_positions[-1] < max_dist - 1:
                segments.append((weld_positions[-1], max_dist, "Post-Weld"))
        
        segmentation_results[year] = segments
    
    return segmentation_results

# Function to detect outliers within each segment
@st.cache_data
def detect_outliers_by_segment(historical_data, segments, contamination=None):
    """Detect outliers within each segment separately."""
    results = []
    segment_stats = []
    
    for year, df in historical_data.items():
        year_segments = segments[year]
        
        for start, end, segment_name in year_segments:
            # Extract data for this segment
            segment_df = df[(df["log distance [m]"] >= start) & 
                           (df["log distance [m]"] < end)].copy()
            
            #if len(segment_df) < 10:  # Skip if too few points
            #    continue
                
            # Dynamic contamination based on segment size if not specified
            if contamination is None:
                segment_contamination = min(0.05, 10/len(segment_df))
            else:
                segment_contamination = contamination
                
            # Extract features for outlier detection
            features = ["log distance [m]", "clock orientation"]
            scaler = StandardScaler()
            X = scaler.fit_transform(segment_df[features])
            
            # Apply Isolation Forest
            model = IsolationForest(contamination=segment_contamination, random_state=42)
            outlier_scores = model.fit_predict(X)
            segment_df["Outlier"] = ["Yes" if score == -1 else "No" for score in outlier_scores]
            segment_df["Outlier_Score"] = model.decision_function(X)
            segment_df["Year"] = year
            segment_df["Segment"] = segment_name
            
            # Collect results
            results.append(segment_df)
            
            # Collect segment statistics
            total_points = len(segment_df)
            outlier_count = (segment_df["Outlier"] == "Yes").sum()
            segment_stats.append({
                "Year": year,
                "Segment": segment_name,
                "Start (m)": round(start, 2),
                "End (m)": round(end, 2),
                "Length (m)": round(end - start, 2),
                "Total Points": total_points,
                "Outliers": outlier_count,
                "Outlier %": round(100 * outlier_count / total_points, 2) if total_points > 0 else 0
            })
    
    # Combine all results
    combined_df = pd.concat(results).reset_index(drop=True) if results else pd.DataFrame()
    segment_stats_df = pd.DataFrame(segment_stats)
    
    return combined_df, segment_stats_df

# Visualization functions for segmented analysis
def plot_segment_outlier_summary(segment_stats_df):
    """Plot a summary of outliers by segment."""
    fig = px.bar(
        segment_stats_df,
        x="Segment",
        y="Outlier %",
        color="Year",
        barmode="group",
        title="Outlier Percentage by Segment",
        text="Outlier %"
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

def plot_segment_scatter(combined_df, year, segment=None):
    """Plot scatter plot of a specific segment's data points."""
    df_to_plot = combined_df[combined_df["Year"] == year]
    
    if segment is not None:
        df_to_plot = df_to_plot[df_to_plot["Segment"] == segment]
    
    title = f"Outliers in {year} - {segment if segment else 'All Segments'}"
    
    fig = px.scatter(
        df_to_plot,
        x="log distance [m]",
        y="clock orientation",
        color="Outlier",
        title=title,
        color_discrete_map={"Yes": "blue", "No": "red"},
        hover_data=["Outlier_Score", "component/anomaly type"]
    )
    st.plotly_chart(fig, use_container_width=True,key=f"segment_distribution_plot_allyears")

# Streamlit App UI with improved layout
st.title("🔍 Inline Inspection Data Matching & Prediction")

# Create sidebar for controls
with st.sidebar:
    st.header("Controls")
    st.info("ℹ️ Upload Excel files containing pipe sensor data from different years.")
    uploaded_files = st.file_uploader("Upload multiple Excel files", 
                                    accept_multiple_files=True, 
                                    type=['xlsx','csv'])
    
    # #outlier detection settings
    # st.markdown("---")
    # st.subheader("Outlier Detection Settings")
    # contamination = st.slider(
    #     "Select Contamination (Outlier Sensitivity)", 
    #     0.01, 0.10, 0.03, 0.01,
    #     help="Lower values are more strict in identifying outliers."
    # )
    contamination = 0.03
    
    # tolerance = st.slider(
    #     "Weld Matching Tolerance", 
    #     0.01, 0.20, 0.05, 0.01,
    #     help="Maximum distance difference to consider welds as matching between years."
    # )
    tolerance = 0.05

    #pipe geometry settings
    st.markdown("---")
    st.subheader("Pipe Geometry")
    inside_diameter_mm = st.number_input(
        "Inside Diameter (mm)", 
        min_value=1.0, 
        max_value=5000.0, 
        value=900.0,  # default value
        step=1.0,
        help="Enter the pipe's inside diameter in millimeters. Used for converting width [mm] to degrees."
    )

if uploaded_files:
    try:
        historical_data = {}

        # Load and process all files
        with st.spinner("Processing uploaded files..."):
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                year = "".join(filter(str.isdigit, file_name))

                # Read file based on extension
                if file_name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                elif file_name.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.warning(f"Unsupported file type: {file_name}")
                    continue

                df = preprocess_data(df)
                historical_data[year] = df
                st.session_state.historical_data = historical_data
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5= st.tabs([
            "📊 Overview",
            "🔧 Data Analysis", 
            "🔎 ILI Data Matching", 
            "📈 Statistical Tests",
            "🛡️ Corrosion Rate & Prediction"
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

            # Determine global min/max depth across all years for slider range
            all_depths = []
            for df in historical_data.values():
                if "depth [%]" in df.columns:
                    all_depths.extend(df["depth [%]"].dropna().tolist())

            if all_depths:
                min_depth = float(np.nanmin(all_depths))
                max_depth = float(np.nanmax(all_depths))
            else:
                min_depth, max_depth = 0.0, 100.0  # Fallback if no depth data

            selected_min_depth, selected_max_depth = st.slider(
                'Select Depth Range [%]',
                min_value=min_depth,
                max_value=max_depth,
                value=(min_depth, max_depth)
            )
            st.write(f"Showing anomalies with depth between {selected_min_depth:.1f}% and {selected_max_depth:.1f}%")

            # Determine the global min/max for distance across all years
            all_distances = []
            for df in historical_data.values():
                all_distances.extend(df["log distance [m]"].dropna().tolist())
            if all_distances:
                min_distance = float(np.nanmin(all_distances))
                max_distance = float(np.nanmax(all_distances))
            else:
                min_distance, max_distance = 0.0, 1.0

            # Add Go to Distance UI
            with st.form(key="zoom_form"):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    goto_distance = st.number_input(
                        "Go to Distance (left edge, meters):",
                        min_value=min_distance,
                        max_value=max_distance,
                        value=min_distance,
                        step=1.0
                    )
                with col2:
                    zoom_width = st.number_input(
                        "Zoom width (meters):",
                        min_value=1.0,
                        max_value=max_distance - min_distance,
                        value=50.0,
                        step=1.0
                    )
                with col3:
                    go_button = st.form_submit_button("Go / Zoom")

            with st.spinner("Preparing scatter plots..."):
                progress_bar = st.progress(0)

            #set Y axis when zoom or go to distance
            all_y = []
            for df in historical_data.values():
                if "clock orientation" in df.columns:
                    all_y.extend(df["clock orientation"].dropna().tolist())
            if all_y:
                ymin = min(all_y)
                ymax = max(all_y)
            else:
                ymin, ymax = 0, 360

            for year, df in historical_data.items():
                # Filter for anomalies only
                anomaly_df = df[df["component/anomaly type"].str.lower().str.contains("anomaly", na=False)]
                # Filter by depth range if column exists
                if "depth [%]" in anomaly_df.columns:
                    anomaly_df = anomaly_df[
                        (anomaly_df["depth [%]"] >= selected_min_depth) &
                        (anomaly_df["depth [%]"] <= selected_max_depth)
                    ]
                if len(anomaly_df) == 0:
                    st.info(f"No anomalies found in {year} within selected depth range.")
                    continue

                fig = go.Figure()

                # Add anomaly points
                fig.add_trace(go.Scattergl(
                    x=anomaly_df["log distance [m]"],
                    y=anomaly_df["clock orientation"],
                    mode='markers',
                    marker=dict(color='blue', size=6),
                    name='Anomaly Center',
                    customdata=np.stack([
                    anomaly_df.get("length [mm]", pd.Series([None]*len(anomaly_df))),
                    anomaly_df.get("width [mm]", pd.Series([None]*len(anomaly_df))),
                    anomaly_df.get("depth [%]", pd.Series([None]*len(anomaly_df)))
                ], axis=-1),
                hovertemplate=
                    "Distance: %{x}<br>"+
                    "Orientation: %{y}<br>"+
                    "Length [mm]: %{customdata[0]:.2f}<br>"+
                    "Width [mm]: %{customdata[1]:.2f}<br>"+
                    "Depth [%]: %{customdata[2]}<br>"+
                    "<extra></extra>"
                ))

                # Get weld positions for this year
                weld_positions = df[df["component/anomaly type"].str.lower().str.contains("weld", na=False)]["log distance [m]"].dropna().unique()

                # Add rectangles for each anomaly
                shapes = []
                for _, row in anomaly_df.iterrows():
                    x = row["log distance [m]"]
                    y = row["clock orientation"]
                    length = row.get("length [mm]", 0) / 1000 if pd.notnull(row.get("length [mm]", 0)) else 0
                    width_mm = row.get("width [mm]", 0) if pd.notnull(row.get("width [mm]", 0)) else 0
                    width_deg = mm_to_deg(width_mm, inside_diameter_mm)
                    x0 = x - length/2
                    x1 = x + length/2
                    y0 = y - width_deg/2
                    y1 = y + width_deg/2
                    shapes.append(dict(
                        type="rect",
                        xref="x", yref="y",
                        x0=x0, x1=x1,
                        y0=y0, y1=y1,
                        line=dict(color="red"),
                        fillcolor="rgba(0,0,0,0)",
                        layer="above"
                    ))

                # Add range slider for interactive zooming
                #fig.update_xaxes(
                #    rangeslider_visible=True,
                #    rangeslider_thickness=0.1
                #)

                #scatter plot loop
                fig.update_layout(
                    title=f"Anomaly Distribution in {year}",
                    xaxis_title="Log Distance (m)",
                    yaxis_title="Clock Orientation (deg)",
                    shapes=shapes,
                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(
                        showspikes=True,
                        spikemode='across',
                        spikesnap='cursor',
                        showline=True,
                        spikethickness=1,
                        spikecolor="black"
                    ),
                    yaxis=dict(
                        showspikes=True,
                        spikemode='across',
                        spikesnap='cursor',
                        showline=True,
                        spikethickness=1,
                        spikecolor="black"
                    ),
                    hoverlabel=dict(bgcolor="white", font_size=12)
                )

                # Fast approach - limit number of welds and remove annotations
                if len(weld_positions) > 30:
                    # Sample evenly spaced welds
                    indices = np.linspace(0, len(weld_positions)-1, 30, dtype=int)
                    weld_positions = weld_positions[indices]

                # Add vertical lines for welds
                for weld_x in weld_positions:
                    fig.add_vline(
                        x=weld_x,
                        line_width=1,
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Weld",
                        #annotation_position="top left"
                    )
                    progress_bar.progress(len(weld_positions))
                
                # 
                if st.session_state.get("xaxis_range") is not None:
                    fig.update_xaxes(range=st.session_state.xaxis_range)
                    fig.update_yaxes(range=[ymin, ymax])
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Set the x-axis range if Go button is pressed
                if go_button:
                    st.session_state.xaxis_range = [goto_distance, goto_distance + zoom_width]

                progress_bar.empty()
        
        if len(historical_data) > 1:
            # Tab 3: Outlier Analysis
            with tab3:                  
                #sub menu global isolation forest anomaly matching
                with st.expander("Isolation Forest Anomaly Matching", expanded=False):
                    
                    st.subheader("Anomaly Matching Across Years")
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
                    
                    if st.button("Run Global Isolation Forest"):
                        st.session_state.run_global_isolation = True
                    
                    if st.session_state.run_global_isolation:
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
                
                #sub menu isolation forest anomaly matching with segments
                with st.expander("Isolation Forest Anomaly Matching within Segments", expanded=False):
                    st.subheader("Segmented Anomaly Matching Analysis")
                
                    st.write("""
                    This analysis divides the pipe into segments based on weld positions and performs
                    outlier detection within each segment.
                    """)
                    
                    # Only create segments and run detection if not already in session state
                    if st.session_state.segments is None:
                        with st.spinner("Creating weld-based segments..."):
                            st.session_state.segments = create_weld_based_segments(historical_data)
                    
                    segments = st.session_state.segments
                    
                    # Check if segments were created
                    has_segments = all(len(segs) > 0 for segs in segments.values())
                    
                    if not has_segments:
                        st.warning("Not enough weld markers found to create segments. Check your data.")
                    else:
                        # Show segment overview
                        st.subheader("Segment Overview")
                    
                    for year, year_segments in segments.items():
                        segment_df = pd.DataFrame([
                            {"Segment": seg_name, "Start (m)": round(start, 2), "End (m)": round(end, 2), 
                            "Length (m)": round(end-start, 2)}
                            for start, end, seg_name in year_segments
                        ])
                        
                        st.markdown(f"### 📁 Year {year} — {len(year_segments)} segments")
                        st.dataframe(segment_df, height=300)
                    
                    # Outlier detection settings
                    st.subheader("Segment-Specific Outlier Detection")
                    
                    segment_contamination = contamination
                    st.info(f"Using global contamination value: {contamination}")
                    
                    # Add a run analysis button to control when the expensive calculations happen
                    run_analysis = st.button("Run Segmented Analysis",key=f"run_analysis_{year}") or st.session_state.segmented_results is not None
                    
                    if run_analysis:
                        # Only run detection if not already in session state
                        if st.session_state.segmented_results is None:
                            with st.spinner("Running segmented outlier detection... (this may take a moment)"):
                                combined_df, segment_stats_df = detect_outliers_by_segment(
                                    historical_data, segments, 
                                    segment_contamination
                                )
                                st.session_state.segmented_results = combined_df
                                st.session_state.segment_stats = segment_stats_df
                        
                        # Use the cached results for display
                        combined_df = st.session_state.segmented_results
                        segment_stats_df = st.session_state.segment_stats
                        
                        if len(segment_stats_df) > 0:
                            # Display segment statistics
                            st.subheader("Segment Statistics")
                            st.dataframe(segment_stats_df)
                            
                            # Plot summary
                            st.subheader("Outlier Distribution by Segment")
                            plot_segment_outlier_summary(segment_stats_df)
                            
                            # Explore individual segments - THIS SECTION WILL BE MUCH FASTER NOW
                            st.subheader("Explore Individual Segments")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                selected_year = st.selectbox("Select Year", sorted(historical_data.keys()))
                            
                            with col2:
                                year_segments = [seg[2] for seg in segments[selected_year]]
                                selected_segment = st.selectbox("Select Segment", ["All Segments"] + year_segments)
                            
                            # This function only filters existing data, doesn't rerun the analysis
                            filter_and_display_segment(combined_df, selected_year, selected_segment)
                            
                            # Plot for all years
                            st.subheader("Outlier Distribution Across All Years")
                            fig = px.scatter(
                                combined_df,
                                x="log distance [m]",
                                y="clock orientation",
                                color="Outlier",
                                title="Outlier Distribution Across All Years",
                                color_discrete_map={"Yes": "blue", "No": "red"},
                                hover_data=["Outlier_Score", "component/anomaly type","Year"]
                            )
                            fig.update_traces(marker=dict(size=5))
                            st.plotly_chart(fig, use_container_width=True,key=f"segment_plot_allyears")

                            # Download segmented results
                            st.subheader("Download Segmented Analysis")
                            csv = combined_df.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Segmented Results", csv, "segmented_outliers.csv", "text/csv")
                        else:
                            st.warning("No valid segments found for analysis. Please check your data.")

                
                #sub menu ruled based defect matching
                with st.expander("Rule Based Defect Matching", expanded=False):
                    st.subheader("Defect Matching Across All Years by Rule Based")

                    if 'historical_data' not in st.session_state or not st.session_state.historical_data:
                        st.warning("Please upload and preprocess data first.")
                    else:
                        with st.form("rule_matching_form"):
                            st.subheader("Defect Matching Controls")
                            tolerance_pct = st.slider(
                                "Matching Tolerance (%)", 
                                0.0, 10.0, 5.0, 0.5,
                                help="Maximum allowed difference in width/length between matched defects"
                            ) / 100  # Convert to decimal
                            run_matching = st.form_submit_button("Run Defect Matching")

                        if run_matching or st.session_state.defect_matching_results is not None:
                            if run_matching or st.session_state.defect_matching_results is None:

                    
                                # Combine all years' anomalies into one DataFrame
                                all_anomalies = []
                                for year, df in st.session_state.historical_data.items():
                                    anomalies = df[df["component/anomaly type"].str.contains("anomaly", case=False, na=False)].copy()
                                    anomalies["Year"] = year
                                    all_anomalies.append(anomalies)
                                if not all_anomalies:
                                    st.warning("No anomalies found in the uploaded data.")
                                    st.stop()
                                all_anomalies = pd.concat(all_anomalies, ignore_index=True)

                                # Ensure numeric
                                all_anomalies["width [mm]"] = pd.to_numeric(all_anomalies["width [mm]"], errors="coerce")
                                all_anomalies["length [mm]"] = pd.to_numeric(all_anomalies["length [mm]"], errors="coerce")
                                all_anomalies = all_anomalies.dropna(subset=["width [mm]", "length [mm]", "log distance [m]", "clock orientation"])

                                # Find overlapping/matched defects between years
                                # For simplicity, match each defect to the next year only
                                matched_rows = []
                                unmatched_rows = []
                                years = sorted(all_anomalies["Year"].unique())
                                for i, year in enumerate(years[:-1]):
                                    df1 = all_anomalies[all_anomalies["Year"] == year]
                                    df2 = all_anomalies[all_anomalies["Year"] == years[i+1]]
                                    for idx1, row1 in df1.iterrows():
                                        # Find matches in next year
                                        width_tol = row1["width [mm]"] * tolerance_pct
                                        length_tol = row1["length [mm]"] * tolerance_pct
                                        matches = df2[
                                            (abs(df2["width [mm]"] - row1["width [mm]"]) <= width_tol) &
                                            (abs(df2["length [mm]"] - row1["length [mm]"]) <= length_tol) &
                                            (abs(df2["log distance [m]"] - row1["log distance [m]"]) <= length_tol)
                                        ]
                                        if not matches.empty:
                                            for idx2, row2 in matches.iterrows():
                                                matched_rows.append({
                                                    "Year1": year, "Index1": idx1, "Distance1": row1["log distance [m]"], "Orientation1": row1["clock orientation"],
                                                    "Width1": row1["width [mm]"], "Length1": row1["length [mm]"],"Depth1": row1.get("depth [%]", np.nan),
                                                    "Year2": years[i+1], "Index2": idx2, "Distance2": row2["log distance [m]"], "Orientation2": row2["clock orientation"],
                                                    "Width2": row2["width [mm]"], "Length2": row2["length [mm]"],"Depth2": row2.get("depth [%]", np.nan) 
                                                })
                                        else:
                                            unmatched_rows.append({
                                                "Year": year, "Index": idx1, "Distance": row1["log distance [m]"], "Orientation": row1["clock orientation"],
                                                "Width": row1["width [mm]"], "Length": row1["length [mm]"],"Depth": row1.get("depth [%]", np.nan)
                                            })

                                matched_df = pd.DataFrame(matched_rows)
                                unmatched_df = pd.DataFrame(unmatched_rows)
                                st.session_state.defect_matching_results = (matched_df, unmatched_df)

                                # Build the interactive plot
                                fig = go.Figure()

                                # Plot all boxes (unmatched: light red, matched: green)
                                for idx, row in all_anomalies.iterrows():
                                    length_m = row["length [mm]"] / 1000
                                    width_deg = mm_to_deg(row["width [mm]"], inside_diameter_mm)
                                    x0 = row["log distance [m]"] - length_m / 2
                                    x1 = row["log distance [m]"] + length_m / 2
                                    y0 = row["clock orientation"] - width_deg / 2
                                    y1 = row["clock orientation"] + width_deg / 2
                                    is_matched = (matched_df["Index1"] == idx).any() or (matched_df["Index2"] == idx).any()
                                    fillcolor = "rgba(44,160,44,0.5)" if is_matched else "rgba(255,99,71,0.4)"
                                    fig.add_shape(
                                        type="rect",
                                        x0=x0, x1=x1, y0=y0, y1=y1,
                                        line=dict(color="green" if is_matched else "red", width=2),
                                        fillcolor=fillcolor,
                                        layer="above"
                                    )

                                # Legend
                                fig.add_trace(go.Scatter(
                                    x=[None], y=[None],
                                    mode='markers',
                                    marker=dict(color="rgba(44,160,44,0.5)", size=15, symbol='square'),
                                    name="Matched Defect"
                                ))
                                fig.add_trace(go.Scatter(
                                    x=[None], y=[None],
                                    mode='markers',
                                    marker=dict(color="rgba(255,99,71,0.4)", size=15, symbol='square'),
                                    name="Unmatched Defect"
                                ))

                                fig.update_layout(
                                    title="All Years Defect Matching",
                                    xaxis_title="Log Distance (m)",
                                    yaxis_title="Clock Orientation (deg)",
                                    showlegend=True,
                                    hovermode='closest',
                                    height=700
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # Table of matched defects
                                st.subheader("Matched Defects Table")
                                if not matched_df.empty:
                                    # Show only first 10 columns for clarity
                                    display_cols = ["Year1", "Distance1", "Orientation1", "Width1", "Length1", "Year2", "Distance2", "Orientation2", "Width2", "Length2"]
                                    st.dataframe(matched_df[display_cols], use_container_width=True)

                                    # Normalize both DataFrames to have consistent columns
                                    matched_df["Match Status"] = "Matched"
                                    unmatched_df["Match Status"] = "Unmatched"

                                    # For consistency, add placeholder columns to unmatched
                                    unmatched_df["Year2"] = None
                                    unmatched_df["Distance2"] = None
                                    unmatched_df["Orientation2"] = None
                                    unmatched_df["Width2"] = None
                                    unmatched_df["Length2"] = None

                                    # Rename unmatched columns to match matched_df
                                    unmatched_df.rename(columns={
                                        "Year": "Year1",
                                        "Index": "Index1",
                                        "Distance": "Distance1",
                                        "Orientation": "Orientation1",
                                        "Width": "Width1",
                                        "Length": "Length1"
                                    }, inplace=True)

                                    # Combine both
                                    combined_df = pd.concat([matched_df, unmatched_df], ignore_index=True)

                                    # Export to CSV
                                    csv_buffer = io.StringIO()
                                    combined_df.to_csv(csv_buffer, index=False)
                                    csv_data = csv_buffer.getvalue()

                                    # Download button
                                    st.download_button(
                                        label="📥 Download Full Defect Report (Matched + Unmatched)",
                                        data=csv_data,
                                        file_name="defect_matching_full_report.csv",
                                        mime="text/csv"
                                    )

                                else:
                                    st.info("No matched defects found with the selected tolerance.")

                                st.write(f"**Matching tolerance:** {tolerance_pct*100:.1f}% (affects both width and length)")
                    
            
            # Tab 2: Weld Analysis
            with tab2:
                
                # First, let's print the unique values to help debugging
                with st.expander("Debug: Unique anomaly types in data", expanded=False):
                    for year, df in historical_data.items():
                        st.write(f"Year {year} unique values:")
                        st.write(df["component/anomaly type"].unique())

                with st.expander("Weld Position Analysis", expanded=False):
                    st.header("Weld Position Analysis")
                    weld_summary = weld_error_summary(historical_data)
                    
                    if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
                        st.warning("Please upload and preprocess data first.")
                    else:
                        # Get weld data from all years
                        historical_data = st.session_state.historical_data
                        weld_positions_by_year = {}
                        for year, df in historical_data.items():
                            welds = df[df["component/anomaly type"].str.lower().str.contains("weld", na=False)]
                            weld_positions = welds["log distance [m]"].dropna().sort_values().unique()
                            weld_positions_by_year[year] = weld_positions

                        # Let user set matching tolerance
                        with st.form(key="weld_analysis_form"):
                            col1, col2 = st.columns([2, 3])
                            with col1:
                                tolerance = st.slider("Matching tolerance (meters)", 0.0, 1.0, 0.0,step=0.1)
                            with col2:
                                submitted = st.form_submit_button("Update Analysis")
                        
                        # Calculate unmatched welds
                        all_welds = np.concatenate(list(weld_positions_by_year.values()))
                        unmatched_welds_by_year = {}
                        for year, welds in weld_positions_by_year.items():
                            other_years_welds = np.concatenate(
                                [w for y, w in weld_positions_by_year.items() if y != year]
                            )
                            unmatched = []
                            for w in welds:
                                if not np.any(np.abs(other_years_welds - w) <= tolerance):
                                    unmatched.append(w)
                            unmatched_welds_by_year[year] = np.array(unmatched)

                        # Create visualization
                        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                        years = list(weld_positions_by_year.keys())
                        ymin, ymax = 0, 1  # y-limits for vertical lines

                        fig, ax = plt.subplots(figsize=(16, 5))

                        for idx, year in enumerate(years):
                            color = color_cycle[idx % len(color_cycle)]
                            welds = weld_positions_by_year[year]
                            unmatched = unmatched_welds_by_year[year]
                            # Plot all welds as solid or dotted lines
                            for x in welds:
                                ls = 'dotted' if x in unmatched else 'solid'
                                ax.axvline(x, ymin=0, ymax=1, color=color, linestyle=ls, linewidth=2, alpha=0.7)
                            # Add a label for the year (for legend)
                            ax.plot([], [], color=color, label=f'{year}')

                        # Legend for matched/unmatched
                        ax.plot([], [], color='black', linestyle='solid', label='Matched weld')
                        ax.plot([], [], color='black', linestyle='dotted', label='Unmatched weld')

                        ax.set_xlabel("Log Distance (m)")
                        ax.set_yticks([])
                        ax.set_title(f"Weld Position Comparison Across Years (Tolerance : {tolerance} m)")
                        ax.legend(loc='upper center', ncol=len(years)+2)
                        ax.set_xlim(
                            min(np.concatenate(list(weld_positions_by_year.values()))) - 5,
                            max(np.concatenate(list(weld_positions_by_year.values()))) + 5
                        )

                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show unmatched weld statistics
                        st.subheader("Unmatched Weld Statistics")
                        for year, unmatched in unmatched_welds_by_year.items():
                            total_welds = len(weld_positions_by_year[year])
                            unmatched_count = len(unmatched)
                            unmatched_pct = (unmatched_count / total_welds) * 100 if total_welds > 0 else 0
                            st.write(
                                f"**{year}**: {unmatched_count} unmatched welds "
                                f"({unmatched_pct:.1f}% of {total_welds} total welds)"
                            )
                            if unmatched_count > 0:
                                st.write(f"Positions: {', '.join(map(lambda x: f'{x:.1f}m', unmatched))}")
            
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

            with tab5:
                #sub menu linear corrosion rate and future depth prediction
                with st.expander("Linear Corrosion Rate & Future Depth Prediction", expanded=False):
                    st.header("Corrosion Rate & Future Depth Prediction")
                    if (
                        'defect_matching_results' in st.session_state
                        and st.session_state.defect_matching_results is not None
                    ):
                        matched_df, unmatched_df = st.session_state.defect_matching_results

                        # Ensure required columns exist and are numeric
                        for col in ["Depth1", "Depth2"]:
                            if col not in matched_df.columns:
                                st.warning("Matched defects must include depth columns (Depth1, Depth2).")
                                st.stop()
                            matched_df[col] = pd.to_numeric(matched_df[col], errors="coerce")

                        matched_df["Year1"] = pd.to_numeric(matched_df["Year1"], errors="coerce")
                        matched_df["Year2"] = pd.to_numeric(matched_df["Year2"], errors="coerce")
                        # Calculate corrosion rate (mm/year)
                        matched_df["Corrosion_Rate_mm_per_year"] = (
                            (matched_df["Depth2"] - matched_df["Depth1"]) /
                            (matched_df["Year2"] - matched_df["Year1"])
                        )

                        st.write("**Corrosion rate is calculated as:**")
                        st.latex(r"\textbf{Corrosion Rate (mm/year)} = \frac{\text{Depth}_2 - \text{Depth}_1}{\text{Year}_2 - \text{Year}_1}")
                        st.latex(r"\textbf{Predicted Depth}_{\text{future}} = \text{Depth}_2 + \text{Corrosion Rate} \times (\text{Future Year} - \text{Year}_2)")

                        st.subheader("Corrosion Rate Table")
                        st.dataframe(
                            matched_df[
                                ["Year1", "Depth1", "Year2", "Depth2", "Corrosion_Rate_mm_per_year"]
                            ],
                            use_container_width=True
                        )

                        # Predict future corrosion depth
                        future_year = st.number_input(
                            "Predict corrosion depth for year:",
                            min_value=int(matched_df["Year2"].max()) + 1,
                            value=int(matched_df["Year2"].max()) + 5,
                            step=1,
                        )
                        matched_df["Predicted_Depth"] = matched_df["Depth2"] + matched_df["Corrosion_Rate_mm_per_year"] * (future_year - matched_df["Year2"])

                        st.subheader(f"Predicted Depths for {future_year}")
                        st.dataframe(
                            matched_df[
                                ["Year2", "Depth2", "Corrosion_Rate_mm_per_year", "Predicted_Depth"]
                            ],
                            use_container_width=True
                        )

                        # Visualize distribution
                        st.subheader("Corrosion Rate Distribution")
                        st.bar_chart(matched_df["Corrosion_Rate_mm_per_year"].dropna())

                        st.subheader("Predicted Depth Distribution")
                        st.bar_chart(matched_df["Predicted_Depth"].dropna())

                        # Download report
                        st.download_button(
                            label="Download Corrosion Rate & Prediction Report (CSV)",
                            data=matched_df.to_csv(index=False),
                            file_name="corrosion_rate_prediction_report.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("Run defect matching first to enable corrosion rate analysis.")
                
                #sub menu machine learning corrosion prediction
                with st.expander("Machine Learning Corrosion Prediction", expanded=False):
                    st.write("This section uses a Random Forest regression model to predict future corrosion depth based on matched defect features.")

                    if (
                        'defect_matching_results' in st.session_state
                        and st.session_state.defect_matching_results is not None
                    ):
                        matched_df, _ = st.session_state.defect_matching_results

                        # Ensure required columns exist and are numeric
                        required_cols = ["Year1", "Year2", "Depth1", "Depth2", "Width1", "Length1"]
                        for col in required_cols:
                            if col not in matched_df.columns:
                                st.warning(f"Matched defects must include column: {col}")
                                st.stop()
                            matched_df[col] = pd.to_numeric(matched_df[col], errors="coerce")

                        # Add a time difference feature
                        matched_df["Years_Elapsed"] = matched_df["Year2"] - matched_df["Year1"]

                        # Prepare features and target
                        feature_cols = ["Depth1", "Width1", "Length1", "Years_Elapsed"]
                        X = matched_df[feature_cols].dropna()
                        y = matched_df.loc[X.index, "Depth2"]

                        if len(X) < 10:
                            st.info("Not enough matched records for ML training. At least 10 required.")
                            st.stop()

                        # Train/test split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                        # Train Random Forest
                        rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf.fit(X_train, y_train)
                        y_pred = rf.predict(X_test)

                        # Show metrics
                        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.3f} | **R²:** {r2_score(y_test, y_pred):.3f}")

                        # Predict for user-selected future year
                        future_year = st.number_input(
                            "Predict corrosion depth for year (ML):",
                            min_value=int(matched_df["Year2"].max()) + 1,
                            value=int(matched_df["Year2"].max()) + 5,
                            step=1,
                        )

                        # Future prediction input
                        pred_X = matched_df[["Depth1", "Width1", "Length1", "Year1"]].copy()
                        pred_X["Years_Elapsed"] = future_year - pred_X["Year1"]
                        pred_X = pred_X[["Depth1", "Width1", "Length1", "Years_Elapsed"]]  # reorder to match model

                        pred_depth = rf.predict(pred_X)
                        matched_df["ML_Predicted_Depth"] = pred_depth

                        # Show prediction table
                        st.subheader(f"Predicted Depths for {future_year} (ML)")
                        st.dataframe(
                            matched_df[
                                ["Year2", "Depth2", "ML_Predicted_Depth"]
                            ],
                            use_container_width=True
                        )

                        # Plot distribution
                        st.subheader("ML Predicted Depth Distribution")
                        st.bar_chart(matched_df["ML_Predicted_Depth"].dropna())

                        # Download report
                        st.download_button(
                            label="Download ML Corrosion Prediction Report (CSV)",
                            data=matched_df.to_csv(index=False),
                            file_name="ml_corrosion_prediction_report.csv",
                            mime="text/csv"
                        )

                    else:
                        st.info("Run defect matching first to enable ML corrosion prediction.")
                
                #sub menu burst pressure cga and thinning mechanism cga
                with st.expander("Burst Pressure CGA", expanded=False):
                    st.write("This section estimates remaining life based on predicted corrosion growth and API 579 burst pressure assessment.")

                    if (
                        'defect_matching_results' in st.session_state
                        and st.session_state.defect_matching_results is not None
                    ):
                        matched_df, _ = st.session_state.defect_matching_results

                        required_cols = ["Length2", "Width2", "Depth2"]
                        for col in required_cols:
                            if col not in matched_df.columns:
                                st.warning(f"Missing required column: {col}")
                                st.stop()
                            matched_df[col] = pd.to_numeric(matched_df[col], errors="coerce")

                        df = matched_df.dropna(subset=required_cols).copy()
                        if df.empty:
                            st.warning("No valid defect records found.")
                            st.stop()

                        st.subheader("Pipeline Parameters")

                        # User input for parameters
                        with st.form("burst_cga_form"):
                            st.subheader("Pipeline Parameters (Burst Pressure)")
                            D_mm = st.number_input("Pipe Diameter D (mm)", min_value=100.0, value=inside_diameter_mm, key="burst_d")
                            t_mm = st.number_input("Wall Thickness t (mm)", min_value=1.0, value=12.7, key="burst_t")
                            MAOP = st.number_input("Maximum Allowable Operating Pressure (MAOP) [psi]", min_value=100, value=1000, key="burst_maop")
                            SMYS = st.number_input("Specified Minimum Yield Strength (SMYS) [psi]", min_value=10000, value=52000, key="burst_sm")
                            safety_factor = st.number_input("Safety Factor", min_value=1.0, max_value=2.0, value=1.1, step=0.05, key="burst_sf")

                            run_cga = st.form_submit_button("Run CGA Analysis")

                        if run_cga:
                            # Convert D and t to inches
                            D = D_mm / 25.4
                            t = t_mm / 25.4

                            # Feature engineering
                            df["aspect_ratio"] = df["Length2"] / df["Width2"]
                            df["volumetric_loss"] = df["Length2"] * df["Width2"] * df["Depth2"]
                            df["depth_ratio"] = df["Depth2"] / t_mm
                            df["pressure"] = 50  # Placeholder
                            df["coating_age"] = np.random.randint(0, 20, len(df))

                            # Model growth rate (simulated target)
                            feature_cols = ["Length2", "Width2", "Depth2", "pressure", "coating_age",
                                            "aspect_ratio", "volumetric_loss", "depth_ratio"]
                            X = df[feature_cols]
                            y = np.abs(np.random.normal(0.2, 0.05, len(df)))  # Simulated growth

                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(X, y)
                            growth_pred = model.predict(X)

                            # Clamp predicted growth to realistic range
                            growth_pred = np.clip(growth_pred, 0.05, 1.5)  # mm/year

                            # API 579 burst pressure formula
                            def api_579_burst_pressure(d_in, t_in, D_in, SMYS_psi):
                                M = np.sqrt(1 + 0.8 * (d_in / t_in)**2)
                                Pf = (2 * SMYS_psi * t_in / D_in) * ((1 - 0.85 * d_in / t_in) / (1 - 0.85 * d_in / t_in / M))
                                return Pf

                            # Remaining life simulation
                            remaining_lives = []
                            for idx, row in df.iterrows():
                                current_depth_mm = row["Depth2"]
                                growth_rate_mm = growth_pred[idx]
                                life = 50  # max years

                                for year in range(1, 51):
                                    depth_mm = current_depth_mm + growth_rate_mm * year
                                    depth_in = depth_mm / 25.4
                                    Pf = api_579_burst_pressure(depth_in, t, D, SMYS)
                                    if Pf <= safety_factor * MAOP:
                                        life = year
                                        break

                                remaining_lives.append(life)

                            df["Predicted_Growth_Rate"] = growth_pred
                            df["Remaining_Life"] = remaining_lives

                            st.subheader("Remaining Life Prediction (API 579)")
                            st.dataframe(df[["Depth2", "Predicted_Growth_Rate", "Remaining_Life"]], use_container_width=True)

                            st.subheader("Remaining Life Distribution")
                            fig = px.histogram(df, x="Remaining_Life", nbins=20, title="Remaining Life (years)")
                            st.plotly_chart(fig, use_container_width=True)

                            st.download_button(
                                label="Download Burst Pressure CGA Report (CSV)",
                                data=df.to_csv(index=False),
                                file_name="burst_pressure_cga_report.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("Fill in the pipeline parameters above and click the button to begin analysis.")

                    else:
                        st.info("Run defect matching first to enable CGA analysis.")

                with st.expander("Thinning Mechanism CGA", expanded=False):
                    st.write("This section calculates remaining life for each matched defect based on wall thinning, using hoop stress and minimum wall thickness criteria.")

                    if (
                        'defect_matching_results' in st.session_state
                        and st.session_state.defect_matching_results is not None
                    ):
                        matched_df, _ = st.session_state.defect_matching_results
                        df = matched_df.copy()
                        df = df.dropna(subset=["Depth2"])  # Ensure valid data

                        with st.form("thinning_form"):
                            st.subheader("Pipeline Parameters (Thinning)")
                            D_mm = st.number_input("Pipe Diameter D (mm)", min_value=100.0, value=inside_diameter_mm, key="thin_d") #later will be converted to inches
                            t_current_mm = st.number_input("Current Wall Thickness t (mm)", min_value=1.0, value=12.7, key="thin_t")
                            t_min_mm = st.number_input("Minimum Required Thickness t_min (mm)", min_value=1.0, value=6.35, key="thin_t_min")
                            P = st.number_input("Operating Pressure P [psi]", min_value=100, value=1000, key="thin_p")
                            allowable_stress = st.number_input("Allowable Hoop Stress [psi]", min_value=10000, value=20000, key="thin_hoop")

                            st.subheader("Operating Conditions")
                            temperature = st.number_input("Temperature (°C)", value=80, key="thin_temp")
                            flow_rate = st.number_input("Flow Rate (m/s)", value=10.0, key="thin_flow")
                            coating_age = st.slider("Coating Age (years)", min_value=0, max_value=30, value=10, key="thin_coating")

                            # Submit button — nothing runs until this is clicked
                            run_thinning = st.form_submit_button("Run Thinning Analysis")

                        if run_thinning:
                            # Convert dimensions to inches
                            D = D_mm / 25.4
                            t_current = t_current_mm / 25.4
                            t_min = t_min_mm / 25.4

                            # Simulate training data and train model
                            np.random.seed(42)
                            train_data = pd.DataFrame({
                                'temp': np.random.normal(80, 10, 500),
                                'flow_rate': np.random.normal(10, 2, 500),
                                'coating_age': np.random.randint(0, 20, 500),
                                'thinning_rate': np.abs(np.random.normal(0.02, 0.005, 500))  # in/year
                            })

                            feature_cols = ['temp', 'flow_rate', 'coating_age']
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(train_data[feature_cols], train_data['thinning_rate'])

                            # Assign input values per defect
                            df["temp"] = temperature
                            df["flow_rate"] = flow_rate
                            df["coating_age"] = coating_age

                            # Predict thinning rate
                            X_test = df[feature_cols]
                            thinning_rate = model.predict(X_test)
                            thinning_rate = np.clip(thinning_rate, 0.005, 0.05)  # Clamp for realism

                            # Calculate results
                            df["Thinning_Rate_in_per_year"] = thinning_rate
                            df["Remaining_Life_years"] = (t_current - t_min) / thinning_rate
                            df["Wall_Thickness_EOL"] = t_current - df["Thinning_Rate_in_per_year"] * df["Remaining_Life_years"]
                            df["Hoop_Stress_EOL_psi"] = (P * D) / (2 * df["Wall_Thickness_EOL"])
                            df["Hoop_Stress_Exceeded"] = df["Hoop_Stress_EOL_psi"] > allowable_stress

                            # Display
                            st.subheader("Thinning Analysis Results (per defect)")
                            display_cols = [
                                "Depth2", "Thinning_Rate_in_per_year", "Remaining_Life_years",
                                "Wall_Thickness_EOL", "Hoop_Stress_EOL_psi", "Hoop_Stress_Exceeded"
                            ]
                            st.dataframe(df[display_cols].round(3), use_container_width=True)

                            # Export
                            st.download_button(
                                label="Download Thinning CGA Report (CSV)",
                                data=df[display_cols + ["Depth2"]].to_csv(index=False),
                                file_name="thinning_cga_per_defect.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("Adjust pipeline and operating conditions, then click the button to run thinning analysis.")

                    else:
                        st.warning("Defect matching data not found. Please run defect matching first.")
        
        else:
            st.warning("Please upload at least two Excel files to enable comparative analysis.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your data format and try again.")
else:
    # Welcome screen when no files are uploaded
    st.write("## 👋 Welcome to the Inline Inspection App!")
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

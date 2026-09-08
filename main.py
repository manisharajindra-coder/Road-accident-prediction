"""
Road Accident Hotspot Detection & Analysis Dashboard
A comprehensive Streamlit application for analyzing road accidents and infrastructure data
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium, folium_static
from folium.plugins import MarkerCluster, HeatMap
import numpy as np
import warnings
import os
import requests
import json
from io import BytesIO
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# Try importing Prophet for forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths - Update these to match your data location
ACCIDENT_DATA_PATH = "Accident_Hotspot_Dataset.csv"
ROAD_TRANSPORT_PATH = "Road_Transport_Statistics.csv"
MOTOR_TRANSPORT_PATH = "Motor_Transport_Statistics.csv"

# Google Street View API Configuration
# Get your API key from: https://developers.google.com/maps/documentation/streetview/get-api-key
# Option 1: Paste your API key directly here (easiest method)
# PASTE YOUR API KEY HERE between the quotes
GOOGLE_STREETVIEW_API_KEY = "AIzaSyCrVneHKn4SadnIrnXBoh3wduFeYXxSURc"

# Google Maps API Configuration (for Traffic Layer)
# Can use the same API key as Street View or a separate one
# Get your API key from: https://console.cloud.google.com/google/maps-apis
# Option 1: Paste your API key directly here (easiest method)
# PASTE YOUR API KEY HERE between the quotes
GOOGLE_MAPS_API_KEY = "AIzaSyCrVneHKn4SadnIrnXBoh3wduFeYXxSURc"

# Option 2: Use environment variables (if you prefer)
# If you leave the above empty, it will try to read from environment variables:
# GOOGLE_STREETVIEW_API_KEY = os.getenv("GOOGLE_STREETVIEW_API_KEY", "")
# GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", GOOGLE_STREETVIEW_API_KEY)

# If you want to use the same key for both, you can do:
if not GOOGLE_MAPS_API_KEY:
    GOOGLE_MAPS_API_KEY = GOOGLE_STREETVIEW_API_KEY

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Road Accident Analytics Dashboard",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================

st.markdown("""
    <style>
    .stApp {
        background-color: #eff6e0;
        color: #000000 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #aec3b0;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #1b4332 !important;
        font-weight: bold;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-weight: bold !important;
    }
    .stButton>button {
        background-color: #8ecae6;
        color: #000000 !important;
        border-radius: 10px;
        padding: 0.5em 1em;
        border: 1px solid #ccc;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ffb703;
        color: #000000 !important;
    }
    .card {
        background: white;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 12px;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# CHART STYLING
# =============================================================================

plt.style.use("seaborn-v0_8")
plt.rcParams.update({
    "figure.facecolor": "#f4f6f9",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "text.color": "#000000",
    "axes.titleweight": "bold",
    "axes.titlecolor": "#1f2a44",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "grid.color": "#e6e6e6"
})

sns.set_theme(
    style="whitegrid",
    palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def apply_plotly_style(fig, title=""):
    """Apply consistent styling to Plotly figures"""
    fig.update_layout(
        title=title,
        title_font=dict(size=18, color="#1f2a44", family="Arial Black"),
        font=dict(color="#000000", size=12),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f4f6f9",
        xaxis=dict(showgrid=True, gridcolor="#e6e6e6", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#e6e6e6", zeroline=False),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def safe_column_check(df, column_name):
    """Safely check if column exists in dataframe"""
    return (df is not None) and (column_name in df.columns)


def create_metric_card(label, value, delta=None):
    """Create a styled metric card"""
    if delta is not None:
        st.metric(label=label, value=value, delta=delta)
    else:
        st.metric(label=label, value=value)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================


@st.cache_data
def load_accident_data(path=None):
    """Load accident hotspot dataset"""
    if path is None:
        path = ACCIDENT_DATA_PATH
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Accident dataset not found at: {path}")
        st.info("Please update ACCIDENT_DATA_PATH in the configuration section")
        return None
    except Exception as e:
        st.error(f"Error loading accident data: {e}")
        return None


@st.cache_data
def load_road_transport_data(path=None):
    """Load road transport statistics dataset"""
    if path is None:
        path = ROAD_TRANSPORT_PATH
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if 'Name of the States' in df.columns:
            df['Name of the States'] = df['Name of the States'].astype(
                str).str.strip()
        return df
    except FileNotFoundError:
        st.warning(f"Road transport data not found at: {path}")
        return None
    except Exception as e:
        st.warning(f"Error loading road transport data: {e}")
        return None


@st.cache_data
def load_motor_transport_data(path=None):
    """Load motor transport statistics dataset"""
    if path is None:
        path = MOTOR_TRANSPORT_PATH
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        if 'State' in df.columns:
            df['State'] = df['State'].astype(str).str.strip()
        return df
    except FileNotFoundError:
        st.warning(f"Motor transport data not found at: {path}")
        return None
    except Exception as e:
        st.warning(f"Error loading motor transport data: {e}")
        return None

# =============================================================================
# GOOGLE STREET VIEW INTEGRATION
# =============================================================================


def get_street_view_image(latitude, longitude, heading=0, pitch=0, fov=90, size="640x640"):
    """
    Get Google Street View image for a given location

    Parameters:
    - latitude: Latitude of the location
    - longitude: Longitude of the location
    - heading: Compass heading (0-360)
    - pitch: Up/down angle (-90 to 90)
    - fov: Field of view (10-100)
    - size: Image size (max 640x640)

    Returns:
    - PIL Image object or None if error
    """
    if not GOOGLE_STREETVIEW_API_KEY:
        return None

    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": size,
        "location": f"{latitude},{longitude}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": GOOGLE_STREETVIEW_API_KEY
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            return None
    except Exception as e:
        return None


def display_street_view_analysis(latitude, longitude, accident_data):
    """
    Display Street View analysis for accident hotspots

    Parameters:
    - latitude: Latitude of hotspot
    - longitude: Longitude of hotspot
    - accident_data: Dictionary with accident information
    """
    if not GOOGLE_STREETVIEW_API_KEY:
        st.warning(
            "Google Street View API key not configured. Please set GOOGLE_STREETVIEW_API_KEY environment variable.")
        st.info(
            "Get your API key from: https://developers.google.com/maps/documentation/streetview/get-api-key")
        return

    st.subheader("Street View Traffic Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**View 1: Forward Direction**")
        img1 = get_street_view_image(latitude, longitude, heading=0, pitch=0)
        if img1:
            st.image(img1, use_container_width=True)
        else:
            st.error("Street View not available for this location")

    with col2:
        st.write("**View 2: Right Side**")
        img2 = get_street_view_image(latitude, longitude, heading=90, pitch=0)
        if img2:
            st.image(img2, use_container_width=True)

    with col3:
        st.write("**View 3: Left Side**")
        img3 = get_street_view_image(latitude, longitude, heading=270, pitch=0)
        if img3:
            st.image(img3, use_container_width=True)

    # Traffic condition analysis
    st.markdown("### Traffic Condition Analysis")
    analysis_text = f"""
    **Location:** {accident_data.get('City', 'N/A')}, {accident_data.get('State', 'N/A')}
    
    **Accident Details:**
    - Severity: {accident_data.get('Severity', 'N/A')}
    - Vehicle Type: {accident_data.get('Vehicle Type', 'N/A')}
    - Time: {accident_data.get('Time', 'N/A')}
    - Casualties: {accident_data.get('Casualties', 0)}
    
    **Road Safety Observations:**
    - Analyze road width, lane markings, and signage visibility
    - Check for pedestrian crossings and traffic signals
    - Assess road surface condition and lighting
    - Evaluate visibility of road signs and speed limits
    """
    st.markdown(analysis_text)

# =============================================================================
# FILTERING FUNCTIONS
# =============================================================================


def setup_sidebar_filters(df):
    """Setup sidebar filters for data"""
    st.sidebar.title("Road Accident Filter")

    # State filter
    state_options = ['All']
    state_col = 'State Name' if 'State Name' in df.columns else (
        'State_Name' if 'State_Name' in df.columns else None)
    if state_col:
        state_options += sorted(df[state_col].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox(
        "Select State", options=state_options)

    # Year filter
    year_options = ['All']
    if 'Year' in df.columns:
        try:
            years = pd.to_numeric(df['Year'], errors='coerce').dropna().astype(
                int).unique().tolist()
            year_options += sorted(years)
        except Exception:
            year_options += sorted(df['Year'].dropna().unique().tolist())
    selected_year = st.sidebar.selectbox("Select Year", options=year_options)

    # Severity filter
    severity_options = df['Accident Severity'].dropna().unique(
    ).tolist() if 'Accident Severity' in df.columns else []
    selected_severity = st.sidebar.multiselect(
        "Select Severity", options=severity_options, default=severity_options)

    # Vehicle type filter
    vehicle_options = df['Vehicle Type Involved'].dropna().unique(
    ).tolist() if 'Vehicle Type Involved' in df.columns else []
    selected_vehicle_types = st.sidebar.multiselect(
        "Select Vehicle Type", options=vehicle_options, default=vehicle_options)

    return selected_state, selected_year, selected_severity, selected_vehicle_types, state_col


def apply_filters(df, selected_state, selected_year, selected_severity, selected_vehicle_types, state_col):
    """Apply filters to dataframe"""
    filtered_df = df.copy()

    if selected_state != "All" and state_col:
        filtered_df = filtered_df[filtered_df[state_col] == selected_state]

    if selected_year != "All" and "Year" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Year"] == selected_year]

    if selected_severity and "Accident Severity" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Accident Severity"].isin(
            selected_severity)]

    if selected_vehicle_types and "Vehicle Type Involved" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Vehicle Type Involved"].isin(
            selected_vehicle_types)]

    return filtered_df

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def display_kpi_metrics(filtered_df):
    """Display Key Performance Indicator metrics"""
    total_accidents = len(filtered_df)

    # Derive injuries from severity and casualties
    if 'Accident Severity' in filtered_df.columns and 'Number of Casualties' in filtered_df.columns:
        filtered_df['Derived Serious Injuries'] = filtered_df.apply(
            lambda row: row['Number of Casualties'] if row['Accident Severity'] == 'Serious' else 0, axis=1
        )
        filtered_df['Derived Minor Injuries'] = filtered_df.apply(
            lambda row: row['Number of Casualties'] if row['Accident Severity'] == 'Minor' else 0, axis=1
        )
    else:
        filtered_df['Derived Serious Injuries'] = 0
        filtered_df['Derived Minor Injuries'] = 0

    # Calculate totals
    total_fatalities = int(filtered_df['Number of Fatalities'].sum(
    )) if 'Number of Fatalities' in filtered_df.columns else 0
    total_serious_injuries = int(filtered_df['Derived Serious Injuries'].sum())
    total_minor_injuries = int(filtered_df['Derived Minor Injuries'].sum())

    # Calculate percentages
    severity_counts = filtered_df['Accident Severity'].value_counts(
    ) if 'Accident Severity' in filtered_df.columns else pd.Series()
    severity_percentages = (
        severity_counts / total_accidents * 100).round(2) if total_accidents > 0 else {}

    fatal_pct = severity_percentages.get("Fatal", 0)
    serious_pct = severity_percentages.get("Serious", 0)
    minor_pct = severity_percentages.get("Minor", 0)

    fatality_rate = (total_fatalities / total_accidents *
                     100) if total_accidents > 0 else 0
    serious_rate = (total_serious_injuries / total_accidents *
                    100) if total_accidents > 0 else 0
    minor_rate = (total_minor_injuries / total_accidents *
                  100) if total_accidents > 0 else 0

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Accidents", f"{total_accidents:,}")
    col2.metric("Total Fatalities", f"{total_fatalities:,}")
    col3.metric("Serious Injuries", f"{total_serious_injuries:,}")
    col4.metric("Minor Injuries", f"{total_minor_injuries:,}")

    colA, colB, colC = st.columns(3)
    colA.metric("Fatality Rate (%)", f"{fatality_rate:.2f}%")
    colB.metric("Serious Injury Rate (%)", f"{serious_rate:.2f}%")
    colC.metric("Minor Injury Rate (%)", f"{minor_rate:.2f}%")

    colX, colY, colZ = st.columns(3)
    colX.metric("Fatal Accidents (%)", f"{fatal_pct:.2f}%")
    colY.metric("Serious Accidents (%)", f"{serious_pct:.2f}%")
    colZ.metric("Minor Accidents (%)", f"{minor_pct:.2f}%")

    return {
        'total_accidents': total_accidents,
        'total_fatalities': total_fatalities,
        'total_serious_injuries': total_serious_injuries,
        'total_minor_injuries': total_minor_injuries,
        'fatality_rate': fatality_rate
    }


def display_city_analysis(filtered_df):
    """Display top cities with most accidents"""
    city_col = 'City Name' if 'City Name' in filtered_df.columns else (
        'City_Name' if 'City_Name' in filtered_df.columns else None)
    if city_col:
        st.subheader("Top 10 Cities with Most Accidents")
        top_cities = filtered_df[city_col].value_counts().nlargest(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        top_cities.plot(kind='bar', ax=ax, color="#1f77b4", edgecolor="black")
        ax.set_title("Top 10 Cities with Most Accidents",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("City", fontsize=12)
        ax.set_ylabel("Number of Accidents", fontsize=12)
        ax.set_xticklabels(top_cities.index, rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)


def display_vehicle_analysis(filtered_df):
    """Display vehicle type analysis"""
    if safe_column_check(filtered_df, 'Vehicle Type Involved') and safe_column_check(filtered_df, 'Number of Casualties'):
        st.subheader("Casualties by Vehicle Type")
        vehicle_casualties = (
            filtered_df.groupby("Vehicle Type Involved")[
                "Number of Casualties"]
            .sum().reset_index()
            .sort_values(by="Number of Casualties", ascending=False)
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(vehicle_casualties, x="Vehicle Type Involved", y="Number of Casualties",
                             text="Number of Casualties", color="Number of Casualties",
                             color_continuous_scale="Blues")
            fig_bar = apply_plotly_style(fig_bar, "Casualties by Vehicle Type")
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_pie = px.pie(vehicle_casualties, names="Vehicle Type Involved",
                             values="Number of Casualties", hole=0.3)
            fig_pie = apply_plotly_style(fig_pie, "Proportion of Casualties")
            st.plotly_chart(fig_pie, use_container_width=True)


def display_severity_distribution(filtered_df):
    """Display accident severity distribution"""
    if safe_column_check(filtered_df, "Accident Severity"):
        severity_counts = filtered_df["Accident Severity"].value_counts(
        ).reset_index()
        severity_counts.columns = ["Accident Severity", "Count"]

        severity_color_map = {"Fatal": "red",
                              "Serious": "orange", "Minor": "green"}

        fig_pie = px.pie(
            severity_counts,
            names="Accident Severity",
            values="Count",
            title="Accident Severity Distribution",
            color="Accident Severity",
            color_discrete_map=severity_color_map
        )
        fig_pie.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#f4f6f9",
            font=dict(color="#1f2a44"),
            title_font=dict(size=18, color="#1f2a44")
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        return severity_color_map
    return {}


def display_time_heatmap(filtered_df):
    """Display heatmap of accidents by day and time"""
    if safe_column_check(filtered_df, 'Day of Week') and safe_column_check(filtered_df, 'Time of Day'):
        st.subheader("Heatmap: Accidents by Day and Time of Day")
        pivot_table = pd.pivot_table(
            filtered_df, index='Day of Week', columns='Time of Day', aggfunc='size', fill_value=0
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_table, cmap='OrRd', annot=True, fmt='d', ax=ax)
        ax.set_title("Accidents by Day & Time", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)


def display_alcohol_analysis(filtered_df):
    """Display alcohol involvement analysis"""
    if safe_column_check(filtered_df, "Alcohol Involvement"):
        st.subheader("Alcohol Involvement in Accidents")
        counts_alc = filtered_df['Alcohol Involvement'].value_counts()
        if counts_alc.sum() > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            counts_alc.plot(kind='bar', ax=ax, color="#2ca02c")
            ax.set_title("Alcohol Involvement Distribution",
                         fontsize=14, fontweight="bold")
            ax.set_xlabel("Alcohol Involvement", fontsize=12)
            ax.set_ylabel("Number of Accidents", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)


def add_traffic_layer_to_map(folium_map, api_key):
    """Add Google Traffic Layer to Folium map"""
    if not api_key:
        return folium_map

    try:
        # Add Google Traffic Layer using JavaScript
        traffic_layer_html = f"""
        <script>
        function initTrafficLayer() {{
            if (typeof google !== 'undefined' && google.maps) {{
                var trafficLayer = new google.maps.TrafficLayer();
                trafficLayer.setMap(map);
            }}
        }}
        </script>
        <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=visualization&callback=initTrafficLayer" async defer></script>
        """
        folium_map.get_root().html.add_child(folium.Element(traffic_layer_html))
    except Exception:
        pass

    return folium_map


def create_traffic_density_map(filtered_df, severity_color_map):
    """Create a map with Google Traffic Layer overlay"""
    # Find latitude and longitude columns
    lat_col = None
    lon_col = None

    possible_lat_cols = ['Latitude', 'latitude', 'LATITUDE', 'lat', 'LAT']
    possible_lon_cols = ['Longitude', 'longitude',
                         'LONGITUDE', 'lon', 'LON', 'lng', 'LNG']

    for col in filtered_df.columns:
        if col in possible_lat_cols:
            lat_col = col
        if col in possible_lon_cols:
            lon_col = col

    if not lat_col or not lon_col:
        return None

    try:
        filtered_df['lat_clean'] = pd.to_numeric(
            filtered_df[lat_col], errors='coerce')
        filtered_df['lon_clean'] = pd.to_numeric(
            filtered_df[lon_col], errors='coerce')

        valid_coords = filtered_df[
            (filtered_df['lat_clean'].notna()) &
            (filtered_df['lon_clean'].notna()) &
            (filtered_df['lat_clean'].between(-90, 90)) &
            (filtered_df['lon_clean'].between(-180, 180))
        ].copy()

        if len(valid_coords) == 0:
            return None

        center_lat = valid_coords['lat_clean'].mean()
        center_lon = valid_coords['lon_clean'].mean()

        # Create map with Google Maps tiles
        traffic_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles=None
        )

        # Add Google Maps tile layer
        if GOOGLE_MAPS_API_KEY:
            folium.TileLayer(
                tiles=f'https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}&key={GOOGLE_MAPS_API_KEY}',
                attr='Google Maps',
                name='Google Maps',
                overlay=False,
                control=True
            ).add_to(traffic_map)
        else:
            folium.TileLayer('OpenStreetMap').add_to(traffic_map)

        # Add Google Traffic Layer
        if GOOGLE_MAPS_API_KEY:
            traffic_layer_js = f"""
            <script src="https://maps.googleapis.com/maps/api/js?key={GOOGLE_MAPS_API_KEY}&libraries=visualization" async defer></script>
            <script>
            var mapElement = document.getElementById('map');
            if (mapElement && typeof google !== 'undefined') {{
                var map = L.map('map').setView([{center_lat}, {center_lon}], 10);
                var trafficLayer = new google.maps.TrafficLayer();
                // Note: This requires Google Maps JavaScript API integration
            }}
            </script>
            """
            traffic_map.get_root().html.add_child(folium.Element(traffic_layer_js))

        return traffic_map
    except Exception as e:
        return None


def display_traffic_density_analysis(filtered_df):
    """Display traffic density analysis with Google Traffic API integration"""
    # Find latitude and longitude columns
    lat_col = None
    lon_col = None

    possible_lat_cols = ['Latitude', 'latitude', 'LATITUDE', 'lat', 'LAT']
    possible_lon_cols = ['Longitude', 'longitude',
                         'LONGITUDE', 'lon', 'LON', 'lng', 'LNG']

    for col in filtered_df.columns:
        if col in possible_lat_cols:
            lat_col = col
        if col in possible_lon_cols:
            lon_col = col

    if not lat_col or not lon_col:
        st.warning(
            "Latitude and Longitude columns required for traffic density analysis.")
        return

    st.subheader("Real-Time Traffic Conditions")

    if not GOOGLE_MAPS_API_KEY:
        st.warning(
            "Google Maps API key not configured. Traffic density visualization requires an API key.")
        st.info("""
        **To enable traffic density visualization:**
        1. Get a Google Maps API key from: https://console.cloud.google.com/google/maps-apis
        2. Enable the following APIs:
           - Maps JavaScript API
           - Traffic Layer API
        3. Set the environment variable: `GOOGLE_MAPS_API_KEY=your-api-key`
        """)

        # Show traffic density heatmap based on accident density as alternative
        st.subheader("Accident Density Heatmap (Traffic Correlation)")
        display_accident_density_heatmap(filtered_df, lat_col, lon_col)
        return

    try:
        filtered_df['lat_clean'] = pd.to_numeric(
            filtered_df[lat_col], errors='coerce')
        filtered_df['lon_clean'] = pd.to_numeric(
            filtered_df[lon_col], errors='coerce')

        valid_coords = filtered_df[
            (filtered_df['lat_clean'].notna()) &
            (filtered_df['lon_clean'].notna()) &
            (filtered_df['lat_clean'].between(-90, 90)) &
            (filtered_df['lon_clean'].between(-180, 180))
        ].copy()

        if len(valid_coords) == 0:
            st.error("No valid coordinates found for traffic analysis.")
            return

        center_lat = valid_coords['lat_clean'].mean()
        center_lon = valid_coords['lon_clean'].mean()

        # Create embedded Google Maps with Traffic Layer
        st.info(
            "**Traffic Color Legend:** Green = Normal, Yellow = Slow, Red = Heavy/Stopped")

        # Create HTML for Google Maps embed (simpler, more reliable)
        map_html = f"""
        <div style="width: 100%; height: 600px; border: 2px solid #4CAF50; border-radius: 8px; margin-bottom: 20px;">
            <iframe
                width="100%"
                height="100%"
                frameborder="0"
                style="border:0; border-radius: 6px;"
                src="https://www.google.com/maps/embed/v1/view?key={GOOGLE_MAPS_API_KEY}&center={center_lat},{center_lon}&zoom=11&maptype=roadmap"
                allowfullscreen>
            </iframe>
        </div>
        <p style="text-align: center; color: #666; font-size: 12px;">
            <strong>Note:</strong> To see real-time traffic, open this location in 
            <a href="https://www.google.com/maps/@{center_lat},{center_lon},11z" target="_blank">Google Maps</a> 
            and enable the traffic layer.
        </p>
        """

        st.markdown(map_html, unsafe_allow_html=True)

        # Add link to Google Maps with traffic layer
        google_maps_url = f"https://www.google.com/maps/@{center_lat},{center_lon},11z/data=!5m1!1e1"
        st.markdown(f"""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; margin-bottom: 20px;">
            <strong>Interactive Traffic Map:</strong> 
            <a href="{google_maps_url}" target="_blank" style="color: #2e7d32; font-weight: bold;">
                Open in Google Maps with Traffic Layer
            </a>
            <br>
            <small>Click the link above to view real-time traffic conditions in Google Maps</small>
        </div>
        """, unsafe_allow_html=True)

        # Show accident density heatmap as additional visualization
        st.subheader(
            "Accident Density Heatmap (Correlates with Traffic Patterns)")
        display_accident_density_heatmap(filtered_df, lat_col, lon_col)

        # Traffic analysis insights
        st.subheader("Traffic Density Analysis Insights")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Accident Locations", len(valid_coords))

        with col2:
            # Calculate density
            if len(valid_coords) > 0:
                # Simple density calculation (accidents per area)
                lat_range = valid_coords['lat_clean'].max(
                ) - valid_coords['lat_clean'].min()
                lon_range = valid_coords['lon_clean'].max(
                ) - valid_coords['lon_clean'].min()
                area_approx = lat_range * lon_range * 111 * 111  # rough km²
                density = len(valid_coords) / max(area_approx, 0.01)
                st.metric("Accident Density", f"{density:.2f} per km²")

        with col3:
            peak_hour = filtered_df['Time of Day'].value_counts(
            ).idxmax() if 'Time of Day' in filtered_df.columns else 'N/A'
            st.metric("Peak Accident Time", str(peak_hour))

        st.info("""
        **Traffic Density Correlation:**
        - High accident density areas often correlate with high traffic density
        - Real-time traffic conditions help identify current congestion hotspots
        - Historical accident data combined with traffic patterns can predict risk zones
        """)

    except Exception as e:
        st.error(f"Error creating traffic density visualization: {str(e)}")
        # Fallback to accident density heatmap
        display_accident_density_heatmap(filtered_df, lat_col, lon_col)


def display_accident_density_heatmap(filtered_df, lat_col, lon_col):
    """Display accident density heatmap using Folium"""
    try:
        filtered_df['lat_clean'] = pd.to_numeric(
            filtered_df[lat_col], errors='coerce')
        filtered_df['lon_clean'] = pd.to_numeric(
            filtered_df[lon_col], errors='coerce')

        valid_coords = filtered_df[
            (filtered_df['lat_clean'].notna()) &
            (filtered_df['lon_clean'].notna()) &
            (filtered_df['lat_clean'].between(-90, 90)) &
            (filtered_df['lon_clean'].between(-180, 180))
        ].copy()

        if len(valid_coords) == 0:
            return

        center_lat = valid_coords['lat_clean'].mean()
        center_lon = valid_coords['lon_clean'].mean()

        # Create heatmap
        heatmap_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )

        # Prepare heatmap data
        heat_data = [[row['lat_clean'], row['lon_clean']]
                     for idx, row in valid_coords.iterrows()]

        # Add heatmap layer
        HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(heatmap_map)

        st_folium(heatmap_map, width=1400, height=500, returned_objects=[])
        st.caption(
            "Heatmap showing accident density - darker areas indicate higher accident frequency")

    except Exception as e:
        st.warning(f"Could not create heatmap: {str(e)}")


def display_accident_map(filtered_df, severity_color_map):
    """Display interactive accident hotspot map with improved error handling"""
    # Find latitude and longitude columns
    lat_col = None
    lon_col = None

    possible_lat_cols = ['Latitude', 'latitude', 'LATITUDE', 'lat', 'LAT']
    possible_lon_cols = ['Longitude', 'longitude',
                         'LONGITUDE', 'lon', 'LON', 'lng', 'LNG']

    for col in filtered_df.columns:
        if col in possible_lat_cols:
            lat_col = col
        if col in possible_lon_cols:
            lon_col = col

    if not lat_col or not lon_col:
        st.warning("Latitude and Longitude columns not found in the dataset.")
        st.info("Required column names: 'Latitude' and 'Longitude' (case-insensitive)")
        return

    st.subheader("Accident Hotspots Map")

    # Add traffic layer toggle
    show_traffic = st.checkbox("Show Real-Time Traffic Density", value=True,
                               help="Enable Google Traffic Layer to see real-time traffic conditions")

    # Filter valid coordinates
    try:
        filtered_df['lat_clean'] = pd.to_numeric(
            filtered_df[lat_col], errors='coerce')
        filtered_df['lon_clean'] = pd.to_numeric(
            filtered_df[lon_col], errors='coerce')

        # Remove invalid coordinates (outside valid ranges)
        valid_coords = filtered_df[
            (filtered_df['lat_clean'].notna()) &
            (filtered_df['lon_clean'].notna()) &
            (filtered_df['lat_clean'].between(-90, 90)) &
            (filtered_df['lon_clean'].between(-180, 180))
        ].copy()

        if len(valid_coords) == 0:
            st.error("No valid coordinates found in the dataset.")
            return

        # Sample data if too large for performance
        max_markers = 1000
        if len(valid_coords) > max_markers:
            map_data = valid_coords.sample(n=max_markers, random_state=42)
            st.info(
                f"Displaying {max_markers} random samples from {len(valid_coords)} total locations")
        else:
            map_data = valid_coords

        # Calculate center of map from data
        center_lat = map_data['lat_clean'].mean()
        center_lon = map_data['lon_clean'].mean()

        # Get column names for display (needed for both traffic map and Folium map)
        state_col = 'State Name' if 'State Name' in filtered_df.columns else (
            'State_Name' if 'State_Name' in filtered_df.columns else None)
        city_col = 'City Name' if 'City Name' in filtered_df.columns else (
            'City_Name' if 'City_Name' in filtered_df.columns else None)

        # If traffic layer is enabled and API key is available, show Google Maps with traffic
        if show_traffic and GOOGLE_MAPS_API_KEY:
            st.info(
                "**Traffic Color Legend:** 🟢 Green = Normal, 🟡 Yellow = Slow, 🔴 Red = Heavy/Stopped")

            # Prepare accident data for markers
            accident_markers_data = []
            # Limit to 50 for performance
            for idx, row in map_data.head(50).iterrows():
                try:
                    lat = float(row['lat_clean'])
                    lon = float(row['lon_clean'])
                    if (-90 <= lat <= 90) and (-180 <= lon <= 180):
                        severity = str(row.get("Accident Severity", "Unknown"))
                        color = severity_color_map.get(severity, "blue")
                        state_name = str(row.get(state_col, 'N/A')
                                         ) if state_col else 'N/A'
                        city_name = str(row.get(city_col, 'N/A')
                                        ) if city_col else 'N/A'
                        casualties = int(row.get('Number of Casualties', 0)) if pd.notna(
                            row.get('Number of Casualties')) else 0
                        fatalities = int(row.get('Number of Fatalities', 0)) if pd.notna(
                            row.get('Number of Fatalities')) else 0
                        vehicle = str(row.get('Vehicle Type Involved', 'N/A'))

                        accident_markers_data.append({
                            'lat': lat,
                            'lng': lon,
                            'severity': severity,
                            'color': color,
                            'city': city_name,
                            'state': state_name,
                            'casualties': casualties,
                            'fatalities': fatalities,
                            'vehicle': vehicle
                        })
                except:
                    continue

            # Create Google Maps with Traffic Layer using JavaScript
            traffic_map_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    #traffic-map {{
                        width: 100%;
                        height: 600px;
                        border: 2px solid #4CAF50;
                        border-radius: 8px;
                    }}
                    .traffic-legend {{
                        position: absolute;
                        top: 10px;
                        right: 10px;
                        background: white;
                        padding: 10px;
                        border-radius: 5px;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                        z-index: 1000;
                        font-size: 12px;
                    }}
                </style>
            </head>
            <body>
                <div id="traffic-map"></div>
                <div class="traffic-legend">
                    <strong>Traffic Layer Active</strong><br>
                    🟢 Normal | 🟡 Slow | 🔴 Heavy
                </div>
                <script>
                    function initMap() {{
                        var center = {{lat: {center_lat}, lng: {center_lon}}};
                        var map = new google.maps.Map(document.getElementById('traffic-map'), {{
                            zoom: 11,
                            center: center,
                            mapTypeId: 'roadmap'
                        }});
                        
                        // Add Traffic Layer
                        var trafficLayer = new google.maps.TrafficLayer();
                        trafficLayer.setMap(map);
                        
                        // Add accident markers
                        var accidents = {json.dumps(accident_markers_data)};
                        var infoWindow = new google.maps.InfoWindow();
                        
                        accidents.forEach(function(accident) {{
                            var marker = new google.maps.Marker({{
                                position: {{lat: accident.lat, lng: accident.lng}},
                                map: map,
                                icon: {{
                                    path: google.maps.SymbolPath.CIRCLE,
                                    scale: 8,
                                    fillColor: accident.color,
                                    fillOpacity: 0.8,
                                    strokeWeight: 2,
                                    strokeColor: '#FFFFFF'
                                }},
                                title: accident.severity + ' Accident'
                            }});
                            
                            marker.addListener('click', function() {{
                                infoWindow.setContent(
                                    '<div style="font-family: Arial; font-size: 12px; padding: 5px;">' +
                                    '<b>Location:</b> ' + accident.city + ', ' + accident.state + '<br>' +
                                    '<b>Severity:</b> ' + accident.severity + '<br>' +
                                    '<b>Casualties:</b> ' + accident.casualties + '<br>' +
                                    '<b>Fatalities:</b> ' + accident.fatalities + '<br>' +
                                    '<b>Vehicle:</b> ' + accident.vehicle +
                                    '</div>'
                                );
                                infoWindow.open(map, marker);
                            }});
                        }});
                    }}
                </script>
                <script async defer
                    src="https://maps.googleapis.com/maps/api/js?key={GOOGLE_MAPS_API_KEY}&libraries=visualization&callback=initMap">
                </script>
            </body>
            </html>
            """

            # Display the traffic map using components
            try:
                import streamlit.components.v1 as components
                components.html(traffic_map_html, height=620, scrolling=False)
            except:
                # Fallback to iframe if components not available
                st.markdown(f"""
                <div style="width: 100%; height: 600px; border: 2px solid #4CAF50; border-radius: 8px; margin-bottom: 20px;">
                    <iframe
                        width="100%"
                        height="100%"
                        frameborder="0"
                        style="border:0; border-radius: 6px;"
                        src="https://www.google.com/maps/embed/v1/view?key={GOOGLE_MAPS_API_KEY}&center={center_lat},{center_lon}&zoom=11&maptype=roadmap"
                        allowfullscreen>
                    </iframe>
                </div>
                """, unsafe_allow_html=True)

            # Add link to open in Google Maps with traffic layer enabled
            google_maps_url = f"https://www.google.com/maps/@{center_lat},{center_lon},11z/data=!5m1!1e1"
            st.markdown(f"""
            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; margin-bottom: 20px;">
                <strong>📊 Interactive Traffic Map:</strong> 
                <a href="{google_maps_url}" target="_blank" style="color: #2e7d32; font-weight: bold; text-decoration: none;">
                    Open in Google Maps with Traffic Layer →
                </a>
                <br>
                <small style="color: #666;">Click above to view real-time traffic conditions in full Google Maps interface.</small>
            </div>
            """, unsafe_allow_html=True)

            # Also show Folium map with accident markers below
            st.subheader("Accident Locations Overlay (Interactive Map)")

        # Create Folium map for accident markers
            accident_map = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10 if show_traffic else 6,
                tiles='OpenStreetMap'
            )

        # Add marker cluster for better performance
        marker_cluster = MarkerCluster().add_to(accident_map)

        # Add markers
        markers_added = 0
        for idx, row in map_data.iterrows():
            try:
                lat = float(row['lat_clean'])
                lon = float(row['lon_clean'])

                # Skip if coordinates are invalid
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    continue

                severity = str(row.get("Accident Severity", "Unknown"))
                color = severity_color_map.get(severity, "blue")

                state_name = str(row.get(state_col, 'N/A')
                                 ) if state_col else 'N/A'
                city_name = str(row.get(city_col, 'N/A')
                                ) if city_col else 'N/A'
                casualties = int(row.get('Number of Casualties', 0)) if pd.notna(
                    row.get('Number of Casualties')) else 0
                fatalities = int(row.get('Number of Fatalities', 0)) if pd.notna(
                    row.get('Number of Fatalities')) else 0
                vehicle = str(row.get('Vehicle Type Involved', 'N/A'))

                # Create popup text
                popup_text = f"""
                <div style="font-family: Arial; font-size: 12px;">
                    <b>Location:</b> {city_name}, {state_name}<br>
                    <b>Severity:</b> {severity}<br>
                    <b>Casualties:</b> {casualties}<br>
                    <b>Fatalities:</b> {fatalities}<br>
                    <b>Vehicle:</b> {vehicle}
                </div>
                """

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    popup=folium.Popup(popup_text, max_width=300),
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(marker_cluster)
                markers_added += 1
            except Exception as e:
                continue

        # Add legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; right: 50px; width: 160px; height: 130px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
        <p style="margin-bottom: 8px; font-weight: bold; border-bottom: 1px solid #ccc; padding-bottom: 5px;">Severity Legend</p>
        <p style="margin: 5px 0;"><span style="color:red; font-size: 18px;">●</span> Fatal</p>
        <p style="margin: 5px 0;"><span style="color:orange; font-size: 18px;">●</span> Serious</p>
        <p style="margin: 5px 0;"><span style="color:green; font-size: 18px;">●</span> Minor</p>
        </div>
        '''
        accident_map.get_root().html.add_child(folium.Element(legend_html))

        # Display map
        st_folium(accident_map, width=1400, height=600, returned_objects=[])
        st.success(
            f"Successfully displayed {markers_added} accident locations on the map")

        if show_traffic:
            if GOOGLE_MAPS_API_KEY:
                st.info(
                    "✅ Traffic layer enabled. View the Google Maps embed above to see real-time traffic conditions.")
            else:
                st.warning(
                    "⚠️ Google Maps API key not configured. Traffic layer requires API key.")

    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        st.info("Please check that your dataset contains valid Latitude and Longitude columns with numeric values.")

# =============================================================================
# MACHINE LEARNING FUNCTIONS
# =============================================================================


@st.cache_data(show_spinner=False)
def train_severity_model_cached(_df_hash, filtered_df):
    """Cached training function for accident severity prediction model"""
    try:
        required_cols = ['Accident Severity', 'Vehicle Type Involved', 'Time of Day',
                         'Day of Week', 'Alcohol Involvement']

        # Validate columns
        missing_cols = [
            col for col in required_cols if col not in filtered_df.columns]
        if missing_cols:
            return None, None, None, None, f"Missing columns: {', '.join(missing_cols)}"

        if len(filtered_df) == 0:
            return None, None, None, None, "Empty dataset"

        # Check minimum data requirement
        if len(filtered_df) < 10:
            return None, None, None, None, f"Insufficient data: {len(filtered_df)} rows (minimum 10 required)"

        ml_df = filtered_df.copy()

        # Remove rows with missing critical data
        ml_df = ml_df.dropna(subset=required_cols)

        if len(ml_df) < 10:
            return None, None, None, None, f"After removing missing values: {len(ml_df)} rows (minimum 10 required)"

        # Check if we have multiple classes
        unique_severities = ml_df['Accident Severity'].nunique()
        if unique_severities < 2:
            return None, None, None, None, f"Need at least 2 severity classes, found {unique_severities}"

        # Encode categorical variables
        label_encoders = {}
        for col in ['Vehicle Type Involved', 'Time of Day', 'Alcohol Involvement', 'Day of Week']:
            le = LabelEncoder()
            ml_df[col] = le.fit_transform(ml_df[col].astype(str))
            label_encoders[col] = le

        X = ml_df[['Vehicle Type Involved', 'Time of Day',
                   'Alcohol Involvement', 'Day of Week']].fillna(0)
        y = ml_df['Accident Severity']

        le_severity = LabelEncoder()
        y_encoded = le_severity.fit_transform(y)

        # Ensure we have enough data for train/test split
        if len(X) < 20:
            # Use all data for training if too small
            X_train, X_test, y_train, y_test = X, X, y_encoded, y_encoded
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42)

        # Use optimized parameters for faster training
        clf = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            max_depth=10,
            n_jobs=-1,
            min_samples_split=5,
            min_samples_leaf=2
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return clf, label_encoders, le_severity, accuracy, None
    except Exception as e:
        return None, None, None, None, f"Training error: {str(e)}"


def train_severity_model(filtered_df):
    """Train and evaluate accident severity prediction model"""
    # Create a hash of the dataframe for caching
    try:
        df_hash = hash(str(filtered_df.shape) +
                       str(filtered_df.columns.tolist()) + str(len(filtered_df)))
        return train_severity_model_cached(df_hash, filtered_df)
    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"


@st.cache_data(show_spinner=False)
def train_casualty_model_cached(_df_hash, filtered_df):
    """Cached training function for casualty prediction model"""
    try:
        required_cols = ['Number of Casualties', 'Vehicle Type Involved', 'Time of Day',
                         'Day of Week', 'Alcohol Involvement', 'Accident Severity']

        # Validate columns
        missing_cols = [
            col for col in required_cols if col not in filtered_df.columns]
        if missing_cols:
            return None, None, None, None, f"Missing columns: {', '.join(missing_cols)}"

        if len(filtered_df) == 0:
            return None, None, None, None, "Empty dataset"

        # Check minimum data requirement
        if len(filtered_df) < 10:
            return None, None, None, None, f"Insufficient data: {len(filtered_df)} rows (minimum 10 required)"

        ml_df = filtered_df.copy()

        # Remove rows with missing critical data
        ml_df = ml_df.dropna(subset=required_cols)

        if len(ml_df) < 10:
            return None, None, None, None, f"After removing missing values: {len(ml_df)} rows (minimum 10 required)"

        # Encode categorical variables
        label_encoders = {}
        for col in ['Vehicle Type Involved', 'Time of Day', 'Alcohol Involvement',
                    'Day of Week', 'Accident Severity']:
            le = LabelEncoder()
            ml_df[col] = le.fit_transform(ml_df[col].astype(str))
            label_encoders[col] = le

        X = ml_df[['Vehicle Type Involved', 'Time of Day', 'Alcohol Involvement',
                   'Day of Week', 'Accident Severity']].fillna(0)
        y = ml_df['Number of Casualties'].fillna(0)

        # Validate target variable
        if y.nunique() < 2:
            return None, None, None, None, "Target variable has insufficient variation"

        # Ensure we have enough data for train/test split
        if len(X) < 20:
            # Use all data for training if too small
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

        # Use optimized parameters for faster training
        reg = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            max_depth=10,
            n_jobs=-1,
            min_samples_split=5,
            min_samples_leaf=2
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return reg, label_encoders, mae, rmse, None
    except Exception as e:
        return None, None, None, None, f"Training error: {str(e)}"


def train_casualty_model(filtered_df):
    """Train and evaluate casualty prediction model"""
    # Create a hash of the dataframe for caching
    try:
        df_hash = hash(str(filtered_df.shape) +
                       str(filtered_df.columns.tolist()) + str(len(filtered_df)))
        return train_casualty_model_cached(df_hash, filtered_df)
    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"

# =============================================================================
# FORECASTING FUNCTIONS
# =============================================================================


def display_prophet_forecast(filtered_df):
    """Display Prophet time series forecast"""
    if not PROPHET_AVAILABLE:
        st.warning("Prophet not installed. Install with: pip install prophet")
        return

    if not (safe_column_check(filtered_df, 'Year') and safe_column_check(filtered_df, 'Month')):
        st.warning("Year and Month columns required for forecasting")
        return

    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    tmp = filtered_df.copy()
    tmp['Year_num'] = pd.to_numeric(
        tmp['Year'], errors='coerce').astype('Int64')

    if tmp['Month'].dtype == object:
        tmp['Month_num'] = tmp['Month'].map(month_map)
        tmp['Month_num'] = tmp['Month_num'].fillna(
            pd.to_numeric(tmp['Month'], errors='coerce'))
    else:
        tmp['Month_num'] = pd.to_numeric(tmp['Month'], errors='coerce')

    tmp = tmp.dropna(subset=['Year_num', 'Month_num'])

    if tmp.empty:
        st.info("Not enough clean date data for forecasting.")
        return

    tmp['Date'] = pd.to_datetime(
        tmp['Year_num'].astype(str) + '-' +
        tmp['Month_num'].astype(int).astype(str) + '-01',
        errors='coerce'
    )

    time_data = tmp.groupby(['Date']).size().reset_index(name='count')
    time_data = time_data.dropna(subset=['Date']).sort_values('Date')
    time_data = time_data.rename(columns={'Date': 'ds', 'count': 'y'})

    if len(time_data) < 12:
        st.warning(
            "Insufficient time series data (need >= 12 months) for forecasting")
        return

    with st.spinner("Training Prophet model..."):
        model = Prophet(yearly_seasonality=True,
                        weekly_seasonality=False, daily_seasonality=False)
        model.fit(time_data)
        future = model.make_future_dataframe(periods=12, freq='MS')
        forecast = model.predict(future)

    st.subheader("Accident Trend Forecast (Next 12 Months)")

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=time_data['ds'], y=time_data['y'], mode='lines+markers', name='Historical'
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(dash='dash')
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', showlegend=False, line=dict(color='rgba(0,0,0,0)')
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty',
        fillcolor='rgba(231,76,60,0.2)', showlegend=False, line=dict(color='rgba(0,0,0,0)')
    ))
    fig_forecast.update_layout(
        height=400, template="plotly_white", hovermode='x unified',
        title="Accident Forecast with Confidence Intervals"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

# =============================================================================
# POLICY RECOMMENDATIONS
# =============================================================================


def generate_policy_recommendations(filtered_df, metrics):
    """Generate data-driven policy recommendations"""
    recommendations = []

    if metrics['fatality_rate'] > 5:
        recommendations.append({
            'priority': 'High',
            'area': 'Road Safety Infrastructure',
            'recommendation': 'Strengthen trauma care systems and enforce helmet/seatbelt laws.',
            'metric': f'Fatality rate: {metrics["fatality_rate"]:.2f}%'
        })

    if safe_column_check(filtered_df, 'Alcohol Involvement'):
        alcohol_share = (filtered_df['Alcohol Involvement'].value_counts(
            normalize=True).get('Yes', 0)) * 100
        if alcohol_share > 10:
            recommendations.append({
                'priority': 'High',
                'area': 'DUI Prevention',
                'recommendation': 'Implement stricter DUI checks and awareness campaigns.',
                'metric': f'Alcohol involvement: {alcohol_share:.1f}%'
            })

    if safe_column_check(filtered_df, 'Vehicle Type Involved') and safe_column_check(filtered_df, 'Number of Casualties'):
        vc = filtered_df.groupby('Vehicle Type Involved')[
            'Number of Casualties'].sum().reset_index()
        vc = vc.sort_values('Number of Casualties', ascending=False)
        two_wheelers = vc[vc['Vehicle Type Involved'].str.contains(
            'Two', case=False, na=False)]
        if not two_wheelers.empty and two_wheelers['Number of Casualties'].iloc[0] > 100:
            recommendations.append({
                'priority': 'Medium',
                'area': 'Two-Wheeler Safety',
                'recommendation': 'Enforce helmet laws and conduct safety awareness drives.',
                'metric': f'High two-wheeler casualties detected'
            })

    if safe_column_check(filtered_df, 'Time of Day'):
        peak_time = filtered_df['Time of Day'].value_counts().idxmax()
        recommendations.append({
            'priority': 'Medium',
            'area': 'Temporal Safety Measures',
            'recommendation': f'Improve road lighting and increase patrols during {peak_time}.',
            'metric': f'Peak accident time: {peak_time}'
        })

    return recommendations

# =============================================================================
# MAIN APPLICATION
# =============================================================================


def main():
    """Main application function"""
    # Header
    st.markdown(
        "<h1 style='text-align: center;'>Road Accident Hotspot Detection & Analysis Dashboard</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; font-size: 18px; color: #7f8c8d;'>"
        "Comprehensive analysis of road accidents and infrastructure across India</p>",
        unsafe_allow_html=True
    )

    # Load datasets
    with st.spinner("Loading datasets..."):
        df_accidents = load_accident_data()
        df_road_transport = load_road_transport_data()
        df_motor_transport = load_motor_transport_data()

    if df_accidents is None:
        st.error("Cannot proceed without accident data. Please check file paths.")
        st.stop()

    # Setup sidebar filters
    selected_state, selected_year, selected_severity, selected_vehicle_types, state_col = setup_sidebar_filters(
        df_accidents)

    # Apply filters
    filtered_df = apply_filters(df_accidents, selected_state, selected_year,
                                selected_severity, selected_vehicle_types, state_col)

    # Display KPI Metrics
    st.markdown("---")
    st.header("Key Performance Indicators")
    metrics = display_kpi_metrics(filtered_df)

    # Road Network Analysis
    if df_road_transport is not None:
        st.markdown("---")
        st.header("Road Network Infrastructure Analysis")

        if "State/UT" in df_road_transport.columns or "Name of the States" in df_road_transport.columns:
            state_col_road = "State/UT" if "State/UT" in df_road_transport.columns else "Name of the States"

            st.subheader("Total Road Length by Type")
            df_melt = df_road_transport.melt(
                id_vars=[state_col_road], var_name="Road Type", value_name="Length (km)"
            )
            df_melt = df_melt.dropna(subset=["Length (km)"])

            fig_road = px.bar(df_melt, x=state_col_road, y="Length (km)", color="Road Type",
                              title="Distribution of Road Types by State",
                              labels={"Length (km)": "Length (in km)"})
            fig_road.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_road, use_container_width=True)

            total_by_type = df_melt.groupby(
                "Road Type")["Length (km)"].sum().reset_index()
            fig_pie_roads = px.pie(total_by_type, names="Road Type", values="Length (km)",
                                   title="Proportion of Road Types in India", hole=0.4)
            st.plotly_chart(fig_pie_roads, use_container_width=True)

    # Visualizations
    st.markdown("---")
    st.header("Data Visualizations")

    display_city_analysis(filtered_df)
    display_vehicle_analysis(filtered_df)

    severity_color_map = display_severity_distribution(filtered_df)
    display_time_heatmap(filtered_df)
    display_alcohol_analysis(filtered_df)

    # Accident Hotspot Map
    st.markdown("---")
    st.header("Accident Hotspot Map")
    display_accident_map(filtered_df, severity_color_map)

    # Traffic Density Visualization
    st.markdown("---")
    st.header("Traffic Density Visualization")
    display_traffic_density_analysis(filtered_df)

    # Street View Analysis Section
    if GOOGLE_STREETVIEW_API_KEY and safe_column_check(filtered_df, 'Latitude') and safe_column_check(filtered_df, 'Longitude'):
        st.markdown("---")
        st.header("Street View Traffic Analysis")

        # Select a hotspot for analysis
        st.subheader("Select Hotspot for Street View Analysis")

        # Get top accident locations
        if state_col:
            hotspot_data = filtered_df.groupby(
                [state_col, 'Latitude', 'Longitude']).size().reset_index(name='Accident Count')
            hotspot_data = hotspot_data.sort_values(
                'Accident Count', ascending=False).head(10)

            if len(hotspot_data) > 0:
                selected_hotspot = st.selectbox(
                    "Select a hotspot location",
                    options=range(len(hotspot_data)),
                    format_func=lambda x: f"{hotspot_data.iloc[x][state_col]} - {hotspot_data.iloc[x]['Accident Count']} accidents"
                )

                selected_row = hotspot_data.iloc[selected_hotspot]
                lat = selected_row['Latitude']
                lon = selected_row['Longitude']

                # Get accident details for this location
                location_accidents = filtered_df[
                    (filtered_df['Latitude'] == lat) & (
                        filtered_df['Longitude'] == lon)
                ].iloc[0]

                accident_info = {
                    'City': location_accidents.get('City Name', location_accidents.get('City_Name', 'N/A')),
                    'State': location_accidents.get(state_col, 'N/A'),
                    'Severity': location_accidents.get('Accident Severity', 'N/A'),
                    'Vehicle Type': location_accidents.get('Vehicle Type Involved', 'N/A'),
                    'Time': location_accidents.get('Time of Day', 'N/A'),
                    'Casualties': location_accidents.get('Number of Casualties', 0)
                }

                display_street_view_analysis(lat, lon, accident_info)

    # Machine Learning Predictions
    st.markdown("---")
    st.header("Machine Learning Predictions")

    tab1, tab2 = st.tabs(["Accident Severity Prediction",
                         "Casualty Count Prediction"])

    with tab1:
        st.subheader("Predict Accident Severity")

        with st.spinner("Training severity prediction model..."):
            clf, label_encoders, le_severity, accuracy, error_msg = train_severity_model(
                filtered_df)

        if clf is not None and label_encoders is not None and le_severity is not None and accuracy is not None:
            st.success(
                f"Model trained successfully! Accuracy: {accuracy*100:.2f}%")

            st.markdown("### Make a Prediction")
            col1, col2 = st.columns(2)
            with col1:
                try:
                    vehicle_input = st.selectbox(
                        "Vehicle Type", label_encoders['Vehicle Type Involved'].classes_)
                    time_input = st.selectbox(
                        "Time of Day", label_encoders['Time of Day'].classes_)
                except Exception as e:
                    st.error(f"Error loading options: {str(e)}")
                    vehicle_input = None
                    time_input = None
            with col2:
                try:
                    alcohol_input = st.selectbox(
                        "Alcohol Involvement", label_encoders['Alcohol Involvement'].classes_)
                    day_input = st.selectbox(
                        "Day of Week", label_encoders['Day of Week'].classes_)
                except Exception as e:
                    st.error(f"Error loading options: {str(e)}")
                    alcohol_input = None
                    day_input = None

            if vehicle_input and time_input and alcohol_input and day_input:
                if st.button("Predict Severity", type="primary"):
                    try:
                        input_data = [[
                            label_encoders['Vehicle Type Involved'].transform([vehicle_input])[
                                0],
                            label_encoders['Time of Day'].transform([time_input])[
                                0],
                            label_encoders['Alcohol Involvement'].transform([alcohol_input])[
                                0],
                            label_encoders['Day of Week'].transform([day_input])[
                                0],
                        ]]
                        prediction = clf.predict(input_data)
                        severity_pred = le_severity.inverse_transform(prediction)[
                            0]

                        # Get prediction probabilities
                        proba = clf.predict_proba(input_data)[0]
                        proba_dict = dict(zip(le_severity.classes_, proba))

                        st.success(
                            f"Predicted Accident Severity: **{severity_pred}**")

                        # Display probabilities
                        st.markdown("**Prediction Probabilities:**")
                        prob_data = {
                            'Severity': list(proba_dict.keys()),
                            'Probability': list(proba_dict.values())
                        }
                        prob_df = pd.DataFrame(prob_data)
                        prob_df = prob_df.sort_values(
                            'Probability', ascending=False)
                        prob_df['Probability'] = prob_df['Probability'].apply(
                            lambda x: f"{x*100:.1f}%")
                        st.dataframe(
                            prob_df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
        else:
            st.warning("Cannot train severity prediction model.")
            if error_msg:
                st.error(f"Error: {error_msg}")
            st.info(
                "**Required columns:** Accident Severity, Vehicle Type Involved, Time of Day, Day of Week, Alcohol Involvement")
            st.info(f"**Current data size:** {len(filtered_df)} rows")
            if len(filtered_df) < 10:
                st.info("**Note:** Need at least 10 rows of data for training")

    with tab2:
        st.subheader("Predict Number of Casualties")

        with st.spinner("Training casualty prediction model..."):
            reg, label_encoders, mae, rmse, error_msg = train_casualty_model(
                filtered_df)

        if reg is not None and label_encoders is not None and mae is not None and rmse is not None:
            st.success(f"Model trained successfully!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Absolute Error", f"{mae:.2f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}")

            st.markdown("### Make a Prediction")
            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    vehicle_input = st.selectbox(
                        "Vehicle Type", label_encoders['Vehicle Type Involved'].classes_, key="v1")
                    time_input = st.selectbox(
                        "Time of Day", label_encoders['Time of Day'].classes_, key="t1")
                except Exception as e:
                    st.error(f"Error loading options: {str(e)}")
                    vehicle_input = None
                    time_input = None
            with col2:
                try:
                    day_input = st.selectbox(
                        "Day of Week", label_encoders['Day of Week'].classes_, key="d1")
                    alcohol_input = st.selectbox(
                        "Alcohol Involvement", label_encoders['Alcohol Involvement'].classes_, key="a1")
                except Exception as e:
                    st.error(f"Error loading options: {str(e)}")
                    day_input = None
                    alcohol_input = None
            with col3:
                try:
                    severity_input = st.selectbox(
                        "Accident Severity", label_encoders['Accident Severity'].classes_, key="s1")
                except Exception as e:
                    st.error(f"Error loading options: {str(e)}")
                    severity_input = None

            if vehicle_input and time_input and day_input and alcohol_input and severity_input:
                if st.button("Predict Casualties", type="primary"):
                    try:
                        input_data = [[
                            label_encoders['Vehicle Type Involved'].transform([vehicle_input])[
                                0],
                            label_encoders['Time of Day'].transform([time_input])[
                                0],
                            label_encoders['Alcohol Involvement'].transform([alcohol_input])[
                                0],
                            label_encoders['Day of Week'].transform([day_input])[
                                0],
                            label_encoders['Accident Severity'].transform([severity_input])[
                                0],
                        ]]
                        prediction = reg.predict(input_data)[0]
                        # Ensure non-negative
                        predicted_value = max(0, int(round(prediction)))
                        st.success(
                            f"Predicted Number of Casualties: **{predicted_value}**")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
        else:
            st.warning("Cannot train casualty prediction model.")
            if error_msg:
                st.error(f"Error: {error_msg}")
            st.info("**Required columns:** Number of Casualties, Vehicle Type Involved, Time of Day, Day of Week, Alcohol Involvement, Accident Severity")
            st.info(f"**Current data size:** {len(filtered_df)} rows")
            if len(filtered_df) < 10:
                st.info("**Note:** Need at least 10 rows of data for training")

    # Forecasting
    st.markdown("---")
    st.header("Time Series Forecasting")
    display_prophet_forecast(filtered_df)

    # Policy Recommendations
    st.markdown("---")
    st.header("Data-Driven Policy Recommendations")
    recommendations = generate_policy_recommendations(filtered_df, metrics)

    if recommendations:
        priority_colors = {'High': '#e74c3c',
                           'Medium': '#f39c12', 'Low': '#2ecc71'}
        for rec in recommendations:
            color = priority_colors.get(rec['priority'], '#95a5a6')
            st.markdown(f"""
            <div class="card" style="border-left:5px solid {color};">
                <h3 style="color:{color}; margin-top:0;">{rec['priority']} Priority: {rec['area']}</h3>
                <p><b>Recommendation:</b> {rec['recommendation']}</p>
                <p style="color:#7f8c8d;"><b>Data Insight:</b> {rec['metric']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No critical risks detected in current filter.")

    # Raw Data
    st.markdown("---")
    st.header("Raw Data")
    with st.expander("Accident Hotspot Dataset"):
        st.dataframe(filtered_df.reset_index(drop=True))
    if df_road_transport is not None:
        with st.expander("Road Transport Statistics"):
            st.dataframe(df_road_transport.reset_index(drop=True))
    if df_motor_transport is not None:
        with st.expander("Motor Transport Statistics"):
            st.dataframe(df_motor_transport.reset_index(drop=True))

# =============================================================================
# RUN APPLICATION
# =============================================================================


if __name__ == "__main__":
    main()

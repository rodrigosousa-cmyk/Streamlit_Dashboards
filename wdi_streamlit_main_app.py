# RUN THIS CODE ON TERMINAL
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import duckdb
from pathlib import Path

# Page configuration
st.set_page_config(layout="wide")
st.title("CO2 Emissions Animation")

# Constants
CSV_PATH = "CO2_Emissions_Dataset.csv"
PARQUET_PATH = "CO2_Emissions_Dataset.parquet"

# Cache database connection
@st.cache_resource
def get_connection():
    return duckdb.connect()

# Cache data loading
@st.cache_data(ttl=3600)
def load_data():
    conn = get_connection()
    
    # Convert to parquet if needed
    if not Path(PARQUET_PATH).exists():
        conn.execute(f"""
            COPY (SELECT * FROM read_csv('{CSV_PATH}')) 
            TO '{PARQUET_PATH}' (FORMAT PARQUET)
        """)
    
    return conn.execute(f"SELECT * FROM read_parquet('{PARQUET_PATH}')").df()

# Load data
with st.spinner('Loading CO2 data...'):
    data = load_data()

# Prepare data for plotting
@st.cache_data
def prepare_data(df):
    df_clean = df[
        (df['emissions_total'] >= 0) & 
        (df['cumulative_emissions_total'] >= 0) &
        (df['Code'].notna())
    ].copy()
    
    # Optimize data types
    for col in ['Code', 'Entity']:
        df_clean[col] = df_clean[col].astype('category')
    df_clean['Year'] = df_clean['Year'].astype('int16')
    
    return df_clean

plot_data = prepare_data(data)

# Create choropleth map
def create_choropleth(df,indicator):
    # Use percentiles to exclude outliers
    valid_data = df[indicator].dropna()
    
    if len(valid_data) > 0:
        # Get 5th and 95th percentiles to exclude extremes
        p5 = np.percentile(valid_data, 10)
        p95 = np.percentile(valid_data, 90)
        
        # Apply log scaling with buffer
        min_val = max(p5 * 0.1, valid_data.min())  # Don't go below actual min
        max_val = min(p95 * 10, valid_data.max())  # Don't go above actual max
    else:
        min_val, max_val = 1e3, 1e9
    fig = px.choropleth(
        df,
        locations="Code",
        color=indicator,
        hover_name="Entity",
        animation_frame="Year",
        #title=str('CO2 '+indicator),
        color_continuous_scale='Plasma',
        #range_color=[1e3, 1e10] # Changed from range_x to range_color
        range_color=[min_val, max_val]
    )
    
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
    return fig

# Create scatter plot
def create_scatter(df):
    scatter_data = df[
        (df['emissions_total'] > 1e3) & 
        (df['cumulative_emissions_total'] > 1e3)
    ]
    
    fig = px.scatter(
        scatter_data,
        x="emissions_total",
        y="cumulative_emissions_total",
        animation_frame="Year",
        animation_group="Code",
        size="cumulative_emissions_total",
        color="Code",
        hover_name="Entity",
        log_x=True,
        log_y=True,
        size_max=60,
        range_x=[1e6, 13e9],
        range_y=[1e6, 11e11],
        title='Emissions vs Cumulative Emissions Over Time'
    )
    return fig

# Display plots in tabs
tab1, tab2 ,tab3 = st.tabs(["🌍 Annual CO2 Map", "🌍 Cumulative CO2 Map", "📊 Scatter Plot"])

with tab1:
    st.plotly_chart(create_choropleth(plot_data,"emissions_total"), use_container_width=True)
     # Convert to CSV for download
    csv_choropleth = plot_data.to_csv(index=False)
    
    st.download_button(
        label="Download .CSV",
        data=csv_choropleth,
        file_name=f"choropleth_data.csv",
        mime="text/csv",
        help="Download the data used in the choropleth map"
    )
with tab2:
    st.plotly_chart(create_choropleth(plot_data,"cumulative_emissions_total"), use_container_width=True)
 
with tab3:
    st.plotly_chart(create_scatter(plot_data), use_container_width=True)

    



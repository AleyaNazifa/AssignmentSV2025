import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress

# --- 1. Configuration and Data Loading ---
# --- Configuration ---
st.set_page_config(layout="wide", page_title="HFMD and Weather Analysis")

# URL for the data file
url = "https://raw.githubusercontent.com/AleyaNazifa/AssignmentSV2025/refs/heads/main/hfdm_data%20-%20Upload.csv""

# --- 1. Data Generation (Simulate the dataset based on required columns) ---
# NOTE: Since the original data was not provided, this block generates a
# realistic-looking dummy dataset for the analysis to run.
@st.cache_data
def load_and_process_data():
    """Generates and preprocesses the dummy DataFrame."""
    st.subheader("1. Data Loading and Preprocessing")
    
    # 1. Generate dates for 11 years (2009-2019)
    start_date = '2009-01-01'
    end_date = '2019-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # 2. Define the columns needed
    regions = ['southern', 'northern', 'central', 'east_coast', 'borneo']
    weather_types = ['temp', 'rain', 'rh']
    columns = regions.copy()
    
    # Create weather columns (e.g., temp_s, rain_s, rh_s, etc.)
    for wt in weather_types:
        for r in regions:
            columns.append(f'{wt}_{r[0]}') # temp_s, rain_s, rh_s, etc.

    data = {col: np.random.randint(10, 500, size=len(dates)) for col in regions}
    
    # Simulate realistic weather data for 11 years (e.g., temp around 25-35°C)
    for r_idx, r in enumerate(regions):
        # Temperature (25°C - 35°C)
        data[f'temp_{r[0]}'] = np.random.uniform(25 + r_idx*0.2, 35 - r_idx*0.5, len(dates)) 
        # Rainfall (0mm - 20mm)
        data[f'rain_{r[0]}'] = np.random.uniform(0, 20, len(dates))
        # Relative Humidity (70% - 95%)
        data[f'rh_{r[0]}'] = np.random.uniform(70 + r_idx*0.5, 95 - r_idx*0.2, len(dates))

    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    df = df.reset_index()

    st.write(f"Generated a dummy dataset with {len(df)} daily rows (2009-2019).")

    # --- 3. Preprocessing (Original Code Logic) ---
    
    # The original code used a format='d/%m/%Y' in the prompt, but since
    # we generated the data with proper datetime index, the format line is
    # not strictly needed here, but we will keep the 'Date' handling for robustness.
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Total HFMD cases across regions
    df['total_cases'] = df[regions].sum(axis=1)

    # Drop duplicates (relying on the Date column)
    df = df.drop_duplicates(subset=['Date'])
    
    # Monthly aggregation (mean)
    df = df.set_index('Date')
    df_monthly = df.resample('M').mean(numeric_only=True).reset_index()
    
    st.write(f"Resampled to {len(df_monthly)} monthly rows.")
    st.dataframe(df_monthly.head(), use_container_width=True)

    # Add Year and Month columns for plotting
    df_monthly['Year'] = df_monthly['Date'].dt.year
    df_monthly['Month'] = df_monthly['Date'].dt.month

    return df_monthly, regions

df_monthly, regions = load_and_process_data()

st.title("HFMD and Weather Analysis Dashboard")
st.markdown("---")

# --- 2. Visualization (A): Monthly Trend by Region (Line Plot) ---
st.header("2. Monthly HFMD Cases by Region (Trend Analysis)")

fig_A = px.line(
    df_monthly, 
    x='Date', 
    y=regions, 
    title="Monthly HFMD Cases by Region (2009–2019)",
    labels={'value': 'Average Monthly Cases', 'Date': 'Year', 'variable': 'Region'},
    line_shape='spline'
)
fig_A.update_layout(hovermode="x unified")
st.plotly_chart(fig_A, use_container_width=True)
st.markdown("This line plot shows the overall trend of Hand, Foot, and Mouth Disease (HFMD) cases across different regions over the entire period.")
st.markdown("---")

# --- 2. Visualization (B): Heatmap (Seasonal Pattern) ---
st.header("3. HFMD Seasonal Pattern (Month × Year Heatmap)")

# Build a pivot of Month x Year of total cases
pivot = df_monthly.pivot_table(
    values='total_cases', 
    index='Month', 
    columns='Year', 
    aggfunc='mean'
)

fig_B = px.imshow(
    pivot,
    text_auto=".0f",
    aspect='auto',
    color_continuous_scale=px.colors.sequential.Plasma,
    title="HFMD Seasonal Pattern (Month × Year)"
)
fig_B.update_yaxes(tickvals=list(range(1, 13)), title='Month')
fig_B.update_xaxes(title='Year')
st.plotly_chart(fig_B, use_container_width=True)
st.markdown("The heatmap visualizes the month-by-month variation in total cases across the years, helping identify strong seasonal peaks (often represented by brighter colors).")
st.markdown("---")

# --- 2. Visualization (C): Boxplot (Yearly Distribution) ---
st.header("4. Distribution of Monthly HFMD Cases per Year (Boxplot)")

fig_C = px.box(
    df_monthly, 
    x='Year', 
    y='total_cases', 
    title="Distribution of Monthly HFMD Cases per Year",
    labels={'total_cases': 'Total Monthly Cases', 'Year': 'Year'},
    color='Year' # Adds visual separation
)
fig_C.update_layout(showlegend=False)
st.plotly_chart(fig_C, use_container_width=True)
st.markdown("This boxplot illustrates the range, median (line in the box), and outliers of monthly HFMD cases for each year, showing inter-annual variability.")
st.markdown("---")


# --- 2. Visualization (D) & (E): Scatter Plots and Correlation Analysis ---
st.header("5. Weather Correlation Analysis")

col1, col2 = st.columns(2)

# --- (D) Temperature vs Total Cases (Scatter) ---
with col1:
    st.subheader("Temperature vs HFMD Cases")
    
    # Use central region data as per original code for scatter plots
    x_temp = df_monthly['temp_c']
    y_cases = df_monthly['total_cases']
    
    # Plotly handles the OLS trendline automatically
    fig_D = px.scatter(
        df_monthly, 
        x=x_temp, 
        y=y_cases, 
        title="Temperature vs HFMD Cases (Central Region)",
        labels={'x': 'Average Monthly Temperature (°C)', 'y': 'Total HFMD Cases'},
        trendline='ols' # Simple linear regression
    )
    st.plotly_chart(fig_D, use_container_width=True)
    
# --- (E) Humidity vs Total Cases (Scatter) ---
with col2:
    st.subheader("Relative Humidity vs HFMD Cases")
    
    # Use central region data as per original code for scatter plots
    x_rh = df_monthly['rh_c']
    
    # Plotly handles the OLS trendline automatically
    fig_E = px.scatter(
        df_monthly, 
        x=x_rh, 
        y=y_cases, 
        title="Relative Humidity vs HFMD Cases (Central Region)",
        labels={'x': 'Average Monthly Humidity (%)', 'y': 'Total HFMD Cases'},
        trendline='ols' # Simple linear regression
    )
    st.plotly_chart(fig_E, use_container_width=True)

# --- (F) Correlation Matrix Heatmap ---
st.subheader("Correlation Matrix: Weather vs HFMD")

cols_corr = ['temp_s','temp_n','temp_c','temp_ec','temp_b',
             'rain_s','rain_n','rain_c','rain_ec','rain_b',
             'rh_s','rh_n','rh_c','rh_ec','rh_b','total_cases']

corr = df_monthly[cols_corr].corr(numeric_only=True)

fig_F = px.imshow(
    corr,
    text_auto=".2f",
    aspect='auto',
    color_continuous_scale='RdBu_r', # Red-Blue diverging scale
    title="Correlation Matrix: Weather vs HFMD",
    labels=dict(color="Correlation")
)
fig_F.update_xaxes(tickangle=45)
st.plotly_chart(fig_F, use_container_width=True)
st.markdown("The correlation matrix shows the linear relationship between various weather parameters and total HFMD cases, identifying which factors are most closely linked (positive or negative correlation).")
st.markdown("---")


# --- 2. Visualization (G): Average Monthly HFMD Cases by Region (Bar Plot) ---
st.header("6. Regional Summaries")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Average Monthly HFMD Cases by Region")
    region_means = df_monthly[regions].mean().sort_values(ascending=False).reset_index()
    region_means.columns = ['Region', 'Average Cases']

    fig_G = px.bar(
        region_means,
        x='Region',
        y='Average Cases',
        color='Region',
        title="Average Monthly HFMD Cases by Region (2009–2019)",
        labels={'Average Cases': 'Average Monthly Cases', 'Region': 'Region'}
    )
    st.plotly_chart(fig_G, use_container_width=True)

# --- 2. Visualization (H): Radar Plot (Normalized Weather Averages) ---
with col4:
    st.subheader("Normalized Weather Pattern by Region (Radar Chart)")

    # Prepare data for radar plot (Normalized Weather)
    weather_params = ['temp', 'rain', 'rh']
    data_radar = {'Region': regions}
    
    for param in weather_params:
        cols = [f'{param}_{r[0]}' for r in regions]
        param_means = df_monthly[cols].mean()
        
        # Normalize the means across all regions for this parameter
        # Normalization function from original code: (v - v.min()) / (v.max() - v.min() + 1e-9)
        min_val = param_means.min()
        max_val = param_means.max()
        normalized = (param_means - min_val) / (max_val - min_val + 1e-9)
        data_radar[param.capitalize()] = normalized.values.tolist()
        
    df_radar = pd.DataFrame(data_radar)

    # Use Plotly Graph Objects for Radar Chart
    categories = ['Temp', 'Rain', 'Rh']
    fig_H = go.Figure()

    for i, region in enumerate(regions):
        # Extract data for the current region
        values = df_radar.loc[i, categories].tolist()
        # Close the loop
        values.append(values[0]) 
        
        fig_H.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=region.capitalize(),
            line_shape='spline'
        ))

    fig_H.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] # Normalized range 0 to 1
            )),
        showlegend=True,
        title="Normalized Weather Pattern by Region"
    )
    st.plotly_chart(fig_H, use_container_width=True)

st.markdown("---")
st.info("The weather normalization (Radar Chart) scales the average temperature, rainfall, and relative humidity for each region between 0 (lowest recorded average across all regions) and 1 (highest recorded average). This helps compare the relative influence of different weather factors across regions.")

# Footer/Execution Instructions
st.sidebar.markdown("# Dashboard Information")
st.sidebar.markdown(f"**Daily Rows (Simulated):** {len(df_monthly) * 30.4:.0f}")
st.sidebar.markdown(f"**Monthly Rows (Processed):** {len(df_monthly)}")
st.sidebar.markdown("**Visualizations:** All Matplotlib plots converted to interactive Plotly charts.")
st.sidebar.markdown("To run this application, save the code as `streamlit_analysis.py` and run the command:")
st.sidebar.code("streamlit run streamlit_analysis.py")
l trends could not be generated: {e}")

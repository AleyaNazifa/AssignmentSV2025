import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# from scipy.stats import linregress  # not used; remove to avoid warnings

# --- 1. Configuration ---
st.set_page_config(layout="wide", page_title="HFMD and Weather Analysis")
st.title("HFMD and Weather Analysis Dashboard")
st.markdown("---")

# URL for the data file  (FIXED: removed extra quote)
url = "https://raw.githubusercontent.com/AleyaNazifa/AssignmentSV2025/refs/heads/main/hfdm_data%20-%20Upload.csv"

# --- Data loader: try URL first; if it fails, simulate dummy data so app still runs ---
@st.cache_data
def load_and_process_data(data_url: str):
    """Loads and preprocesses data. If URL fails, generates realistic dummy data."""
    st.subheader("1. Data Loading and Preprocessing")

    try:
        # Try loading the real CSV from GitHub
        df_real = pd.read_csv(data_url)
        # Expecting Date in %d/%m/%Y
        df_real["Date"] = pd.to_datetime(df_real["Date"], format="%d/%m/%Y", errors="coerce")

        regions = ["southern", "northern", "central", "east_coast", "borneo"]
        # Basic validation: ensure expected columns exist; else fallback to dummy
        expected_weather = [
            "temp_s","rain_s","rh_s","temp_n","rain_n","rh_n",
            "temp_c","rain_c","rh_c","temp_ec","rain_ec","rh_ec","temp_b","rain_b","rh_b"
        ]
        if not all(c in df_real.columns for c in regions + expected_weather):
            raise ValueError("Columns missing in URL CSV. Falling back to simulated data.")

        df_real["total_cases"] = df_real[regions].sum(axis=1)
        df_real = df_real.drop_duplicates(subset=["Date"]).set_index("Date")
        df_monthly = df_real.resample("M").mean(numeric_only=True).reset_index()
        df_monthly["Year"] = df_monthly["Date"].dt.year
        df_monthly["Month"] = df_monthly["Date"].dt.month

        st.write(f"Loaded real dataset from URL. Monthly rows: {len(df_monthly)}")
        st.dataframe(df_monthly.head(), use_container_width=True)
        return df_monthly, regions

    except Exception as e:
        st.warning(f"Could not load URL data ({e}). Generating dummy dataset instead.")

        # ----- Simulate realistic dummy data (your original approach) -----
        start_date = "2009-01-01"
        end_date = "2019-12-31"
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        regions = ["southern", "northern", "central", "east_coast", "borneo"]
        weather_types = ["temp", "rain", "rh"]
        data = {}

        # simulate HFMD cases
        for r in regions:
            data[r] = np.random.randint(10, 500, size=len(dates))

        # simulate weather (temp/rain/humidity) per region-initial (s, n, c, e, b)
        for idx, r in enumerate(regions):
            initial = r[0]
            data[f"temp_{initial}"] = np.random.uniform(25 + idx*0.2, 35 - idx*0.5, len(dates))
            data[f"rain_{initial}"] = np.random.uniform(0, 20, len(dates))
            data[f"rh_{initial}"]   = np.random.uniform(70 + idx*0.5, 95 - idx*0.2, len(dates))

        df = pd.DataFrame(data, index=dates).reset_index().rename(columns={"index": "Date"})
        st.write(f"Generated a dummy dataset with {len(df)} daily rows (2009–2019).")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["total_cases"] = df[regions].sum(axis=1)
        df = df.drop_duplicates(subset=["Date"]).set_index("Date")
        df_monthly = df.resample("M").mean(numeric_only=True).reset_index()
        df_monthly["Year"] = df_monthly["Date"].dt.year
        df_monthly["Month"] = df_monthly["Date"].dt.month

        st.write(f"Resampled to {len(df_monthly)} monthly rows.")
        st.dataframe(df_monthly.head(), use_container_width=True)
        return df_monthly, regions

df_monthly, regions = load_and_process_data(url)

st.markdown("---")

# --- 2. Monthly Trend by Region (Line Plot) ---
st.header("2. Monthly HFMD Cases by Region (Trend Analysis)")
fig_A = px.line(
    df_monthly,
    x="Date",
    y=regions,
    title="Monthly HFMD Cases by Region (2009–2019)",
    labels={"value": "Average Monthly Cases", "Date": "Year", "variable": "Region"},
    line_shape="spline",
)
fig_A.update_layout(hovermode="x unified")
st.plotly_chart(fig_A, use_container_width=True)
st.markdown("---")

# --- 3. Seasonal Pattern (Heatmap) ---
st.header("3. HFMD Seasonal Pattern (Month × Year Heatmap)")
pivot = df_monthly.pivot_table(values="total_cases", index="Month", columns="Year", aggfunc="mean")
fig_B = px.imshow(
    pivot,
    text_auto=".0f",
    aspect="auto",
    color_continuous_scale=px.colors.sequential.Plasma,
    title="HFMD Seasonal Pattern (Month × Year)",
)
fig_B.update_yaxes(tickvals=list(range(1, 13)), title="Month")
fig_B.update_xaxes(title="Year")
st.plotly_chart(fig_B, use_container_width=True)
st.markdown("---")

# --- 4. Yearly Distribution (Boxplot) ---
st.header("4. Distribution of Monthly HFMD Cases per Year (Boxplot)")
fig_C = px.box(
    df_monthly,
    x="Year",
    y="total_cases",
    title="Distribution of Monthly HFMD Cases per Year",
    labels={"total_cases": "Total Monthly Cases", "Year": "Year"},
    color="Year",
)
fig_C.update_layout(showlegend=False)
st.plotly_chart(fig_C, use_container_width=True)
st.markdown("---")

# --- 5. Weather Correlation (Scatters + Trendline) ---
st.header("5. Weather Correlation Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Temperature vs HFMD Cases (Central)")
    fig_D = px.scatter(
        df_monthly,
        x="temp_c",  # requires temp_c column present or simulated
        y="total_cases",
        labels={"temp_c": "Average Monthly Temperature (°C)", "total_cases": "Total HFMD Cases"},
        title="Temperature vs HFMD Cases (Central)",
        trendline="ols",  # requires statsmodels
    )
    st.plotly_chart(fig_D, use_container_width=True)

with col2:
    st.subheader("Relative Humidity vs HFMD Cases (Central)")
    fig_E = px.scatter(
        df_monthly,
        x="rh_c",
        y="total_cases",
        labels={"rh_c": "Average Monthly Humidity (%)", "total_cases": "Total HFMD Cases"},
        title="Relative Humidity vs HFMD Cases (Central)",
        trendline="ols",  # requires statsmodels
    )
    st.plotly_chart(fig_E, use_container_width=True)

st.subheader("Correlation Matrix: Weather vs HFMD")
cols_corr = [
    "temp_s","temp_n","temp_c","temp_ec","temp_b",
    "rain_s","rain_n","rain_c","rain_ec","rain_b",
    "rh_s","rh_n","rh_c","rh_ec","rh_b","total_cases",
]
corr = df_monthly[cols_corr].corr(numeric_only=True)
fig_F = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Correlation Matrix: Weather vs HFMD",
    labels=dict(color="Correlation"),
)
fig_F.update_xaxes(tickangle=45)
st.plotly_chart(fig_F, use_container_width=True)
st.markdown("---")

# --- 6. Regional Summaries (Bar + Radar) ---
st.header("6. Regional Summaries")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Average Monthly HFMD Cases by Region")
    region_means = df_monthly[regions].mean().sort_values(ascending=False).reset_index()
    region_means.columns = ["Region", "Average Cases"]
    fig_G = px.bar(
        region_means,
        x="Region",
        y="Average Cases",
        color="Region",
        title="Average Monthly HFMD Cases by Region (2009–2019)",
        labels={"Average Cases": "Average Monthly Cases", "Region": "Region"},
    )
    st.plotly_chart(fig_G, use_container_width=True)

with col4:
    st.subheader("Normalized Weather Pattern by Region (Radar)")
    weather_params = ["temp", "rain", "rh"]
    data_radar = {"Region": regions}
    for param in weather_params:
        cols = [f"{param}_{r[0]}" for r in regions]
        means = df_monthly[cols].mean()
        norm = (means - means.min()) / (means.max() - means.min() + 1e-9)
        data_radar[param.capitalize()] = norm.values.tolist()
    df_radar = pd.DataFrame(data_radar)
    categories = ["Temp", "Rain", "Rh"]
    fig_H = go.Figure()
    for i, region in enumerate(regions):
        values = df_radar.loc[i, categories].tolist()
        values.append(values[0])
        fig_H.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=region.capitalize(),
            line_shape="spline",
        ))
    fig_H.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Normalized Weather Pattern by Region",
    )
    st.plotly_chart(fig_H, use_container_width=True)

# --- Sidebar info (cleaned) ---
st.sidebar.markdown("## Dashboard Info")
st.sidebar.markdown(f"**Monthly Rows (Processed):** {len(df_monthly)}")
st.sidebar.markdown("**Notes:** Trendlines require `statsmodels`. If not desired, remove `trendline='ols'`.")

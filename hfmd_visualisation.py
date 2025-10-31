import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go  # for multi-line plot
# statsmodels is needed only if you keep trendline="ols" in the scatter charts

# -------------------------------
# 1) Page setup & title
# -------------------------------
st.set_page_config(layout="wide", page_title="HFMD Scientific Visualization 📊")
st.title("HFMD Malaysia (2009–2019): Scientific Visualization")
st.markdown("---")

# -------------------------------
# 2) Data source controls (URL or upload)
# -------------------------------
st.subheader("Data Source")
st.caption("Use a GitHub Raw CSV link or upload a CSV file to begin.")
col_url, col_up = st.columns([2, 1])
with col_url:
    RAW_CSV_URL = st.text_input(
        "GitHub Raw CSV URL",
        value="https://raw.githubusercontent.com/<your-username>/<your-repo>/main/hfmd_data.csv",
        placeholder="https://raw.githubusercontent.com/<user>/<repo>/main/file.csv",
    )
with col_up:
    uploaded = st.file_uploader("...or Upload CSV", type=["csv"])

# -------------------------------
# 3) Loader & cache
# -------------------------------
@st.cache_data
def load_and_prepare(src):
    """Load dataset (daily) and return daily + monthly dataframes."""
    df = pd.read_csv(src)

    # Expecting columns including:
    # Date, southern, northern, central, east_coast, borneo,
    # temp_s, rain_s, rh_s, temp_n, rain_n, rh_n, temp_c, rain_c, rh_c,
    # temp_ec, rain_ec, rh_ec, temp_b, rain_b, rh_b
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")

    # Create total cases across regions
    regions = ["southern", "northern", "central", "east_coast", "borneo"]
    df["total_cases"] = df[regions].sum(axis=1)

    # Drop duplicates
    df = df.drop_duplicates()

    # Monthly aggregation
    df_m = df.resample("M", on="Date").mean(numeric_only=True).reset_index()
    df_m["Year"] = df_m["Date"].dt.year
    df_m["Month"] = df_m["Date"].dt.month

    return df, df_m

# Choose source (upload overrides URL if present)
if uploaded is not None:
    df_daily, df = load_and_prepare(uploaded)
elif RAW_CSV_URL.strip():
    df_daily, df = load_and_prepare(RAW_CSV_URL.strip())
else:
    st.info("Provide a valid GitHub Raw CSV URL or upload a CSV file to proceed.")
    st.stop()

# -------------------------------
# 4) KPI metrics (like your example)
# -------------------------------
avg_total = df["total_cases"].mean()
max_total = df["total_cases"].max()
last_year = int(df["Year"].max())
avg_last_year = df.loc[df["Year"] == last_year, "total_cases"].mean()

avg_temp = df["temp_c"].mean() if "temp_c" in df.columns else np.nan
avg_humid = df["rh_c"].mean() if "rh_c" in df.columns else np.nan

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Monthly Total Cases", f"{avg_total:.1f}", help="Mean of monthly total HFMD cases", border=True)
col2.metric("Max Monthly Cases", f"{max_total:.0f}", help="Highest monthly total (2009–2019)", border=True)
col3.metric(f"Avg Cases in {last_year}", f"{(avg_last_year if not np.isnan(avg_last_year) else 0):.1f}", border=True)
col4.metric("Avg Temp / RH (Central)", f"{(avg_temp if not np.isnan(avg_temp) else 0):.1f}°C / {(avg_humid if not np.isnan(avg_humid) else 0):.1f}%", border=True)

st.markdown("---")

# -------------------------------
# 5) Data preview
# -------------------------------
st.header("1) Data Preview")
st.caption("Top rows of the **daily** dataset and a quick numeric summary of the **monthly** aggregation.")
st.dataframe(df_daily.head(), use_container_width=True)

with st.expander("Monthly aggregation – quick stats"):
    st.dataframe(df.describe(numeric_only=True).T, use_container_width=True)

st.markdown("---")

# -------------------------------
# 6) Regional distribution (pie + bar)
# -------------------------------
st.header("2) Regional Distribution (Average Monthly Cases)")
region_means = df[["southern", "northern", "central", "east_coast", "borneo"]].mean().reset_index()
region_means.columns = ["Region", "Avg_Monthly_Cases"]

c1, c2 = st.columns(2)
with c1:
    fig_reg_pie = px.pie(
        region_means,
        names="Region",
        values="Avg_Monthly_Cases",
        title="Share of Average Monthly HFMD Cases by Region",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_reg_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_reg_pie, use_container_width=True)

with c2:
    fig_reg_bar = px.bar(
        region_means,
        x="Region", y="Avg_Monthly_Cases",
        title="Average Monthly HFMD Cases by Region",
        labels={"Avg_Monthly_Cases": "Average Monthly Cases"},
        color="Region",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_reg_bar.update_layout(xaxis_title="Region", yaxis_title="Average Monthly Cases")
    st.plotly_chart(fig_reg_bar, use_container_width=True)

st.info(
    "Central and Southern typically contribute a larger portion of monthly HFMD cases; "
    "East Coast & Borneo are lower on average."
)

st.markdown("---")

# -------------------------------
# 7) Annual distribution (box)
# -------------------------------
st.header("3) Annual Distribution of Monthly Total Cases (Box Plot)")
try:
    fig_box = px.box(
        df, x="Year", y="total_cases", color="Year",
        title="Distribution of Monthly Total HFMD Cases by Year",
        labels={"total_cases": "Monthly Total Cases"},
        points="outliers"
    )
    fig_box.update_layout(xaxis_title="Year", yaxis_title="Monthly Total Cases", showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    st.info(
        "Some years (e.g., 2012, 2018) show higher medians and wider spread, indicating larger or more "
        "variable monthly outbreaks."
    )
except Exception as e:
    st.warning(f"Box plot could not be generated: {e}")

st.markdown("---")

# -------------------------------
# 8) Seasonality heatmap (Month × Year)
# -------------------------------
st.header("4) Seasonality (Month × Year Heatmap)")
try:
    pivot = df.pivot_table(values="total_cases", index="Month", columns="Year", aggfunc="mean")
    fig_heat = px.imshow(
        pivot, aspect="auto", origin="lower",
        title="HFMD Seasonal Pattern (Month × Year)",
        labels=dict(x="Year", y="Month", color="Avg Monthly Cases"),
        color_continuous_scale="OrRd"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.info("Clear seasonal peaks typically appear around mid-year months; intensity varies by year.")
except Exception as e:
    st.warning(f"Heatmap could not be generated: {e}")

st.markdown("---")

# -------------------------------
# 9) Weather relationships (central reference)
# -------------------------------
st.header("5) Weather Relationships (Central Reference)")
sc1, sc2 = st.columns(2)

with sc1:
    try:
        fig_sc1 = px.scatter(
            df, x="temp_c", y="total_cases",
            title="Temperature vs HFMD Cases (Central)",
            labels={"temp_c": "Average Monthly Temperature (°C)", "total_cases": "Monthly Total HFMD Cases"},
            opacity=0.65,
            trendline="ols"  # requires statsmodels
        )
        st.plotly_chart(fig_sc1, use_container_width=True)
    except Exception as e:
        st.warning(f"Temperature scatter failed: {e}")

with sc2:
    try:
        fig_sc2 = px.scatter(
            df, x="rh_c", y="total_cases",
            title="Relative Humidity vs HFMD Cases (Central)",
            labels={"rh_c": "Average Monthly Humidity (%)", "total_cases": "Monthly Total HFMD Cases"},
            opacity=0.65,
            trendline="ols"  # requires statsmodels
        )
        st.plotly_chart(fig_sc2, use_container_width=True)
    except Exception as e:
        st.warning(f"Humidity scatter failed: {e}")

st.info(
    "HFMD incidence tends to increase with warmer temperatures and higher humidity, "
    "suggesting environment-driven transmission suitability during warm & humid months."
)

st.markdown("---")

# -------------------------------
# 10) Regional trends (multi-line)
# -------------------------------
st.header("6) Regional Trends Over Time")
try:
    fig_lines = go.Figure()
    for col in ["southern", "northern", "central", "east_coast", "borneo"]:
        fig_lines.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=col.capitalize()))
    fig_lines.update_layout(
        title="Monthly HFMD Cases by Region (2009–2019)",
        xaxis_title="Year", yaxis_title="Average Monthly Cases"
    )
    st.plotly_chart(fig_lines, use_container_width=True)
    st.success("Analysis complete! All charts and interpretations are displayed.")
except Exception as e:
    st.warning(f"Regional trends could not be generated: {e}")

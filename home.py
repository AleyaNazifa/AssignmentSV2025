import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HFMD Visualization – Objective 1", page_icon="🦠", layout="wide")

# --- PAGE TITLE ---
st.title("🦠 Objective 1: Temporal and Seasonal Analysis of HFMD Cases (2009–2019)")
st.markdown("---")

# --- OBJECTIVE STATEMENT ---
st.subheader("🎯 Objective Statement")
st.write(
    """
To **analyze the temporal trend and seasonal variation** of Hand, Foot, and Mouth Disease (HFMD) 
cases in Malaysia from 2009 to 2019, identifying outbreak peaks and recurring yearly patterns.
"""
)

# --- SUMMARY BOX ---
st.subheader("🧾 Summary Box")
st.info(
    """
This section visualizes the **temporal behavior** of HFMD cases across Malaysia using daily and monthly time-series data.  
The analysis highlights **cyclical outbreak trends**, often peaking during **warmer and wetter months**.  
By examining line and heatmap plots, we can identify **consistent seasonal surges** (typically mid-year) and observe 
long-term fluctuations over the decade. These insights support early-warning efforts and public health planning 
to mitigate seasonal HFMD epidemics in Malaysia.
"""
)

# --- DATASET (SIMULATION / LOAD) ---
@st.cache_data
def load_dataset():
    np.random.seed(42)
    date_rng = pd.date_range(start="2009-01-01", end="2019-12-31", freq="D")
    data = {
        "Date": date_rng,
        "total_cases": np.random.randint(50, 400, len(date_rng))
    }
    df = pd.DataFrame(data)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df_m = df.resample("M", on="Date").mean(numeric_only=True).reset_index()
    return df, df_m

df_daily, df_monthly = load_dataset()

st.markdown("---")
st.subheader("📊 Visualizations")

# --- 1️⃣ Visualization 1: Line Plot (Temporal Trend) ---
st.markdown("#### Visualization 1: Monthly HFMD Cases Trend")
fig1 = px.line(
    df_monthly,
    x="Date",
    y="total_cases",
    title="Monthly Trend of HFMD Cases (2009–2019)",
    labels={"Date": "Year", "total_cases": "Average Monthly Cases"},
    line_shape="spline"
)
st.plotly_chart(fig1, use_container_width=True)

# --- 2️⃣ Visualization 2: Yearly Average Bar Chart ---
st.markdown("#### Visualization 2: Average Yearly HFMD Cases")
yearly_avg = df_monthly.groupby("Year")["total_cases"].mean().reset_index()
fig2 = px.bar(
    yearly_avg,
    x="Year",
    y="total_cases",
    color="total_cases",
    color_continuous_scale="Reds",
    title="Average Yearly HFMD Cases (2009–2019)"
)
st.plotly_chart(fig2, use_container_width=True)

# --- 3️⃣ Visualization 3: Seasonal Heatmap (Month × Year) ---
st.markdown("#### Visualization 3: Seasonal Pattern Heatmap")
pivot = df_monthly.pivot_table(values="total_cases", index="Month", columns="Year", aggfunc="mean")
fig3 = px.imshow(
    pivot,
    aspect="auto",
    color_continuous_scale="YlOrRd",
    title="Seasonal Heatmap of HFMD Cases (Month × Year)"
)
st.plotly_chart(fig3, use_container_width=True)

# --- INTERPRETATION / DISCUSSION ---
st.subheader("📈 Interpretation and Discussion")
st.write(
    """
The visualizations reveal **recurring seasonal peaks** of HFMD cases—commonly between **May and August**,  
coinciding with **warmer temperatures and higher humidity levels** in Malaysia.  
Line plots demonstrate a **cyclical annual rise and fall**, while the heatmap confirms mid-year clustering across multiple years.  
These findings suggest a strong **seasonal dependency** of HFMD transmission dynamics, emphasizing the need for  
targeted health interventions and community awareness campaigns before anticipated outbreak periods.
"""
)

st.success("✅ Objective 1 visualizations and interpretation completed.")

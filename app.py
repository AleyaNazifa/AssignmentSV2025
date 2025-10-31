
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="HFMD Malaysia (2009–2019) – Scientific Visualization", layout="wide")

# ------------------------------
# Sidebar: Dataset Loader
# ------------------------------
st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload the HFMD CSV file", type=["csv"])

# Default path hint (for local testing)
default_path = None  # e.g., "hfdm_data - Upload.csv"

@st.cache_data(show_spinner=False)
def load_and_prepare(csv_bytes_or_path):
    # Load dataset
    if isinstance(csv_bytes_or_path, (str, bytes, io.BytesIO)):
        df = pd.read_csv(csv_bytes_or_path)
    else:
        df = pd.read_csv(csv_bytes_or_path)

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")

    # Create total cases
    regions = ["southern","northern","central","east_coast","borneo"]
    df["total_cases"] = df[regions].sum(axis=1)

    # Drop duplicates
    df = df.drop_duplicates()

    # Monthly aggregation
    df_monthly = df.resample("M", on="Date").mean(numeric_only=True).reset_index()

    # Add helper columns
    df_monthly["Year"] = df_monthly["Date"].dt.year
    df_monthly["Month"] = df_monthly["Date"].dt.month

    return df, df_monthly

if uploaded is not None:
    df_daily, df_monthly = load_and_prepare(uploaded)
elif default_path:
    df_daily, df_monthly = load_and_prepare(default_path)
else:
    st.info("Please upload the HFMD CSV file in the sidebar to begin.")
    st.stop()

# ------------------------------
# Header
# ------------------------------
st.title("HFMD Malaysia (2009–2019) – Scientific Visualization")
st.markdown("""
This Streamlit app presents three pages of scientific visualizations for the HFMD + Weather dataset (Malaysia).
Use the navigation below to switch pages. Each page corresponds to one visualization objective.
""")

page = st.radio(
    "Go to:",
    options=["Page 1 – Temporal Trend Analysis", "Page 2 – Weather Relationships", "Page 3 – Regional Comparison"],
    horizontal=True
)

# ------------------------------
# Helper: Matplotlib plotting wrappers
# ------------------------------
def plot_multiline_regions(df_monthly):
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ["southern","northern","central","east_coast","borneo"]:
        ax.plot(df_monthly["Date"], df_monthly[col], label=col.capitalize())
    ax.set_title("Monthly HFMD Cases by Region (2009–2019)")
    ax.set_xlabel("Year"); ax.set_ylabel("Average Monthly Cases")
    ax.grid(True, alpha=0.3); ax.legend()
    st.pyplot(fig)

def plot_seasonality_heatmap(df_monthly):
    # Build pivot Month x Year
    tmp = df_monthly.copy()
    pivot = tmp.pivot_table(values="total_cases", index="Month", columns="Year", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot, aspect="auto")
    fig.colorbar(im, ax=ax, label="Total Monthly Cases")
    ax.set_title("HFMD Seasonal Pattern (Month × Year)")
    ax.set_xlabel("Year"); ax.set_ylabel("Month")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(range(0, 12)); ax.set_yticklabels(range(1, 13))
    ax.grid(False)
    st.pyplot(fig)

def plot_yearly_boxplot(df_monthly):
    tmp = df_monthly.copy()
    years = sorted(tmp["Year"].unique())
    data_to_plot = [tmp.loc[tmp["Year"]==y, "total_cases"].dropna().values for y in years]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data_to_plot, labels=years, showfliers=True)
    ax.set_title("Distribution of Monthly HFMD Cases per Year")
    ax.set_xlabel("Year"); ax.set_ylabel("Total Monthly Cases")
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_scatter_with_fit(x, y, x_label, y_label, title):
    # Simple linear fit without specifying colors
    mask = ~(x.isna() | y.isna())
    x_valid, y_valid = x[mask], y[mask]
    coef = np.polyfit(x_valid, y_valid, 1)
    poly1d_fn = np.poly1d(coef)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_valid, y_valid, alpha=0.6)
    xs = np.linspace(x_valid.min(), x_valid.max(), 100)
    ax.plot(xs, poly1d_fn(xs), linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_corr_heatmap(df_monthly, cols, title):
    corr = df_monthly[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title(title)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    plt.tight_layout()
    st.pyplot(fig)

def plot_bar_region_means(df_monthly):
    region_means = df_monthly[["southern","northern","central","east_coast","borneo"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(region_means.index, region_means.values)
    ax.set_title("Average Monthly HFMD Cases by Region (2009–2019)")
    ax.set_xlabel("Region"); ax.set_ylabel("Average Monthly Cases")
    plt.setp(ax.get_xticklabels(), rotation=15)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_polar_weather(df_monthly):
    regions = ["Southern","Northern","Central","East Coast","Borneo"]
    temp_means = [df_monthly["temp_s"].mean(), df_monthly["temp_n"].mean(), df_monthly["temp_c"].mean(),
                  df_monthly["temp_ec"].mean(), df_monthly["temp_b"].mean()]
    rain_means = [df_monthly["rain_s"].mean(), df_monthly["rain_n"].mean(), df_monthly["rain_c"].mean(),
                  df_monthly["rain_ec"].mean(), df_monthly["rain_b"].mean()]
    rh_means   = [df_monthly["rh_s"].mean(),   df_monthly["rh_n"].mean(),   df_monthly["rh_c"].mean(),
                  df_monthly["rh_ec"].mean(),   df_monthly["rh_b"].mean()]

    def normalize(v):
        v = np.array(v, dtype=float)
        return (v - v.min()) / (v.max() - v.min() + 1e-9)

    angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot(111, polar=True)
    for i, reg in enumerate(regions):
        values = [normalize(temp_means)[i], normalize(rain_means)[i], normalize(rh_means)[i]]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, alpha=0.8, label=reg)
    ax.set_title("Normalized Weather Pattern by Region")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.05))
    st.pyplot(fig)

def plot_region_trends(df_monthly):
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ["southern","northern","central","east_coast","borneo"]:
        ax.plot(df_monthly["Date"], df_monthly[col], label=col.capitalize())
    ax.set_title("HFMD Trends by Region (2009–2019)")
    ax.set_xlabel("Year"); ax.set_ylabel("Monthly HFMD Cases")
    ax.grid(True, alpha=0.3); ax.legend()
    st.pyplot(fig)

# ------------------------------
# PAGES
# ------------------------------
if page.startswith("Page 1"):
    st.subheader("Objective: Temporal Trend Analysis")
    st.caption("To visualize the temporal trends and seasonal variations of HFMD cases across Malaysian regions from 2009 to 2019.")

    with st.expander("Summary (100–150 words)"):
        st.write(\"\"\"
The visualizations reveal consistent seasonal peaks of HFMD cases across 2009–2019, with higher incidence in mid-year months. 
The multi-line plot shows recurrent cycles for all regions, while the heatmap emphasizes stronger intensities during specific months. 
Annual boxplots indicate that some years (e.g., 2012 and 2018) experienced higher variability and outlier months. 
These temporal patterns suggest a repeatable seasonal signal in HFMD incidence, likely influenced by climatic and behavioral factors 
(e.g., school cycles, humidity, and temperature). Understanding these cycles helps anticipate outbreak periods and plan public-health interventions.
        \"\"\")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**A. Monthly HFMD Cases by Region**")
        plot_multiline_regions(df_monthly)
    with c2:
        st.markdown("**B. Seasonal Pattern (Month × Year)**")
        plot_seasonality_heatmap(df_monthly)

    st.markdown("**C. Annual Distribution of Monthly Total Cases**")
    plot_yearly_boxplot(df_monthly)

elif page.startswith("Page 2"):
    st.subheader("Objective: Weather Relationships")
    st.caption("To examine how temperature, rainfall, and relative humidity correlate with HFMD incidence across Malaysian regions.")

    with st.expander("Summary (100–150 words)"):
        st.write(\"\"\"
Correlation-oriented plots indicate that HFMD cases tend to rise with warmer temperatures and elevated relative humidity, 
while rainfall shows weaker and less consistent associations. Simple regression overlays suggest a positive relationship 
with temperature and humidity, consistent with virological and epidemiological expectations in tropical climates. 
The correlation matrix further clarifies which weather variables are most related to HFMD incidence. 
These results support weather-aware surveillance and can guide early alerts when climatic conditions become favorable for transmission.
        \"\"\")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**A. Temperature vs HFMD Cases**")
        plot_scatter_with_fit(df_monthly["temp_c"], df_monthly["total_cases"],
                              "Average Monthly Temperature (°C)", "Total HFMD Cases",
                              "Temperature vs HFMD Cases")
    with c2:
        st.markdown("**B. Humidity vs HFMD Cases**")
        plot_scatter_with_fit(df_monthly["rh_c"], df_monthly["total_cases"],
                              "Average Monthly Humidity (%)", "Total HFMD Cases",
                              "Relative Humidity vs HFMD Cases")

    st.markdown("**C. Correlation Matrix: Weather vs HFMD**")
    cols = ['temp_s','temp_n','temp_c','temp_ec','temp_b',
            'rain_s','rain_n','rain_c','rain_ec','rain_b',
            'rh_s','rh_n','rh_c','rh_ec','rh_b','total_cases']
    plot_corr_heatmap(df_monthly, cols, "Correlation Matrix: Weather vs HFMD")

elif page.startswith("Page 3"):
    st.subheader("Objective: Regional Comparison")
    st.caption("To compare HFMD incidence and associated meteorological conditions among Malaysian regions (Central, Northern, Southern, East Coast, Borneo).")

    with st.expander("Summary (100–150 words)"):
        st.write(\"\"\"
Regional comparisons show higher average HFMD incidence in Central and Southern Malaysia, 
likely reflecting greater population density and closer child-to-child interactions. 
Borneo and the East Coast tend to exhibit lower, more variable levels. 
A normalized polar plot of average temperature, rainfall, and humidity reveals that regions with warmer and more humid conditions 
correspond to higher HFMD rates, supporting the climate–disease linkage found earlier. 
Trend lines by region illustrate that peaks often co-occur temporally, but with differing magnitudes across regions.
        \"\"\")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**A. Average Monthly HFMD Cases by Region**")
        plot_bar_region_means(df_monthly)
    with c2:
        st.markdown("**B. Normalized Weather Pattern by Region**")
        plot_polar_weather(df_monthly)

    st.markdown("**C. Regional HFMD Trends (2009–2019)**")
    plot_region_trends(df_monthly)

st.markdown("---")
st.markdown("**Tip:** Use the expander 'Summary' on each page to copy the 100–150 words directly into your PDF report.")


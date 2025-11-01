import streamlit as st

# --- Page Setup ---
st.set_page_config(page_title="HFMD Malaysia Dashboard", page_icon="🦠", layout="wide")

# --- Title and Intro Section ---
st.title("🦠 HFMD Malaysia (2009–2019) – Scientific Visualization Dashboard")
st.markdown(
    """
Welcome to the **Hand, Foot, and Mouth Disease (HFMD) Scientific Visualization App** —  
an interactive dashboard built to explore the **epidemiological trends, regional patterns, and 
climatic relationships** of HFMD cases across Malaysia from **2009 to 2019**.

---

### 🎯 Project Overview
This dashboard was developed as part of the course **JIE42303 – Scientific Visualization**  
at **Universiti Malaysia Kelantan (UMK)**. It allows users to visualize HFMD patterns using
real or simulated datasets, offering insight into:
- 📅 **Temporal Trends** – Changes in HFMD incidence over time  
- 🌦️ **Weather Influence** – Temperature, rainfall, and humidity relationships  
- 🗺️ **Regional Comparison** – Disease distribution across Malaysian regions  

Each visualization is paired with clear interpretations to help connect scientific patterns with public health implications.

---

### 📂 Dashboard Structure
Use the **navigation bar (Menu)** on the left to explore:
- 🏠 **Homepage** – Project background and objectives  
- 📊 **HFMD and Weather Analysis** – Full interactive visualization dashboard

---

### 👩‍💻 Developer Information
**Author:** Najwatul Intan Tasnim (Wawa Anafi)  
**Course:** JIE42303 – Scientific Visualization  
**Institution:** Universiti Malaysia Kelantan (UMK)  
**Year:** 2025  

---

### 🧾 Dataset Source
- **Dataset:** [HFMD and Weather Data (Malaysia, 2009–2019)](https://data.mendeley.com/datasets/fkr7cwf6j8/1)  
- **Source:** Mendeley Data  
- **Type:** Time-series data combining HFMD case counts and meteorological factors  
- **Variables:** Regional HFMD cases, temperature, rainfall, relative humidity  

---

> 💡 *Tip:* You can upload your own CSV dataset (same format) in the **HFMD and Weather Analysis** page to visualize new or updated data interactively.
"""
)

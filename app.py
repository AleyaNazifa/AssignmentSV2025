import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HFMD Malaysia Visualization",
    page_icon="🦠",
)

# --- IMPORT YOUR PAGES ---
home = st.Page('home.py', title='Home', icon=":material/home:", default=True)
visualise = st.Page('hfmd_visualisation.py', title='HFMD Scientific Visualization', icon=":material/bar_chart:")

# --- PAGE NAVIGATION ---
pg = st.navigation(
    {
        "Main Menu": [home, visualise],
    }
)

pg.run()

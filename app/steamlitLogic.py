import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from app.data_utils import clean_laptops_df
from app.clustering_utils import get_available_numeric_cols, run_kmeans
import plotly.express as px
import plotly.graph_objects as go

def updatePemilihanColom():
    """Update selected columns in session state based on user selections."""
    st.session_state.selected_columns = {
        "Inches": st.session_state.get("inchesColSelection", None),
        "Screen Resolution": st.session_state.get("screenResolutionColSelection", None),
        "CPU": st.session_state.get("cpuColSelection", None),
        "RAM": st.session_state.get("ramColSelection", None),
        "Memory": st.session_state.get("memoryColSelection", None),
        "GPU": st.session_state.get("gpuColSelection", None),
        "Weight": st.session_state.get("weightColSelection", None),
        "Price": st.session_state.get("priceColSelection", None),
        "ID": st.session_state.get("idColSelection", None),
        "Name": st.session_state.get("nameColSelection", None),
        "Company": st.session_state.get("companyColSelection", None),
        "Laptop Type": st.session_state.get("laptopTypeColSelection", None),
        "Operating System": st.session_state.get("operatingSystemColSelection", None)
    }
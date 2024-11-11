import pandas as pd
import streamlit as st


@st.cache_data(persist=True)
def load_data():
    return pd.read_csv("Guardados/clean_crime_rate.csv")

@st.cache_data(persist=True)
def total_data():
    return pd.read_csv("Guardados/Total_crime_rate.csv")
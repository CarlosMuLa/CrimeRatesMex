import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import time


# Cargar datos
@st.cache_data
def load_data():
    return pd.read_csv("./Guardados/clean_crime_rate.csv", encoding = "latin1")
@st.cache_data
def load_data2():
    return pd.read_csv("./Data/crimen_nac.csv", encoding = "latin1")

data_df = load_data()
data_df2 = load_data2()

st.title("Mexico Crime Rates")
st.divider()
st.write("El data frame con el que empezamos a trabajar es una concatenacion de dos bases de datos")
st.write()




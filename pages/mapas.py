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
    return pd.read_csv("./Guardados/clean_crime_rate.csv")
#esto sirve para que no se cargue cada vez que se refresca la página y se guarde en cache

data_df = load_data()

st.title("Mexico Crime Rates Maps")
estados = data_df["Entidad"].unique()
    
    # Seleccionar estados para visualizar en el mapa
selected_states = st.multiselect("Selecciona los estados a visualizar:", options=estados, default=estados)
    
if selected_states:
        filtered_states = data_df[data_df["Entidad"].isin(selected_states)]
        crimes = filtered_states["Tipo de delito"].unique()
        selected_crime = st.selectbox("Selecciona el tipo de delito a visualizar:", options=crimes)

        
        if selected_crime:
            filtered_crime = filtered_states[filtered_states["Tipo de delito"]==selected_crime]
            #sumar la cantidad de delitos por estado de cada mes
            
            #agrupar la cantidad total de estados
            total_crimesby_state = filtered_crime.groupby("Entidad").size().reset_index(name="Total")
            layer = pdk.Layer('HexagonLayer', data=filtered_crime, get_position='[lon, lat]', radius=20000, elevation_scale=4, elevation_range=[0, 1000], pickable=True, extruded=True)
            view_state = pdk.ViewState(latitude=23.6345, longitude=-102.5528, zoom=4, bearing=0, pitch=0)
            r = pdk.Deck(layers=[layer], initial_view_state=view_state)
            st.pydeck_chart(r)

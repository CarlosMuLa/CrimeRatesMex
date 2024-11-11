import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import time
from sklearn.preprocessing import MinMaxScaler
from utils import load_data, total_data

data_df = load_data()
total_df = total_data()
scaler = MinMaxScaler()
# Cargar datos

st.title("Mexico Crime Rates Maps")

estados = total_df["Entidad"].unique()
    
    # Seleccionar estados para visualizar en el mapa
selected_states = st.multiselect("Selecciona los estados a visualizar:", options=estados, default=estados)
    
if selected_states:
        filtered_states = total_df[data_df["Entidad"].isin(selected_states)]
        crimes = filtered_states["Tipo de delito"].unique()
        selected_crime = st.selectbox("Selecciona el tipo de delito a visualizar:", options=crimes)

        
        if selected_crime:
            filtered_crime = filtered_states[filtered_states["Tipo de delito"]==selected_crime]
            #sumar la cantidad de delitos por estado de cada mes
            
            #agrupar la cantidad total de estados
            total_crimesby_state = filtered_crime.groupby("Entidad").size().reset_index(name="Total")
            total_crimesby_state['Normalized Total'] = scaler.fit_transform(total_crimesby_state[['Total']])
            total_crimesby_state['Scaled Radius'] = total_crimesby_state['Normalized Total'] * 50000  # Ajusta el factor según sea necesario
            #normalizar los datos
            print(total_crimesby_state)
            layer = pdk.Layer('ScatterplotLayer', data=filtered_crime, get_position='[lon, lat]', radius='Scaked Radius', elevation_scale=4, elevation_range=[0, 1000], pickable=True, extruded=True,get_radius='Total * 15',get_fill_color=[255, 0, 0,160],)
            view_state = pdk.ViewState(latitude=23.6345, longitude=-102.5528, zoom=4, bearing=0, pitch=0)
            r = pdk.Deck(layers=[layer], initial_view_state=view_state)
            st.pydeck_chart(r)

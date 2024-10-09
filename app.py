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

#----- Configuración Inicial del Panel Central --------------------
st.title("Mexico Crime Rates")

# Usar checkbox en lugar de botones
col1, col2, col3 = st.columns(3)
with col1:
    show_home = st.checkbox("Inicio", value=True)
with col2:
    show_mapa = st.checkbox("Mapas")
with col3:
    show_grafica = st.checkbox("Gráficas")

# Asegurarse de que solo una opción esté seleccionada
if show_mapa:
    show_home = False
    show_grafica = False
elif show_grafica:
    show_home = False
    show_mapa = False
elif not (show_home or show_mapa or show_grafica):
    show_home = True

if show_home:
    st.header("Bienvenido a Mexico Crime Rates")
    st.divider()
    st.write("Proyecto realizado por: Carlos Muñiz, Renata Tejeda, Yamile Garcia y Arlyn Linette")
    st.write("Este proyecto tiene como objetivo analizar y visualizar los datos de crimen en México")
    st.write("Para comenzar, selecciona una de las opciones en la barra de navegación")
    st.image("logo1.png")

if show_mapa:
    st.header("Mapas")
    st.divider()
    estados = data_df["Entidad"].unique()
    
    # Seleccionar estados para visualizar en el mapa
    selected_states = st.multiselect("Selecciona los estados a visualizar:", options=estados)
    
    if selected_states:
        filtered_states = data_df[data_df["Entidad"].isin(selected_states)]
        crimes = filtered_states["Tipo de delito"].unique()
        selected_crime = st.selectbox("Selecciona el tipo de delito a visualizar:", options=crimes)
        
        if selected_crime:
            filtered_crime = filtered_states[filtered_states["Tipo de delito"]==selected_crime]
            


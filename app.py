import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import time


#----- Configuración Inicial del Panel Central --------------------

st.title("Mexico Crime Rates")


st.header("Bienvenido a Mexico Crime Rates")
st.divider()
st.write("Proyecto realizado por: Carlos Muñiz, Renata Tejeda, Yamile Garcia y Arlyn Linette")
st.write("Este proyecto tiene como objetivo analizar y visualizar los datos de crimen en México")
st.write("Para comenzar, selecciona una de las opciones en la barra de navegación")
#crear 3 columnas
col1,col2,col3 = st.columns(3)
with col1:
    if st.button("Mapas"):
        st.switch_page("pages/mapas.py")   
with col2:
    if st.button("KMeans"):
        st.switch_page("pages/modelos.py")
with col3:  
    if st.button("Regresion Lineal"):
        st.switch_page("pages/modelo1.py")
st.image("logo1.png")





import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import time


#----- Configuración Inicial del Panel Central --------------------
st.title("Mexico Crime Rates")

#selected= option_menu(menu_title=None, options=[ "Mapas", "Gráficas", "Modelo", "Acerca de"],icons=["bar-chart-fill","bezier","geo","info"], orientation="horizontal")
#if selected == "Mapas":
#    st.switch_page("pages/mapas.py")

st.header("Bienvenido a Mexico Crime Rates")
st.divider()
st.write("Proyecto realizado por: Carlos Muñiz, Renata Tejeda, Yamile Garcia y Arlyn Linette")
st.write("Este proyecto tiene como objetivo analizar y visualizar los datos de crimen en México")
st.write("Para comenzar, selecciona una de las opciones en la barra de navegación")
st.image("logo1.png")





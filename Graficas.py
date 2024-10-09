import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

#lectura de datos
df_before = pd.read_csv('Data/crimen_nac.csv', encoding='latin1')
df_after = pd.read_csv('Guardados/clean_crime_rate.csv')

st.title("Análisis Antes y Después de la Limpieza")

#datos originales
st.write("## Datos Originales")
st.write(df_before.head())

#numeros y filas del df original
num_filas_before, num_columnas_before = df_before.shape
st.write(f"Número de filas en datos originales: {num_filas_before}")
st.write(f"Número de columnas en datos originales: {num_columnas_before}")

#tipos de datos del df original
st.write("## Tipos de Datos (Datos Originales)")
st.write(df_before.dtypes)

#datos ya limpios
st.write("## Datos Después de la Limpieza")
st.write(df_after.head())

#num y filas del df limpio
num_filas_after, num_columnas_after = df_after.shape
st.write(f"Número de filas en datos limpios: {num_filas_after}")
st.write(f"Número de columnas en datos limpios: {num_columnas_after}")

#tipos de datos del df limpio
st.write("## Tipos de Datos (Datos Limpios)")
st.write(df_after.dtypes)



#lista de tipos de delito para el selectbox
tipos_delito = df_before['Tipo de delito'].unique()
#elegir el tipo de delito
tipo_selected = st.sidebar.selectbox('Elección de Tipo de Delito:', tipos_delito)

# contador de filas por tipo de delito antes de la limpieza
before_count = df_before[df_before['Tipo de delito'] == tipo_selected].shape[0]
#contador de filas por tipo de delito después de la limpieza
after_count = df_after[df_after['Tipo de delito'] == tipo_selected].shape[0]

#grafica
st.subheader(f'Comparativa de {tipo_selected} Antes y Después de la Limpieza')
fig, ax = plt.subplots()
bars = ax.bar(['Antes', 'Después'], [before_count, after_count], color=['pink', 'blue'])
ax.set_ylabel('Número de Incidentes')
ax.set_title(f'Comparativa de Incidentes de {tipo_selected}')

#texto con cantidad 
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Cargar el DataFrame
crimes_df = pd.read_csv("./Guardados/clean_crime_rate.csv")

# Justificacion de uso de Modelo K-Means
texto_modelo_kmeans = """
# **Modelo K-Means (para clustering):**

El modelo K-Means es un algoritmo de clustering o agrupamiento no supervisado utilizado para dividir un conjunto de datos en un número predefinido de grupos o clusters.

- **Justificación:** Útil para segmentar zonas con patrones de criminalidad similares, lo cual puede ayudar a identificar "hotspots" o áreas de alta incidencia.
- **Aplicación:** Agrupación de entidades en clústeres basados en la cantidad y tipo de delitos.
- **Ventajas:** Permite identificar patrones ocultos y segmentar datos sin necesidad de etiquetarlos.
"""
st.markdown(texto_modelo_kmeans)

st.markdown("## Muestra de datos de crimen")
st.write("Las primeras filas del DataFrame son:")
st.dataframe(crimes_df.head())  

# Definición y justificación de columnas elegidas para la segmentación
st.markdown("## Datos elegidos para la segmentación")
texto_formateado = """
### **Tipo de delito:**
Esta variable nos ayuda a clasificar los crímenes en diferentes categorías.
Al incluirlo podemos identificar patrones y zonas con alta incidencia de crímenes específicos.

### **Subtipo de delito:**
Similar al tipo de delito, el subtipo proporciona más detalle sobre la naturaleza del crimen. Esto permite un análisis aún más detallado de la criminalidad en cada zona.

### **Sexo/Averiguación previa:**
Esta variable puede es útil para analizar las diferencias en los crímenes basados en el sexo de los involucrados o si la averiguación previa está vinculada a un patrón de criminalidad.

### **Rango de edad:**
El rango de edad es relevante, ya que la criminalidad puede variar según las diferentes edades de los individuos involucrados. Por ejemplo, los crímenes cometidos por jóvenes podrían tener patrones distintos a los cometidos por personas mayores.

### **Bien jurídico afectado:**
Esta variable brinda más información sobre el daño o impacto de cada delito. Esto nos ayuda a no solamente clasificar los delitos, sino también entender el impacto.

### **Cantidad:**
Esta variable representa el número total de delitos registrados en Aguascalientes. Es clave para medir la magnitud de la criminalidad en la zona y para detectar tendencias o cambios en la incidencia de crímenes a lo largo del tiempo.
"""
st.markdown(texto_formateado)

st.markdown("## Conversión de datos categóricos a numéricos")

le_delito = LabelEncoder()
le_subtipo = LabelEncoder()
le_sexo = LabelEncoder()
le_edad = LabelEncoder()
le_bien = LabelEncoder()

crimes_df['Tipo de delito_codificado'] = le_delito.fit_transform(crimes_df['Tipo de delito'])
crimes_df['Subtipo de delito_codificado'] = le_subtipo.fit_transform(crimes_df['Subtipo de delito'])
crimes_df['Sexo_codificado'] = le_sexo.fit_transform(crimes_df['Sexo/Averiguación previa'])
crimes_df['Rango de edad_codificado'] = le_edad.fit_transform(crimes_df['Rango de edad'])
crimes_df['Bien jurídico afectado_codificado'] = le_bien.fit_transform(crimes_df['Bien jurídico afectado'])

meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

crimes_df['Cantidad'] = crimes_df[meses].sum(axis=1)

columnas_mostrar = [
    'Tipo de delito_codificado', 
    'Subtipo de delito_codificado', 
    'Sexo_codificado', 
    'Rango de edad_codificado', 
    'Bien jurídico afectado_codificado', 
    'Cantidad'
]

st.dataframe(crimes_df[columnas_mostrar])

# Normalización de datos y justificación de uso de MinMaxScaler
st.markdown("## Normalización de datos usando MinMaxScaler")
st.write("""Decidimos usar MinMaxScaler para normalizar los datos en un rango específico entre 0 y 1, 
         asegurando que todas las características tengan la misma escala. Esto evita que las variables con rangos más 
         amplios dominen el cálculo de las distancias, mejorando la precisión del modelo.""")

numerical_columns = ['Tipo de delito_codificado', 'Subtipo de delito_codificado', 'Sexo_codificado', 'Rango de edad_codificado', 'Bien jurídico afectado_codificado', 'Cantidad']

scaler = MinMaxScaler()

crimes_df[numerical_columns] = scaler.fit_transform(crimes_df[numerical_columns])

st.dataframe(crimes_df[numerical_columns])

# Determinación del Número de Clústeres con el Método del Codo
st.markdown("## Determinación de Número de Clústers")
st.write("""Para este caso, usamos el método del codo para determinar el número de clústers en el que segmentaremos
         la información. Con base en la siguiente gráfica, determinamos que en el valor 5, la gráfica se empieza a suavizar""")

X = np.array(crimes_df[['Tipo de delito_codificado', 'Subtipo de delito_codificado', 'Sexo_codificado', 
                        'Rango de edad_codificado', 'Bien jurídico afectado_codificado', 'Cantidad']])
y = np.array(crimes_df['Entidad'])

Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i, random_state=42) for i in Nc]

inertia = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]

plt.plot(Nc, inertia, color='green')
plt.xlabel('Número de Clústers')
plt.ylabel('Inercia')
plt.title('Gráfico de Codo')

st.pyplot(plt)

# Continuará con gráficas etc, etc

import streamlit as st
import pandas as pd

# Cargar el DataFrame
crimes_df = pd.read_csv("./Guardados/clean_crime_rate.csv")

# Texto con formato Markdown
texto_modelo_kmeans = """
# **Modelo K-Means (para clustering):**

El modelo K-Means es un algoritmo de clustering o agrupamiento no supervisado utilizado para dividir un conjunto de datos en un número predefinido de grupos o clusters.

- **Justificación:** Útil para segmentar zonas con patrones de criminalidad similares, lo cual puede ayudar a identificar "hotspots" o áreas de alta incidencia.
- **Aplicación:** Agrupación de entidades en clústeres basados en la cantidad y tipo de delitos.
- **Ventajas:** Permite identificar patrones ocultos y segmentar datos sin necesidad de etiquetarlos.
"""
st.markdown(texto_modelo_kmeans)

# Mostrar las primeras filas del DataFrame
st.markdown("## Muestra de datos de crimen")
st.write("Las primeras filas del DataFrame son:")
st.dataframe(crimes_df.head())  # Usar st.dataframe para mostrar el DataFrame de manera interactiva


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
"""
st.markdown(texto_formateado)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score
from utils import load_data, total_data

# Cargar el DataFrame
crimes_df = load_data()

######## JUSTIFICACIÓN DE USO DE MODELO KMEANS ########
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

######## DEFINICIÓN Y JUSTIFICACIÓN DE COLUMNAS ELEGIDAS PARA LA SEGMENTACIÓN ########
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

######## FILTRADO DE DATAFRAME POR ENTIDAD ########
st.markdown("## Filtrado de DataFrame por Entidad")
# Crear un selector para las entidades
entidad_seleccionada = st.sidebar.selectbox("Selecciona la entidad", crimes_df['Entidad'].unique())
# Filtrar el DataFrame según la entidad seleccionada
crimes_df_filtered = crimes_df[crimes_df['Entidad'] == entidad_seleccionada]
# Mostrar el DataFrame filtrado
st.write(crimes_df_filtered)

######## CONVERSION DE DATOS CATEGÓRICOS A NÚMERICOS ########
st.markdown("## Conversión de datos categóricos a numéricos")
le_delito = LabelEncoder()
le_subtipo = LabelEncoder()
le_sexo = LabelEncoder()
le_edad = LabelEncoder()
le_bien = LabelEncoder()

# Codificar las columnas categóricas para el subconjunto de datos filtrado
crimes_df_filtered['Tipo de delito_codificado'] = le_delito.fit_transform(crimes_df_filtered['Tipo de delito'])
crimes_df_filtered['Subtipo de delito_codificado'] = le_subtipo.fit_transform(crimes_df_filtered['Subtipo de delito'])
crimes_df_filtered['Sexo_codificado'] = le_sexo.fit_transform(crimes_df_filtered['Sexo/Averiguación previa'])
crimes_df_filtered['Rango de edad_codificado'] = le_edad.fit_transform(crimes_df_filtered['Rango de edad'])
crimes_df_filtered['Bien jurídico afectado_codificado'] = le_bien.fit_transform(crimes_df_filtered['Bien jurídico afectado'])

# Crear una columna de cantidad sumando los meses o la columna correspondiente
meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
crimes_df_filtered['Cantidad'] = crimes_df_filtered[meses].sum(axis=1)

columnas_mostrar = [
    'Tipo de delito_codificado', 
    'Subtipo de delito_codificado', 
    'Sexo_codificado', 
    'Rango de edad_codificado', 
    'Bien jurídico afectado_codificado', 
    'Cantidad'
]

st.dataframe(crimes_df_filtered[columnas_mostrar])

######## NORMALIZACIÓN DE DATOS Y JUSTIFICACIÓN DE USO DE STANDARDSCALER ########
st.markdown("## Normalización de datos usando StandardScaler")
st.write("""Decidimos usar StandardScaler para normalizar los datos en un rango específico, 
         asegurando que todas las características tengan la misma escala. Esto evita que las variables con rangos más 
         amplios dominen el cálculo de las distancias, mejorando la precisión del modelo.""")

numerical_columns = ['Cantidad']  # Agrega aquí otras columnas numéricas que quieras normalizar
categorical_columns = ['Tipo de delito_codificado', 'Subtipo de delito_codificado', 'Sexo_codificado', 'Rango de edad_codificado', 'Bien jurídico afectado_codificado']

scaler = StandardScaler()
crimes_df_scaled_filtered = pd.DataFrame(scaler.fit_transform(crimes_df_filtered[numerical_columns + categorical_columns]), 
                                          columns=numerical_columns + categorical_columns)

st.dataframe(crimes_df_scaled_filtered)

######## Determinación del Número de Clústeres con el Método del Codo ########
st.markdown("## Determinación de Número de Clústers")
num_clusters = st.sidebar.slider("Selecciona el número de clústeres", min_value=2, max_value=10, value=5, step=1)

st.markdown("## Determinación de Número de Clústers")
st.write(f"""
Para este caso, usamos el método del codo para determinar el número de clústeres en el que segmentaremos
la información. Con base en la siguiente gráfica, determinamos que en el valor {num_clusters}, la gráfica se empieza a suavizar.
""")

X = np.array(crimes_df_scaled_filtered[['Tipo de delito_codificado', 'Subtipo de delito_codificado', 'Sexo_codificado', 
                        'Rango de edad_codificado', 'Bien jurídico afectado_codificado', 'Cantidad']])

Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i, random_state=42) for i in Nc]

inertia = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]

plt.plot(Nc, inertia, color='pink')
plt.xlabel('Número de Clústers')
plt.ylabel('Inercia')
plt.title('Gráfico de Codo')

st.pyplot(plt)

######## ENTRENAMIENTO DE MODELO ########
st.sidebar.title("Configuración de KMeans")
colores = ['purple', 'green', 'blue', 'pink', 'red', 'purple', 'orange', 'cyan', 'black', 'gray']

# Entrenamiento del modelo KMeans
st.markdown("## Entrenamiento de Modelo con KMeans")
st.write(f"""Al entrenar el modelo con **{num_clusters} clústeres**, graficamos los puntos correspondientes a cada clúster junto con sus 
         centroides, obteniendo la siguiente gráfica:""")

kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(X)
centroids = kmeans.cluster_centers_

colores_usados = colores[:num_clusters]

# Predicción y asignación de colores
labels = kmeans.predict(X)
asignar = [colores_usados[label] for label in labels]


# Gráfica 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar puntos del dataset
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar, s=60)

# Graficar centroides
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
           marker='*', c=colores[:num_clusters], s=1000, label='Centroides')

ax.set_title("Clusters y Centroides (3D)")
st.pyplot(fig)

st.write("""De igual manera, hemos creado un DataFrame para mostrar información más detallada sobre cada uno de estos clústeres:""")
# Crear un DataFrame con las etiquetas de los clusters
copy = pd.DataFrame()
copy['label'] = labels

# Contar los usuarios por cada cluster
cantidadGrupo = pd.DataFrame()
cantidadGrupo['color'] = colores_usados
cantidadGrupo['cantidad'] = copy.groupby('label').size().values

# Mostrar el número de usuarios por cada cluster
print("La cantidad de usuarios en cada clúster es:")
st.table(cantidadGrupo)

######## VISUALIZACIÓN DE GRUPOS Y SU CLASIFICACIÓN ########
st.markdown("## Proyecciones a partir de gráfica 3D")
st.write("""Proyecciones a partir del gráfico 3D, las cuales serán de ayuda para visualizar los grupos y su clasificación.""")

# Gráfica 1: 'Sexo/Averiguación previa' vs 'Tipo de delito'
f1 = crimes_df_scaled_filtered['Tipo de delito_codificado'].values
f2 = crimes_df_scaled_filtered['Sexo_codificado'].values
fig1, ax1 = plt.subplots()
ax1.scatter(f1, f2, c=asignar, s=70)
ax1.scatter(centroids[:, 0], centroids[:, 2], marker='*', c=colores_usados, s=1000)
ax1.set_xlabel("Tipo de delito")
ax1.set_ylabel("Sexo/Averiguación previa")
ax1.set_title("Proyección 2D: 'Tipo de delito' vs 'Sexo/Averiguación previa'")

# Gráfica 2: 'Sexo/Averiguación previa' vs 'Subtipo de delito'
f1 = crimes_df_scaled_filtered['Subtipo de delito_codificado'].values
f2 = crimes_df_scaled_filtered['Sexo_codificado'].values
fig2, ax2 = plt.subplots()
ax2.scatter(f1, f2, c=asignar, s=70)
ax2.scatter(centroids[:, 0], centroids[:, 2], marker='*', c=colores_usados, s=1000)
ax2.set_xlabel("Subtipo de delito")
ax2.set_ylabel("Sexo/Averiguación previa")
ax2.set_title("Proyección 2D: 'Subtipo de delito' vs 'Sexo/Averiguación previa'")

# Primera fila de gráficos
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig1)
with col2:
    st.pyplot(fig2)

# Gráfica 3: 'Rango de edad' vs 'Bien jurídico afectado'
f1 = crimes_df_scaled_filtered['Bien jurídico afectado_codificado'].values
f2 = crimes_df_scaled_filtered['Rango de edad_codificado'].values
fig3, ax3 = plt.subplots()
ax3.scatter(f1, f2, c=asignar, s=70)
ax3.scatter(centroids[:, 0], centroids[:, 2], marker='*', c=colores_usados, s=1000)
ax3.set_xlabel("Bien jurídico afectado")
ax3.set_ylabel("Rango Edad")
ax3.set_title("Proyección 2D: 'Bien jurídico afectado' vs 'Rango Edad'")

# Gráfica 4: 'Tipo de delito' vs 'Cantidad'
f1 = crimes_df_scaled_filtered['Tipo de delito_codificado'].values
f2 = crimes_df_scaled_filtered['Cantidad'].values
fig4, ax4 = plt.subplots()
ax4.scatter(f1, f2, c=asignar, s=70)
ax4.scatter(centroids[:, 0], centroids[:, 2], marker='*', c=colores_usados, s=1000)
ax4.set_xlabel("Tipo de delito")
ax4.set_ylabel("Cantidad de Delitos")
ax4.set_title("Proyección 2D: 'Tipo de delito' vs 'Cantidad'")

# Segunda fila de gráficos
col3, col4 = st.columns(2)
with col3:
    st.pyplot(fig3)
with col4:
    st.pyplot(fig4)

######## ANÁLISIS DE CLÚSTERS ########
# Asignar los clusters al DataFrame
crimes_df_filtered['Cluster'] = kmeans.predict(X)

# Diccionario para mapear las categorías originales de las columnas codificadas
category_mapping = {
    'Tipo de delito1': le_delito.classes_,
    'Subtipo de delito1': le_subtipo.classes_,
    'Sexo1': le_sexo.classes_,
    'Rango de edad1': le_edad.classes_,
    'Bien jurídico afectado1': le_bien.classes_
}

st.markdown("## Análisis de Clústers")
st.write("Este análisis muestra los valores más representativos de cada cluster con su propia tabla.")

# Iterar sobre cada cluster y mostrar la información
for i in range(kmeans.n_clusters):
    # Filtrar los datos para el cluster actual
    cluster_data = crimes_df_filtered[crimes_df_filtered['Cluster'] == i]
    
    # Determinar las categorías más comunes en este cluster
    most_common_delito = cluster_data['Tipo de delito_codificado'].mode()[0]
    most_common_subtipo = cluster_data['Subtipo de delito_codificado'].mode()[0]
    most_common_sexo = cluster_data['Sexo_codificado'].mode()[0]
    most_common_edad = cluster_data['Rango de edad_codificado'].mode()[0]
    most_common_bien = cluster_data['Bien jurídico afectado_codificado'].mode()[0]
    
    # Decodificar los valores para interpretación
    cluster_summary = pd.DataFrame({
        "Categoría": [
            "Delito más común",
            "Subtipo de delito más común",
            "Sexo más común",
            "Rango de edad más común",
            "Bien jurídico afectado más común"
        ],
        "Valor": [
            category_mapping['Tipo de delito1'][most_common_delito],
            category_mapping['Subtipo de delito1'][most_common_subtipo],
            category_mapping['Sexo1'][most_common_sexo],
            category_mapping['Rango de edad1'][most_common_edad],
            category_mapping['Bien jurídico afectado1'][most_common_bien]
        ]
    })
    
    # Mostrar el título y la tabla
    st.subheader(f"Cluster {i}:")
    st.write(f"**Cantidad de registros en este cluster:** {cluster_data.shape[0]}")
    st.table(cluster_summary)
    st.markdown("---")  # Separador visual


######## ANÁLISIS DE DESEMPEÑO ########
st.markdown("## Análisis de Desempeño de Modelo")
st.write("""Para analizar el desempeño del modelo usamos Silhouette Score y Davies-Bouldin 
         Index ya que es un algortimo de clustering no supervisado""")
st.markdown("### Silhouette Score")
st.write("""El **Silhouette Score** mide qué tan bien se agrupan los datos. El valor varía entre -1 y 1, 
           donde un valor cercano a 1 indica que los puntos están bien agrupados, y un valor cercano a -1 indica que los puntos están mal asignados a su clúster.""")
st.markdown("### Davies-Bouldin Index")
st.write("""El **Davies-Bouldin Index** evalúa la relación entre la dispersión interna de los clústeres y la distancia entre los clústeres. 
           Un valor más bajo indica un mejor desempeño del modelo de clustering.""")


silhouette_avg = silhouette_score(X, labels)
davies_bouldin_avg = davies_bouldin_score(X, labels)

# Mostrar los resultados
st.write(f"**Silhouette Score**: {silhouette_avg:.3f}")
st.write(f"**Davies-Bouldin Index**: {davies_bouldin_avg:.3f}")

metrics = ['Silhouette Score', 'Davies-Bouldin Index']
values = [silhouette_avg, davies_bouldin_avg]

# Crear el gráfico de barras
fig, ax = plt.subplots()
ax.bar(metrics, values, color=['pink', 'purple'])
ax.set_title('Evaluación del Modelo K-Means')
ax.set_ylabel('Valor de la Métrica')
st.pyplot(fig)

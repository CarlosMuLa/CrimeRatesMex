import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from utils import load_data, total_data
import pydeck as pdk

scaler = MinMaxScaler()

######## JUSTIFICACIÓN DE USO DE MODELO DE REGRESIÓN LINEAL ########
texto_modelo_regresion = """
# **Modelo de Regresión Lineal:**

El modelo de regresión lineal es un enfoque de modelado predictivo supervisado que se utiliza para predecir el valor de una variable dependiente basada en una o más variables independientes.

- **Justificación:** Útil para entender la relación entre variables (por ejemplo, la clave de entidad y el total anual de delitos) y hacer predicciones basadas en esa relación.
- **Aplicación:** Predicción del total anual de delitos en función de la clave de entidad.
- **Ventajas:** Fácil de interpretar y aplicar, proporciona una relación matemática clara entre las variables.
"""

# Mostrar la justificación del uso del modelo al inicio de la página
st.markdown(texto_modelo_regresion)

# Cargar el DataFrame
crimes_df = load_data()

st.markdown("## Muestra de datos de crimen")
st.write("Las primeras filas del DataFrame son:")
st.dataframe(crimes_df.head())

st.write("Fue complicado darle sentido a los datos que teníamos ya que no había muchos datos numéricos además del año y el total de delitos, por lo que decidimos hacer una regresión lineal para predecir el total de delitos en un año específico.")
st.image("./Guardados/heatmap.png")

######## CREACIÓN DE COLUMNA 'Total_Delitos' ########
st.markdown("## Creación de columna 'Total'")
crimes_df = total_data()
selected_crimes = crimes_df['Tipo de delito'].unique()

# Crear un diccionario que mapea cada tipo de delito a un índice
crimes_dict = {crime: i for i, crime in enumerate(selected_crimes)}
inverse_crimes_dict = {i: crime for crime, i in crimes_dict.items()}
st.write(crimes_df)
# Reemplazar los valores de 'Tipo de delito' con sus índices correspondientes
crimes_df['Tipo de delito'] = crimes_df['Tipo de delito'].map(crimes_dict)

# Crear un DataFrame de entidades con sus claves para luego hacer un diccionario sin índice
entidades_df = crimes_df[['Entidad', 'Clave_Ent']].drop_duplicates()
entidades_dict = entidades_df.set_index('Entidad').to_dict()['Clave_Ent']

st.markdown("### Variable Independiente")
st.write("Clave de la entidad")
st.write("La clave de la entidad es un identificador único para cada una de las entidades federativas de México. Lo que  permite al modelo distinguir claramente entre diferentes regiones del país.")
st.markdown("### Variable Dependiente")
st.write("Total de Delitos")
st.write("El total de delitos es una medida agregada que resume la incidencia delictiva en una entidad durante un año en específico. El total de delitos es un indicador que nos ayuda tomar decisiones sobre la implementación de políticas de seguridad y la seguridad en una región. Predecir el total de delitos ayuda a los gobiernos y organizaciones a tomar decisiones informadas para mejorar la seguridad pública.")

######## SELECCIÓN DE VARIABLES ########
X = crimes_df[['Clave_Ent', 'Año', 'Tipo de delito']]
y = crimes_df['Total']

######## VISUALIZACIÓN DE DATOS ########
st.markdown("## Total de delitos por entidad")
plt.figure(figsize=(12, 6))
entidad_totales = crimes_df.groupby('Entidad')['Total'].sum().sort_values(ascending=False)
sns.barplot(x=entidad_totales.index, y=entidad_totales.values, palette="viridis")
plt.xticks(rotation=90)
plt.title("Total de delitos por entidad")
plt.xlabel("Entidad")
plt.ylabel("Total de delitos")
st.pyplot(plt.gcf())
  
######## DIVISIÓN DE DATOS ########
#st.markdown("## División de los datos en conjunto de entrenamiento y prueba")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

######## ENTRENAMIENTO DEL MODELO ########
#st.markdown("## Entrenamiento del modelo de regresión lineal")
model = LinearRegression()
model.fit(X_train, y_train)

######## PREDICCIONES ########
y_pred = model.predict(X_test)

######## EVALUACIÓN DEL MODELO ########
st.markdown("## Evaluación del modelo")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
st.write(f'La pendiente significa que por cada unidad que aumenta la clave de entidad, el total anual de delitos aumenta en promedio en {model.coef_[0]:.2f} unidades.')
st.write(f"Pendiente (β1): {model.coef_[0]}")
st.write(f'La intersección significa que si la clave de entidad es 0, el total anual de delitos es de {model.intercept_:.2f} aproximadamente.')
st.write(f"Intersección (β0): {model.intercept_}")
st.write(f'El MSE es una medida de la calidad de un estimador: cuanto más cercano a 0, mejor es el modelo.')
st.write("Al tener valores grandes en nuestros datos, el MSE también será grande.")
st.write(f"MSE: {mse}")
st.write(f'El RMSE es la raíz cuadrada del MSE y es una medida de la calidad de un estimador: cuanto más cercano a 0, mejor es el modelo.')
st.write(f"RMSE: {rmse}")

######## PREDICCIÓN PARA UNA CLAVE DE ENTIDAD ESPECÍFICA ########
st.markdown("## Predicción para una clave de entidad específica")
estados = crimes_df["Entidad"].unique()
selected_state = st.selectbox("Selecciona una entidad para predecir el total anual de delitos:", options=estados)
# get only the crimes names committed in the selected state in unique values
crimes = crimes_df[crimes_df['Entidad'] == selected_state]['Tipo de delito'].unique()
crimes_names = [inverse_crimes_dict[crime] for crime in crimes] #get the names of the crimes from the dictionary
selected_crime = st.selectbox("Selecciona el tipo de delito a visualizar:", options=crimes_names) #get the selected crime
selected_crime_key = crimes_dict[selected_crime] #get the key of the selected crime
clave_entidad = entidades_dict[selected_state] #get the key of the selected state
year = st.text_input("Selecciona el año a predecir:", "2024") #input the year to predict
if year.isdigit() == False:
    st.error("Por favor ingrese un año válido") #if the input is not a year, ask the user to input a valid year
else:
    prediccion = model.predict([[clave_entidad, int(year), selected_crime_key]]) #predict the total of crimes
    st.write(f"La predicción para la clave de entidad {clave_entidad} ({selected_state}) para el año {year} es de {prediccion[0]:.2f} aproximadamente.") #show the prediction

# Generar un DataFrame con el caso base y la predicción
prediction = pd.DataFrame({'Clave_Ent': [clave_entidad], 'Año': [int(year)], 'Key de Tipo de delito': [selected_crime_key], 'Total': [prediccion[0]], 'Tipo de delito': [selected_crime]})
# Obtener la latitud y longitud de la entidad y agregarla al DataFrame
lat_lon = crimes_df[crimes_df['Entidad'] == selected_state][['lat', 'lon']].iloc[0]
prediction['lat'] = lat_lon['lat']
prediction['lon'] = lat_lon['lon']

# Generar un mapa con la cantidad de delitos por entidad, con el total de delitos y la predicción
st.markdown("## Mapa de delitos por entidad")
prediction['Scaled Radius'] = scaler.fit_transform(prediction[['Total']]) * 50000  # Ajusta el factor según sea necesario
layer = pdk.Layer('ScatterplotLayer', data=prediction, get_position='[lon, lat]', radius='Scaled Radius', elevation_scale=4, elevation_range=[0, 1000], pickable=True, extruded=True, get_radius='Total * 20', get_fill_color=[255, 0, 0, 160],)
view_state = pdk.ViewState(latitude=lat_lon['lat'], longitude=lat_lon['lon'], zoom=4, bearing=0, pitch=0)
r = pdk.Deck(layers=[layer], initial_view_state=view_state)
st.pydeck_chart(r)

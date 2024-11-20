import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils import load_data, total_data
import pydeck as pdk

# Cargar el DataFrame
crimes_df = load_data()

######## JUSTIFICACIÓN DE USO DE MODELO DE REGRESIÓN LINEAL ########
texto_modelo_regresion = """
# **Modelo de Regresión Lineal:**

El modelo de regresión lineal es un enfoque de modelado predictivo supervisado que se utiliza para predecir el valor de una variable dependiente basada en una o más variables independientes.

- **Justificación:** Útil para entender la relación entre variables (por ejemplo, la clave de entidad y el total anual de delitos) y hacer predicciones basadas en esa relación.
- **Aplicación:** Predicción del total anual de delitos en función de la clave de entidad.
- **Ventajas:** Fácil de interpretar y aplicar, proporciona una relación matemática clara entre las variables.
"""
st.markdown(texto_modelo_regresion)

st.markdown("## Muestra de datos de crimen")
st.write("Las primeras filas del DataFrame son:")
st.dataframe(crimes_df.head())
#get all crimes and give them a key
# Obtener los valores únicos de 'Tipo de delito'


######## CREACIÓN DE COLUMNA 'Total_Delitos' ########
st.markdown("## Creación de columna 'Total_Delitos'")
crimes_df=total_data()
selected_crimes = crimes_df['Tipo de delito'].unique()

# Crear un diccionario que mapea cada tipo de delito a un índice
crimes_dict = {crime: i for i, crime in enumerate(selected_crimes)}
inverse_crimes_dict = {i: crime for crime, i in crimes_dict.items()}
# Reemplazar los valores de 'Tipo de delito' con sus índices correspondientes
crimes_df['Tipo de delito'] = crimes_df['Tipo de delito'].map(crimes_dict)
st.write(crimes_df[['Año', 'Entidad', 'Clave_Ent', 'Total']])
#make a entidades dataframe with its clave_ent to after make a dictionary without index
entidades = crimes_df[['Entidad', 'Clave_Ent']]
entidades.drop_duplicates(inplace=True)
entidades = entidades.set_index('Entidad')
entidades = entidades.to_dict()

######## SELECCIÓN DE VARIABLES ########
X = crimes_df[['Clave_Ent', 'Año','Tipo de delito']]
y = crimes_df['Total']

######## VISUALIZACIÓN DE DATOS ########
st.markdown("## Total de delitos por entidad")
plt.figure(figsize=(12, 6))
entidad_totales = crimes_df.groupby('Entidad')['Total'].sum().sort_values(ascending=False)
sns.barplot(x=entidad_totales.index, y=entidad_totales.values, palette="viridis")
plt.xticks(rotation=90)  # Rotar nombres si son largos
plt.title("Total de delitos por entidad")
plt.xlabel("Entidad")
plt.ylabel("Total de delitos")
st.pyplot(plt.gcf())
  

######## DIVISIÓN DE DATOS ########
st.markdown("## División de los datos en conjunto de entrenamiento y prueba")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

######## ENTRENAMIENTO DEL MODELO ########
st.markdown("## Entrenamiento del modelo de regresión lineal")
model = LinearRegression()
model.fit(X_train, y_train)

######## PREDICCIONES ########
st.markdown("## Predicciones y visualización")
y_pred = model.predict(X_test)


######## EVALUACIÓN DEL MODELO ########
st.markdown("## Evaluación del modelo")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
st.write(f"Pendiente (β1): {model.coef_[0]}")
st.write(f"Intersección (β0): {model.intercept_}")
st.write(f"MSE: {mse}")
st.write(f"RMSE: {rmse}")

######## PREDICCIÓN PARA UNA CLAVE DE ENTIDAD ESPECÍFICA ########
st.markdown("## Predicción para una clave de entidad específica")
estados = crimes_df["Entidad"].unique()
selected_state = st.selectbox("Selecciona una entidad para predecir el total anual de delitos:", options=estados)
# get only the crimes names commited in the selected state in unique values
crimes = crimes_df[crimes_df['Entidad'] == selected_state]['Tipo de delito'].unique()
crimes_names = [inverse_crimes_dict[crime] for crime in crimes]
selected_crime = st.selectbox("Selecciona el tipo de delito a visualizar:", options=crimes_names)
selected_crime_key = crimes_dict[selected_crime]
clave_entidad = entidades['Clave_Ent'][selected_state]
year=st.text_input("Selecciona el año a predecir:", "2024")
if year.isdigit() == False:
    st.error("Por favor ingrese un año válido")
else:
    prediccion = model.predict([[clave_entidad, int(year), selected_crime_key]])
st.write(f"La predicción para la clave de entidad {clave_entidad} para el año {year} es de {prediccion[0]:.2f} aproximadamente.")
#generar un df con caso base y predicción
prediction = pd.DataFrame({'Clave_Ent': [clave_entidad], 'Año': [int(year)], 'Key de Tipo de delito': [selected_crime_key], 'Total': [prediccion[0]], 'Tipo de delito': [selected_crime]})
#agregar caso base a df
st.write(prediction)

#generar un mapa con la cantidad de delitos por entidad, con el total de delitos y la predicción
st.markdown("## Mapa de delitos por entidad")

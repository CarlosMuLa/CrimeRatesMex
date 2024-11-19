import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar el DataFrame
crimes_df = pd.read_csv("./Guardados/clean_crime_rate.csv")

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

######## CREACIÓN DE COLUMNA 'Total_Delitos' ########
st.markdown("## Creación de columna 'Total_Delitos'")
crimes_df['Total_Delitos'] = crimes_df[['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']].sum(axis=1)
st.write(crimes_df[['Año', 'Entidad', 'Clave_Ent', 'Total_Delitos']])

######## SELECCIÓN DE VARIABLES ########
X = crimes_df[['Clave_Ent']]
y = crimes_df['Total_Delitos']

######## VISUALIZACIÓN DE DATOS ########
st.markdown("## Visualización de datos")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Clave_Ent', y='Total_Delitos', data=crimes_df)
plt.title('Relación entre Clave de Entidad y Total Anual de Delitos')
plt.xlabel('Clave de la Entidad')
plt.ylabel('Total Anual de Delitos')
st.pyplot(plt.gcf())  # Asegúrate de usar plt.gcf() para obtener la figura actual

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

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Datos de prueba', color='b', alpha=0.7)
plt.plot(X_test, y_pred, label='Regresión lineal', color='r', linewidth=2)
plt.legend()
plt.title('Regresión lineal entre Clave de Entidad y Total Anual de Delitos')
plt.xlabel('Clave de Entidad')
plt.ylabel('Total Anual de Delitos')
st.pyplot(plt.gcf())  # Asegúrate de usar plt.gcf() para obtener la figura actual

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
clave_entidad = st.sidebar.number_input("Introduce la clave de entidad para predecir el total de delitos", min_value=1, max_value=32, step=1, value=1)
prediccion = model.predict(np.array([[clave_entidad]]))
st.write(f"La predicción para la clave de entidad {clave_entidad} es {prediccion[0]:.2f} delitos anuales.")

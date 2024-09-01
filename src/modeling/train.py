# Este script entrena un modelo de Machine Learning y lo guarda en la carpeta models/.
# Ejemplo: Entrenamiento de un modelo de regresión lineal para predecir precios de viviendas.
# Este script entrena un modelo de Machine Learning utilizando los datos preprocesados.
# Incluye la separación de datos en conjuntos de entrenamiento y prueba, el entrenamiento del modelo, 
# y la evaluación del desempeño del modelo.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_model():
    df = pd.read_csv('data/processed/diabetes_data_clean.csv')
    
    # Separar características y variable objetivo
    X = df.drop('target', axis=1)
    y = df['target']

    # Separar en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Guardar el modelo entrenado
    joblib.dump(model, 'models/linear_regression_model.pkl')
    print("Modelo guardado en models/linear_regression_model.pkl")

if __name__ == "__main__":
    train_model()

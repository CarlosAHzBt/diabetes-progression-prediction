import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))) # Por pedos en la importacion a raiz de los init

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from src.services import plot_predictions_vs_real   # Importar desde services, usando __init__.py


def evaluate_model():
    # Cargar los datos preprocesados
    df = pd.read_csv('data/processed/diabetes_data_clean.csv')
    real_values = df['target']
    
    # Cargar el modelo entrenado
    model = joblib.load('models/linear_regression_model.pkl')
    
    # Realizar predicciones
    predictions = model.predict(df.drop('target', axis=1))
    
    # Evaluar el modelo
    mse = mean_squared_error(real_values, predictions)
    r2 = r2_score(real_values, predictions)
    print(f"MSE: {mse}")
    print(f"R^2: {r2}")
    
    # Visualizar las predicciones vs los valores reales
    plot_predictions_vs_real(real_values, predictions)

if __name__ == "__main__":
    evaluate_model()
# Este script realiza el preprocesamiento y la ingeniería de características en los datos.
# Incluye la limpieza de datos, la eliminación de valores nulos, la transformación de características, etc.


import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print("Datos originales:")
    print(df.head())

    # Ejemplo de preprocesamiento: normalización de los datos
    df_normalized = (df - df.min()) / (df.max() - df.min())
    
    # Guardar los datos preprocesados en la carpeta data/processed/
    df_normalized.to_csv('data/processed/diabetes_data_clean.csv', index=False)
    print("Datos preprocesados guardados en data/processed/diabetes_data_clean.csv")

if __name__ == "__main__":
    preprocess_data('data/raw/diabetes_data.csv')

# Este script carga datos desde fuentes externas y los guarda en la carpeta data/raw/.
# Ejemplo: Descargar un conjunto de datos de una API y guardarlo como CSV.
# Este script se encarga de la carga y guardado de datos desde y hacia diferentes fuentes.
# Incluye funciones para cargar datos crudos, transformarlos y guardarlos en la estructura del proyecto.


from sklearn.datasets import load_diabetes
import pandas as pd

def load_and_save_diabetes_data():
    # Cargar el conjunto de datos de diabetes desde sklearn
    diabetes = load_diabetes()
    
    # Convertir los datos en un DataFrame de pandas
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    
    # Guardar el DataFrame en la carpeta data/raw/
    df.to_csv('data/raw/diabetes_data.csv', index=False)
    print("Datos guardados en data/raw/diabetes_data.csv")

if __name__ == "__main__":
    load_and_save_diabetes_data()

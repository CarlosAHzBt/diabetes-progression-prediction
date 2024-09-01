#En el archivo plots se encuentran las funciones que generan los gráficos de los datos.
#Estas funciones se utilizan para visualizar los datos y los resultados de los modelos.

import matplotlib.pyplot as plt

def plot_predictions_vs_real(real_values, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(real_values, predictions, alpha=0.5)
    plt.plot([min(real_values), max(real_values)], [min(real_values), max(real_values)], color='red', linestyle='--')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Comparación de Predicciones vs Valores Reales')
    plt.show()

# Se pueden añadir más funciones de visualización aquí.

# src/services/__init__.py

from .dataset import load_and_save_diabetes_data
from .features import preprocess_data
from .plots import plot_predictions_vs_real

__all__ = ['load_and_save_diabetes_data', 'preprocess_data', 'plot_predictions_vs_real']


# Esto convierte la carpeta en un paquete, permitiendo importaciones desde ella.

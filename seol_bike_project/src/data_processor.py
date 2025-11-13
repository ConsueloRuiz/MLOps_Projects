# src/data_processor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

class DataProcessor:
    """
    Clase que encapsula la manipulación, limpieza y preprocesamiento de datos.
    Aplica DVC: Los datos procesados se guardan para versionado.
    """
    def __init__(self, raw_path: str, processed_path: str):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.target_col = 'rented_bike_count'
        self.feature_cols = [
            'hour', 'temperaturec', 'humidity', 'wind_speed_ms', 
            'visibility_10m', 'dew_point_temperaturec', 
            'solar_radiation_mjm2', 'rainfallmm', 'snowfallcm', 'mixed_type_col'
        ]


    def save_processed_data(self, X_scaled: np.ndarray, y: np.ndarray, scaler: StandardScaler):
        """Guarda los datos procesados y el scaler para su versionado (DVC)."""
        data_to_save = {
            'X_scaled': X_scaled,
            'y': y,
            'scaler': scaler
        }
        with open(self.processed_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"Datos preprocesados y scaler guardados en: {self.processed_path}")
        # En la terminal, este paso se seguiría con:
        # dvc add data/processed/features.pkl
        # dvc commit -m "Datos procesados con limpieza vX.X"
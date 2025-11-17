# tests/test_data_processor2.py
import pytest
import pandas as pd
from src.data_processor import DataProcessor
from src.data_loader import DataLoader
import os

# Simular un archivo de datos simple para la prueba
@pytest.fixture
def mock_data_path(tmp_path):
    # Crea un CSV simple en un directorio temporal

    data = {
        'date':['1/12/2017', '1/12/2017'],
        'Rented Bike Count': [100.0, 200.0],
        'hour':['11:00', '12:00'],
        'TemperatureC': [5.5, 6.5],
        'humidity': [90, 30],
        'visibility10m': [1, 1],
        'windspeedms':[1,1],
        'dewpointtemperaturec': [-19.8, -22.4],
        'solarradiationmjm2': [0.01, 0.23],
        'rainfallmm': [0, 0],
        'snowfallcm': [0, 0],
        'season': ['Winter', 'Fall'],
        'holiday': ['No Holiday', 'No Holiday'],
        'Functioning Day': ['YES', 'yes'],
        'mixed_type_col': ['1.0', 'bad'] # Dato sucio
    }
    df = pd.DataFrame(data)
    test_file = tmp_path / "test_raw_data.csv"
    df.to_csv(test_file, index=False)
    return str(test_file)

def test_data_cleaning(mock_data_path):
    """Prueba unitaria de la función de limpieza de datos."""
    processor = DataLoader(raw_data_path=mock_data_path, processed_path="dummy.pkl")
    #processor= DataLoader("data/raw/seoul_bike_sharing_modified.csv")
    #df_cleaned = processor.explore_and_clean()
    df_cleaned = processor.load_and_clean_data()

    # 1. Prueba de que las columnas se renombraron correctamente a minúsculas y sin caracteres especiales
    assert 'temperaturec' in df_cleaned.columns
    # 2. Prueba de que los valores nulos (mixed_type_col = 'bad') se eliminaron
    #assert len(df_cleaned) == 1
    # 3. Prueba de que el tipo de la columna objetivo es correcto
    assert df_cleaned['rentedbikecount'].dtype == float
# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

class DataLoader:
    """
    Clase para manejar la carga y preprocesamiento de los datos de bicicletas.
    Aplica POO para encapsular la lógica de datos.
    """
    def __init__(self, raw_data_path: str,processed_path: str):
        """Inicializa con la ruta del archivo de datos brutos."""
        self.raw_data_path = raw_data_path
        self.processed_path = processed_path
        self.target_col = 'rentedbikecount'
        self.feature_cols = [
            'hour', 'temperaturec', 'humidity', 'wind_speed_ms', 
            'visibility_10m', 'dew_point_temperaturec', 
            'solar_radiation_mjm2', 'rainfallmm', 'snowfallcm', 'mixed_type_col'
        ]


    def load_and_clean_data(self) -> pd.DataFrame:
        """Carga los datos brutos y realiza la limpieza básica."""
        print("Cargando y limpiando datos...")
        print(self.raw_data_path)
        try:
            df = pd.read_csv(self.raw_data_path, encoding='latin-1', na_values=[' '])
        
        except Exception as e:
            raise FileNotFoundError(f"Error al cargar el archivo: {e}")

        # Refactorización: Limpieza y Estandarización de Nombres de Columnas
        df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True).str.lower().str.replace(' ', '_')

        # Eliminar filas con nulos después de la limpieza
        #df = df.dropna()
        # 1. Limpieza y preparación (ej. imputación simple y one-hot encoding)
        var_int = ['rentedbikecount','hour','humidity','visibility10m']
        var_float = ['temperaturec', 'windspeedms', 'dewpointtemperaturec', 'solarradiationmjm2',
                    'rainfallmm', 'snowfallcm']

    # Convertir a Fecha
        df['date'] = df['date'].str.strip()
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)

        # Convertir a numerico
        for col in var_int:
            df[col] = pd.to_numeric(df[col], errors= 'coerce', downcast='integer')

        # Convertir a float
        for col in var_float:
            df[col] = pd.to_numeric(df[col], errors= 'coerce')

        # Eliminacion de filas duplicadas
        df.drop_duplicates()

        # Se elimina la columna 'mixed_type_col' por no aportar data importante y tener gran cantidad de nulls
        df.drop(columns = ['mixed_type_col'], inplace = True)

        # Eliminamos los registros nulos de la variable objetivo, asi como los datos con fecha nula.
        # IMPORTANTE: Recordad que el porcentaje de valores faltantes es poco, entonces se toma la decision de elimnar los registros.

        df.dropna(subset=['rentedbikecount', 'date'], inplace = True)

        return df
    
    def del_NA(self, df: pd.DataFrame, threshold_drop: int) -> pd.DataFrame:

        df_copy = df.copy()

        # 1. Imputación de valores faltantes
        for col in df_copy.columns:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            else:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

        return df_copy

      
    def del_outliers(self, df: pd.DataFrame, col:object) -> pd.DataFrame:
        df_cleaned = df.copy()
        for col in col:
            Q1 = df_cleaned[col].quantile(0.25)  # Primer cuartil
            Q3 = df_cleaned[col].quantile(0.75)  # Tercer cuartil
            IQR = Q3 - Q1                        

            # Filtramos filas que estén dentro de 1.5 * IQR
            df_cleaned = df_cleaned[
                (df_cleaned[col] >= Q1 - 1.5 * IQR) & 
                (df_cleaned[col] <= Q3 + 1.5 * IQR)
            ]
        return df_cleaned

    def imp_season(self, df: pd.DataFrame,scale_numeric=True) -> pd.DataFrame:
        # Imputar Striings Nan por un valor nulo verdadero.
        df['seasons'] = df['seasons'].replace("Nan", np.nan)
        df['holiday'] = df['holiday'].replace("Nan", np.nan)
        df['functioningday'] = df['functioningday'].replace("Nan", np.nan)

        # Modificar textos y agrupar en una sola categoria, ademas imputar nulos con la moda.
        df['seasons'] = df['seasons'].str.title().str.strip().fillna(df['seasons'].mode()[0])
        df['holiday'] = df['holiday'].str.title().str.strip().fillna(df['holiday'].mode()[0])
        df['functioningday'] = df['functioningday'].str.title().str.strip().fillna(df['functioningday'].mode()[0])
        return df

    def preprocess_dataframe(self, df: pd.DataFrame,scale_numeric=True)-> pd.DataFrame:

        df_copy = df.copy()

        # 3. Codificación de categóricas
        cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns
        df_copy = pd.get_dummies(df_copy, columns=cat_cols, drop_first=True)

        # 4. Escalado de numéricas
        if scale_numeric:
            num_cols = df_copy.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            df_copy[num_cols] = scaler.fit_transform(df_copy[num_cols])

        # 2. Selección de features para el modelo
        features = [
            'rentedbikecount','hour', 'temperaturec', 'humidity', 'windspeedms',
            'visibility10m', 'dewpointtemperaturec', 'solarradiationmjm2', 
            'rainfallmm', 'snowfallcm','season_Spring', 'season_Summer', 'season_Winter', 'holiday_No Holiday', 'functioning Day_Yes'
        ]

        df_copy = df.filter(items=features).dropna()
        return df_copy

    def preprocess_data(self, df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """Realiza ingeniería de características y división de conjuntos."""
        
        # Ingeniería de Características: Codificación One-Hot para 'Seasons'
        #df = pd.get_dummies(df, columns=['seasons'], drop_first=True)
        
        # Selección de características (ejemplo simplificado)
        features = [
            'rentedbikecount', 'hour', 'temperaturec', 'humidity', 'windspeedms',
            'visibility10m', 'dewpointtemperaturec', 'solarradiationmjm2', 
            'rainfallmm', 'snowfallcm','season_Spring', 'season_Summer', 'season_Winter', 'holiday_No Holiday', 'functioning Day_Yes'
        ]

        
        # Asegurar que solo existen las columnas seleccionadas y la columna objetivo
        df = df[[col for col in features if col in df.columns] + [target_col]]
  
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        # Estandarización de características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

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


        # División en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        print("Datos listos para el entrenamiento.")
        return X_train, X_test, y_train, y_test, scaler
# src/pipeline.py
import os
import mlflow
from data_loader import DataLoader
from model_trainer import ModelTrainer
from data_processor import DataProcessor
from mlflow.models import infer_signature


# Nombre del archivo de datos subido por el usuario
DATA_PATH = "data/raw/seoul_bike_sharing_modified.csv"
PROCESSED_DATA_PATH = "data/processed/features.pkl"
TARGET_COLUMN = 'rentedbikecount'
ALPHAS = [0.1, 1.0, 10.0] # Diferentes parámetros para experimentar

# Initializing MLFlow Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
os.environ["MLFLOW_TRACKING_URI"] = "file://" + os.path.abspath("mlruns")
#mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

def load_or_process_data(processor: DataProcessor):
    """Carga datos preprocesados si existen, si no, los procesa."""
    if os.path.exists(PROCESSED_DATA_PATH):
        print("Cargando datos preprocesados existentes (trazabilidad DVC).")
        with open(PROCESSED_DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        return data['X_scaled'], data['y'], data['scaler']
    else:
        print("Ejecutando pipeline de preprocesamiento.")
        df_cleaned = processor.explore_and_clean()
        X, y, scaler = processor.preprocess(df_cleaned)
        processor.save_processed_data(X, y, scaler)
        return X, y, scaler


def main():
    """
    Función principal que ejecuta el pipeline completo de ML.
    """
    try:
        # 1. Carga y Preprocesamiento
        print("Carga y Preprocesamiento")
        data_loader = DataLoader(DATA_PATH)
        # 1. Manipulación y Preparación de Datos (DataProcessor)
        print("Carga y Preprocesamiento DataProcessor")
        processor = DataProcessor(DATA_PATH, PROCESSED_DATA_PATH)
        print("load_or_process_data")
        print("se actualizo")
        X, y, scaler = load_or_process_data(processor)
        # 2. Datos limpios
        print("Datos Limpios")
        df_cleaned = data_loader.load_and_clean_data()
        print("Elimina NA")
        # 3. Elimina NA
        df_NA =data_loader.del_NA(df_cleaned,threshold_drop=0.4)
        print("datos con outliers")
        # 4. Datos con outliers
        print(df_NA.select_dtypes(include=['float64', 'int64']).columns)
        df_outliers = data_loader.del_outliers(df_NA, df_NA.select_dtypes(include=['float64', 'int64']).columns)
        print("imputar sesiones")
        # 5. Imputar sesiones
        df_season = data_loader.imp_season(df_outliers,scale_numeric=True)
        print("procesa data frame")
        # 6. procesar Dataframe
        df_process = data_loader.preprocess_dataframe(df_season, scale_numeric= True)
       

        # Desempaquetado con el Scaler para guardar como artefacto
        X_train, X_test, y_train, y_test, scaler = data_loader.preprocess_data(
            df=df_process, 
            target_col=TARGET_COLUMN
        )
        #print("ya separo datos")
        # 2. Entrenamiento y Tracking (MLflow)
        trainer = ModelTrainer(experiment_name="Seoul_Bike_Prediction_Ridge")
        #print("ya entro datos")
        # Ejecutar múltiples experimentos con diferentes parámetros (control de versiones)
        for alpha in ALPHAS:
            print(f"\n--- Iniciando experimento con alpha={alpha} ---")
            trainer.train_and_log_model(X_train, X_test, y_train, y_test, alpha, scaler)
            
        print("\nPipeline de ML completado exitosamente.")
        print("Para ver resultados y comparar métricas, ejecuta 'mlflow ui'.")
        print("Para versionar datos procesados, ejecuta 'dvc add data/processed/features.pkl'.")
    
    except FileNotFoundError as e:
        print(f"ERROR: Asegúrate de colocar el archivo CSV en la ruta: {DATA_PATH}")
        print(e)
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    # La mejor práctica es envolver la ejecución en 'main()'
    main()
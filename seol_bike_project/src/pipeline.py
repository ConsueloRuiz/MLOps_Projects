# src/pipeline.py
import os
import mlflow
from data_loader import DataLoader
from model_trainer import ModelTrainer
from data_processor import DataProcessor
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
import pickle


# Nombre del archivo de datos subido por el usuario
DATA_PATH = "data/raw/seoul_bike_sharing_modified.csv"
PROCESSED_DATA_PATH = "data/processed/features.pkl"
TARGET_COLUMN = 'rentedbikecount'

#ALPHAS = [0.1, 1.0, 10.0] # Diferentes parámetros para experimentar
params = [
        {"alpha": 0.1, "run_name": "Ridge_v1_Optimized"},
        {"alpha": 1.0, "run_name": "Ridge_v2_Baseline"},
        {"alpha": 10.0, "run_name": "Ridge_v3_StrongReg"}
    ]


# Rutas donde los scripts de Serving y Drift esperan los archivos
PROD_MODEL_PATH = "models/production_model.pkl"
PROD_SCALER_PATH = "models/production_scaler.pkl"
BASELINE_DATA_PATH = "data/processed/test_data_baseline.pkl"

# Initializing MLFlow Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
os.environ["MLFLOW_TRACKING_URI"] = "file://" + os.path.abspath("mlruns")
<<<<<<< HEAD
=======
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

>>>>>>> 3cb215f2e561860f8b6f1d3bcbd458c434dd991e

def main():
    """
    Función principal que ejecuta el pipeline completo de ML.
    """
    try:
        # 1. Carga y Preprocesamiento
        print("Carga y Preprocesamiento")
<<<<<<< HEAD
        data_loader = DataLoader(raw_data_path=DATA_PATH,processed_path=PROCESSED_DATA_PATH)
=======
        data_loader = DataLoader(DATA_PATH)
        # 1. Manipulación y Preparación de Datos (DataProcessor)
        print("Carga y Preprocesamiento DataProcessor")
        processor = DataProcessor(DATA_PATH, PROCESSED_DATA_PATH)
        print("load_or_process_data")
        print("se actualizo")
        X, y, scaler = load_or_process_data(processor)
>>>>>>> 3cb215f2e561860f8b6f1d3bcbd458c434dd991e
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

        """Guarda los datos procesados y el scaler para su versionado (DVC)."""
        data_to_save = {
            'X_scaled': X_train,
            'y': y_train,
            'scaler': scaler
        }
        with open(PROCESSED_DATA_PATH, 'wb') as f:
            pickle.dump(data_to_save, f)
            

        # 2. Entrenamiento y Tracking (MLflow)
        trainer = ModelTrainer(experiment_name="Seoul_Bike_Prediction_Ridge")
        # Ejecutar múltiples experimentos con diferentes parámetros (control de versiones)
        best_rmse = float('inf')
        best_model = None
        best_run_name = ""


        #for alpha in ALPHAS:
        for i, p in enumerate(params):
            print(f"\n--- Iniciando experimento con alpha={p["alpha"]} ---")
            model, rmse= trainer.train_and_log_model(X_train, X_test, y_train, y_test, p["alpha"], scaler)

            # Lógica de PROMOción: ¿Es este el mejor modelo?
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_run_name = p["run_name"]
            
        print("\nPipeline de ML completado exitosamente.")
        print("Para ver resultados y comparar métricas, ejecuta 'mlflow ui'.")
        print("Para versionar datos procesados, ejecuta 'dvc add data/processed/features.pkl'.")
<<<<<<< HEAD

    # --- PASO DE SIMULACIÓN DE PROMOCIÓN DE ARTEFACTOS ---
        if best_model:
            print(f"\n✨ Promoviendo '{best_run_name}' (RMSE: {best_rmse:.2f}) a Producción...")
            
            # a) Guardar el Modelo de Producción
            with open(PROD_MODEL_PATH, 'wb') as f:
                pickle.dump(best_model, f)

            # b) Guardar el Scaler de Producción
            with open(PROD_SCALER_PATH, 'wb') as f:
                pickle.dump(scaler, f)
                
            # c) Guardar los Datos de Prueba (Línea Base para Drift Detection)
            # Esto contiene los datos originales contra los que se comparará la performance futura.


            baseline_data = {
                'X_test_scaled': X_test, # Ya que X_test ya está escalado si lo pasamos desde aquí
                'y_test': y_test
            }
            with open(BASELINE_DATA_PATH, 'wb') as f:
                pickle.dump(baseline_data, f)
                
            print("Archivos de Serving y Drift generados exitosamente.")

=======
>>>>>>> 3cb215f2e561860f8b6f1d3bcbd458c434dd991e
    
    except FileNotFoundError as e:
        print(f"ERROR: Asegúrate de colocar el archivo CSV en la ruta: {DATA_PATH}")
        print(e)
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    # La mejor práctica es envolver la ejecución en 'main()'
    main()
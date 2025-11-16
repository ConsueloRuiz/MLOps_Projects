# src/pipeline2.py
from data_processor import DataProcessor
from model_trainer import ModelTrainer
import pickle
import os

# Configuración de rutas (siguiendo Cookiecutter)
RAW_DATA_PATH = "data/raw/seoul_bike_sharing_modified.csv"
PROCESSED_DATA_PATH = "data/processed/features.pkl"

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
    """Ejecuta el pipeline completo de ML."""
    
    # 1. Manipulación y Preparación de Datos (DataProcessor)
    processor = DataProcessor(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    X, y, scaler = load_or_process_data(processor)

    # 2. Construir, ajustar y evaluar Modelos (ModelTrainer)
    trainer = ModelTrainer(experiment_name="Bike_Rental_Prediction")
    
    # Iteración para 'Ajustar' y 'Comparar' diferentes modelos/parámetros
    params = [
        {"alpha": 0.1, "run_name": "Ridge_v1_Optimized"},
        {"alpha": 1.0, "run_name": "Ridge_v2_Baseline"},
        {"alpha": 10.0, "run_name": "Ridge_v3_StrongReg"}
    ]
    
    best_rmse = float('inf')
    best_run_name = ""

    for i, p in enumerate(params):
        rmse = trainer.train_and_log_model(
            X=X, 
            y=y, 
            alpha=p["alpha"], 
            scaler=scaler, 
            run_name=p["run_name"]
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_run_name = p["run_name"]

    print(f"\n✅ Pipeline Completado. Mejor Modelo: {best_run_name} (RMSE: {best_rmse:.2f})")
    print("Para ver resultados y comparar métricas, ejecuta 'mlflow ui'.")
    print("Para versionar datos procesados, ejecuta 'dvc add data/processed/features.pkl'.")

if __name__ == "__main__":
    main()
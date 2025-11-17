# drift_analysis/data_drift_detector.py
import pandas as pd
import numpy as np
import pickle
from evidently.report import Report
from evidently.metric_preset import data_drift,regression_performance

# --- Rutas de Artefactos ---
MODEL_PATH = "models/production_model.pkl"
SCALER_PATH = "models/production_scaler.pkl"
BASELINE_DATA_PATH = "data/processed/test_data_baseline.pkl" # Datos de prueba originales

def load_for_drift():
    """Carga los artefactos de producción y los datos de la línea base."""
    print("entro a load_for_drift")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(BASELINE_DATA_PATH, 'rb') as f:
        # Asume que este archivo contiene un diccionario con 'X_test_scaled' y 'y_test'
        baseline_data = pickle.load(f) 
    # Crear un DataFrame de línea base con predicciones
    feature_names = [
        'hour', 'temperaturec', 'humidity', 'windspeedms',
       'visibility10m', 'dewpointtemperaturec', 'solarradiationmjm2',
       'rainfallmm', 'snowfallcm'    ]

    baseline_df = pd.DataFrame(baseline_data['X_test_scaled'], columns=feature_names)
    baseline_df['target'] = baseline_data['y_test'][:,0]
    baseline_df['prediction'] = model.predict(baseline_data['X_test_scaled'])[:,0]
    return model, scaler, baseline_df

def generate_drift_data(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un set de datos sintéticos que simulan un cambio (drift) en las características.
    Simulación: Aumento de la temperatura (+5C) y mayor velocidad del viento (x 1.5).
    """
    print("Entro a generate_drift_data")
    drift_df = baseline_df.copy().drop(columns=['prediction'])
    
    # 1. Data Drift: Cambios en la distribución
    drift_df['temperaturec'] = drift_df['temperaturec'] + 5 
    drift_df['windspeedms'] = drift_df['windspeedms'] * 1.5

    return drift_df

def evaluate_drift(model, scaler, baseline_df: pd.DataFrame):
    """
    Ejecuta los reportes de Data Drift y Performance Drift.
    """
    print("--- 1. Generando Datos de Producción Sintéticos ---")
    current_df_no_pred = generate_drift_data(baseline_df)
    print("E1")
    # Preprocesar y predecir sobre el set con drift (simulación de producción)
    X_current_scaled = scaler.transform(current_df_no_pred.drop(columns=['target']).values)
    print("E2")
    current_df = current_df_no_pred.copy()
    current_df['prediction'] = model.predict(X_current_scaled)[:,0]
    # Mantener la columna 'target' para evaluar el Performance Drift
    print("E3")
    # --- 2. Reporte de Data Drift (Cambio de Datos) ---
    print("\n--- 2. Ejecutando Reporte de Data Drift (Evidently) ---")


    data_drift_report = Report(metrics=[
        data_drift.DataDriftPreset(
            # Definir un umbral de significancia del 5%
            drift_share=0.05
        ),
    ])
    print("E4")
    data_drift_report.run(
        reference_data=baseline_df, 
        current_data=current_df
    )
    data_drift_report.save_html("drift_analysis/data_drift_report.html")
    print("E5")
    # --- 3. Reporte de Performance Drift (Pérdida de Rendimiento) ---
    print("\n--- 3. Ejecutando Reporte de Pérdida de Performance (Evidently) ---")
    perf_report = Report(metrics=[
       regression_performance.RegressionPreset(
            # Se usa el error cuadrático medio (RMSE)
        )
    ])
    print("E5")
    perf_report.run(
        reference_data=baseline_df, 
        current_data=current_df
    )
    print("E6")
    #perf_report.save_html("drift_analysis/performance_drift_report.html")
    print("\n✅ Análisis de Drift completado. Reportes HTML generados en /drift_analysis.")
    print("E7")
if __name__ == "__main__":
    try:
        model, scaler, baseline_df = load_for_drift()
        evaluate_drift(model, scaler, baseline_df)

        
    except FileNotFoundError as e:
        print(f"ERROR: Archivos no encontrados para la detección de drift. Asegúrate de que los archivos 'production_model.pkl', 'production_scaler.pkl' y 'test_data_baseline.pkl' existan. Detalle: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la detección de drift: {e}")
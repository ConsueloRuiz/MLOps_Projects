# api/main.py
from fastapi import FastAPI, HTTPException
from typing import Dict
import pickle
import numpy as np
import pandas as pd

# Importar modelos de datos y asumir rutas de artefactos
from api.data_models import BikeFeatures, PredictionResponse
MODEL_PATH = "models/production_model.pkl"
SCALER_PATH = "models/production_scaler.pkl" 
MODEL_VERSION = "Ridge_v1_Optimized"

MODEL = None
SCALER = None

def load_artifacts():
    """Carga el modelo y el scaler serializados desde la carpeta models/"""
    global MODEL, SCALER
    try:
        with open(MODEL_PATH, 'rb') as f:
            MODEL = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            SCALER = pickle.load(f)
        print(f"Artefactos cargados. Modelo: {MODEL_VERSION}")
    except FileNotFoundError as e:
        print(f"Error fatal: No se encontró el archivo del modelo o scaler: {e}")
        # En un sistema de producción, esto forzaría el fallo del contenedor
        raise RuntimeError("Artefactos de ML no encontrados. El servicio no puede iniciar.")

app = FastAPI(
    title="Bike Rental Prediction API (FastAPI)",
    description="Servicio de Serving para el modelo de predicción de bicicletas alquiladas."
)

@app.on_event("startup")
async def startup_event():
    """Llamar a la carga de artefactos al iniciar el servidor."""
    load_artifacts()

# Endpoint de Predicción
@app.post("/predict", response_model=PredictionResponse)
def predict(features: BikeFeatures):
    """Realiza la predicción, escala los datos de entrada y devuelve el resultado."""
    
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
        
    # 1. Preparar datos en el formato correcto (DataFrame)
    input_data = features.dict(by_alias=True)
    
    # Definir el orden de las features (CRUCIAL para el escalado y el modelo)
    feature_order = [
        'hour', 'temperaturec', 'humidity', 'wind_speed_ms', 
        'visibility_10m', 'dew_point_temperaturec', 'solar_radiation_mjm2', 
        'rainfallmm', 'snowfallcm', 'mixed_type_col',
        'seasons_Winter', 'seasons_Spring', 'seasons_Autumn'
    ]
    
    X_df = pd.DataFrame([input_data])[feature_order]

    # 2. Preprocesamiento (Escalado)
    X_scaled = SCALER.transform(X_df.values)

    # 3. Predicción
    prediction = MODEL.predict(X_scaled)[0]
    
    # 4. Retorno (Asegurando que la predicción no sea negativa)
    prediction = max(0, prediction)

    return PredictionResponse(
        prediction=prediction, 
        model_version=MODEL_VERSION
    )

# Endpoint de Salud
@app.get("/health", response_model=Dict[str, str])
def health_check():
    """Verifica si la API está viva y si el modelo está cargado."""
    return {"status": "ok", "model_version": MODEL_VERSION}
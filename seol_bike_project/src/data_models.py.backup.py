# api/data_models.py
from pydantic import BaseModel, Field
from typing import Optional

class BikeFeatures(BaseModel):
    """Schema para las características de la solicitud de predicción."""
    # Nota: Los nombres de campo deben coincidir con los features del modelo
    hour: int = Field(..., ge=0, le=23, description="Hora del día (0-23)")
    temperaturec: float = Field(..., description="Temperatura en °C")
    humidity: float = Field(..., ge=0, le=100, description="Humedad en %")
    wind_speed_ms: float = Field(..., ge=0, description="Velocidad del viento (m/s)")
    visibility_10m: float = Field(..., ge=0, description="Visibilidad (10m)")
    dew_point_temperaturec: float = Field(..., description="Temperatura de punto de rocío")
    solar_radiation_mjm2: float = Field(..., ge=0, description="Radiación solar (MJ/m2)")
    rainfallmm: float = Field(..., ge=0, description="Lluvia (mm)")
    snowfallcm: float = Field(..., ge=0, description="Nieve (cm)")
    mixed_type_col: float = Field(..., description="Columna mixta numérica")
    
    # Dummies de las estaciones (el modelo espera todas)
    seasons_Winter: Optional[int] = Field(0, description="Dummy para Invierno (1 si es Invierno, 0 sino)")
    seasons_Spring: Optional[int] = Field(0, description="Dummy para Primavera (1 si es Primavera, 0 sino)")
    seasons_Autumn: Optional[int] = Field(0, description="Dummy para Otoño (1 si es Otoño, 0 sino)")
    
    class Config:
        schema_extra = {
            "example": {
                "hour": 15, "temperaturec": 25.5, "humidity": 50.0, 
                "wind_speed_ms": 2.5, "visibility_10m": 1800.0, 
                "dew_point_temperaturec": 14.0, "solar_radiation_mjm2": 2.0, 
                "rainfallmm": 0.0, "snowfallcm": 0.0, "mixed_type_col": 800.0,
                "seasons_Spring": 1 
            }
        }

class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Número predicho de bicicletas alquiladas")
    model_version: str = Field(..., description="Versión del modelo usado para la predicción")
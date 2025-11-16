# src/model_trainer2.py
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

class ModelTrainer:
    """
    Clase para el entrenamiento, evaluación, y logging con MLflow.
    """
    def __init__(self, experiment_name: str = "Bike_Rental_Prediction"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)
        
    def train_and_log_model(self, X: np.ndarray, y: np.ndarray, alpha: float, scaler: object, run_name: str):
        """
        Entrena, evalúa, registra métricas, parámetros y artefactos en MLflow.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run(run_name=run_name) as run:
            # 1. Registro de Parámetros
            mlflow.log_param("model_type", "Ridge Regression")
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("data_version", "v1.0") # Se asume una etiqueta de DVC
            
            # 2. Construcción y Entrenamiento del Modelo
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train, y_train)
            
            # 3. Evaluación
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            # 4. Registro de Métricas (Visualizar y Comparar Resultados)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)
            
            print(f"Run {run_name}: RMSE={rmse:.2f}, R2={r2:.2f}")

            # 5. Registro de Modelo (Data Registry) y Artefactos
            # Para el versionado de modelos
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="SeoulBikeRidgeModel"
            )
            
            # Guardar y registrar el Scaler (necesario para la reproducibilidad)
            with open("scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact("scaler.pkl")
            
            # Obtener Run ID para trazabilidad
            mlflow.set_tag("mlflow.runName", run_name)
            
        return rmse
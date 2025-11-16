# src/model_trainer.py
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle # Para guardar el scaler
from sklearn.preprocessing import StandardScaler
import os


class ModelTrainer:
    """
    Clase para el entrenamiento, evaluación y logging con MLflow.
    """
    def __init__(self, experiment_name: str = "Seoul_Bike_Prediction_Ridge"):
        """Inicializa y configura el experimento de MLflow."""
        # Initializing MLFlow Server
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        os.environ["MLFLOW_TRACKING_URI"] = "file://" + os.path.abspath("mlruns")
        #mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)
        print(f"MLflow Experimento: {self.experiment_name} iniciado.")

        
    def train_and_log_model(self, X_train, X_test, y_train, y_test, alpha: float, scaler: StandardScaler):
        """
        Entrena un modelo Ridge, registra métricas, parámetros y artefactos en MLflow.
        """
        with mlflow.start_run(run_name=f"Ridge_alpha_{alpha}"):
            # 1. Registro de Parámetros
            mlflow.log_param("model_type", "Ridge Regression")
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("test_split", 0.2)
            
            # 2. Entrenamiento del Modelo
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train, y_train)
            
            # 3. Predicción y Evaluación
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            # 4. Registro de Métricas
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)
            
            print(f"Modelo entrenado con alpha={alpha}. RMSE: {rmse:.2f}, R2: {r2:.2f}")

            # 5. Registro de Modelo y Artefactos
            # Registro del modelo para Data Registry y versionamiento
            mlflow.sklearn.log_model(
                sk_model=model,
                name="seol_model",
                input_example=X_train,
                #artifact_path="model",
                registered_model_name="SeoulBikeRidgeModel"
            )
            
            # Guardar y registrar el Scaler (necesario para la reproducibilidad)
            scaler_path = "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path)

            print(f"Datos preprocesados y scaler guardados en: {scaler_path}")

            
            # 6. Visualizar Resultados (simulación, ya que el log lo hace MLflow)
            print("\nResultados y modelo registrados en MLflow.")
            return model, rmse
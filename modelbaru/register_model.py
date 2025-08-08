import mlflow
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))

print("Registering model with MLflow...")

# Create or get experiment
experiment_name = "tourism-recommendation"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except mlflow.exceptions.MlflowException:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

mlflow.set_experiment(experiment_name)

# Start MLflow run
with mlflow.start_run():
    mlflow.log_param("model_type", "tourism_recommender")
    mlflow.log_param("version", "1.0.0")
    mlflow.log_metric("dummy_accuracy", 0.85)
    
    # Log model artifacts
    mlflow.log_artifacts("/app/models", "model")
    
    print("Model registered successfully!")
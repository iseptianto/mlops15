# tourism-api/app/app/app.py - FIXED VERSION
import os
import logging
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from mlflow.exceptions import MlflowException

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TourismInput(BaseModel):
    user_id: str

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    status: str

app = FastAPI(
    title="Tourism Recommender API",
    description="API for tourism destination recommendations",
    version="1.0.0"
)

# Global model variable
model = None
model_info = {
    "name": "tourism-recommender-model",
    "alias": "production",
    "status": "not_loaded",
    "error": None
}

@app.on_event("startup")
def load_model():
    """Load tourism recommendation model on startup"""
    global model, model_info
    
    try:
        # Set MLflow configuration
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5001")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logging.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        
        # Set AWS credentials for S3/MinIO
        os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        
        # Load model from MLflow Model Registry
        model_uri = f"models:/{model_info['name']}@{model_info['alias']}"
        logging.info(f"Loading model from: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        
        model_info.update({
            "status": "ready",
            "error": None
        })
        
        logging.info("✅ Tourism recommendation model loaded successfully!")
        
        # Test model with sample data
        test_df = pd.DataFrame({"user_id": ["user_001"]})
        try:
            test_predictions = model.predict(test_df)
            logging.info(f"✅ Model test successful. Sample prediction: {test_predictions}")
        except Exception as e:
            logging.warning(f"⚠️ Model test failed: {e}")
        
    except MlflowException as e:
        error_msg = f"MLflow error: {e}"
        model_info.update({
            "status": "error",
            "error": error_msg
        })
        logging.error(f"❌ {error_msg}")
        
    except Exception as e:
        error_msg = f"General error loading model: {e}"
        model_info.update({
            "status": "error", 
            "error": error_msg
        })
        logging.error(f"❌ {error_msg}")

@app.get("/")
def read_root():
    """Root endpoint with API and model status"""
    return {
        "api_status": "ok",
        "model_name": model_info["name"],
        "model_alias": model_info["alias"], 
        "model_status": model_info["status"],
        "model_error": model_info["error"]
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    is_healthy = model is not None and model_info["status"] == "ready"
    status_code = 200 if is_healthy else 503
    
    return {
        "healthy": is_healthy,
        "model_ready": model is not None,
        "status": model_info["status"]
    }

@app.post("/predict", response_model=List[RecommendationResponse])
def predict(users: List[TourismInput]):
    """Generate tourism recommendations for users"""
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not ready. Status: {model_info['status']}, Error: {model_info.get('error', 'Unknown')}"
        )
    
    if not users:
        raise HTTPException(status_code=400, detail="No users provided")
    
    try:
        # Prepare input DataFrame
        input_df = pd.DataFrame([{"user_id": user.user_id} for user in users])
        logging.info(f"Processing recommendations for {len(users)} users")
        
        # Get predictions from model
        predictions = model.predict(input_df)
        logging.info(f"Model returned {len(predictions)} predictions")
        
        # Format response
        responses = []
        for i, user in enumerate(users):
            try:
                user_prediction = predictions[i] if i < len(predictions) else []
                
                # Handle different prediction formats
                if isinstance(user_prediction, dict):
                    # If model returns dict format
                    responses.append(RecommendationResponse(
                        user_id=user.user_id,
                        recommendations=user_prediction.get("recommendations", []),
                        status=user_prediction.get("status", "success")
                    ))
                elif isinstance(user_prediction, list):
                    # If model returns list of places
                    recommendations = [
                        {"place": place, "score": None} if isinstance(place, str) 
                        else place for place in user_prediction
                    ]
                    responses.append(RecommendationResponse(
                        user_id=user.user_id,
                        recommendations=recommendations,
                        status="success"
                    ))
                else:
                    # Fallback for other formats
                    responses.append(RecommendationResponse(
                        user_id=user.user_id,
                        recommendations=[{"place": str(user_prediction), "score": None}],
                        status="success"
                    ))
                    
            except Exception as e:
                logging.error(f"Error processing prediction for user {user.user_id}: {e}")
                responses.append(RecommendationResponse(
                    user_id=user.user_id,
                    recommendations=[],
                    status=f"error: {str(e)}"
                ))
        
        return responses
        
    except Exception as e:
        logging.error(f"Error during batch prediction: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/refresh-model")
def refresh_model():
    """Manually refresh the model"""
    logging.info("Manual model refresh requested")
    load_model()
    
    if model_info["status"] == "ready":
        return {"message": "Model refreshed successfully", "status": model_info["status"]}
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to refresh model. Status: {model_info['status']}, Error: {model_info.get('error')}"
        )

@app.get("/model-info")
def get_model_info():
    """Get detailed model information"""
    return model_info
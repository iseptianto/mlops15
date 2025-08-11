# fastapibaru/app/app.py
# Run: uvicorn app:app --host 0.0.0.0 --port 8000
import os
import logging
from typing import List, Dict, Any, Optional, Set

import numpy as np
import pandas as pd
import mlflow
from mlflow.exceptions import MlflowException
from fastapi import FastAPI, HTTPException, Path, Body
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# App & Logging
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Tourism Recommender API",
    description="API for tourism destination recommendations",
    version="1.0.0",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class TourismInput(BaseModel):
    user_id: str

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    status: str

class ExistenceReq(BaseModel):
    user_id: str
    place_id: Optional[str] = None

# -----------------------------------------------------------------------------
# Global state
# -----------------------------------------------------------------------------
model = None
model_info = {
    "name": os.getenv("MLFLOW_REGISTERED_MODEL", "tourism-recommender-model"),
    "alias": "production",
    "status": "not_loaded",
    "error": None,
}

# Catalog (optional, buat cek user/place & UX API)
known_users: Set[str] = set()
known_places: Set[str] = set()
user_seen: Dict[str, Set[str]] = {}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _normalize_pred_to_list_of_dict(pred) -> List[Dict[str, Any]]:
    """
    Ubah berbagai bentuk prediksi jadi list[{'place': str, 'score': float|None}]
    Tahan banting untuk: numpy array, pandas Series/DataFrame, dict, list, scalar.
    """
    # DataFrame -> records
    if isinstance(pred, pd.DataFrame):
        rows = pred.to_dict(orient="records")
        return [
            r if isinstance(r, dict) else {"place": str(r), "score": None}
            for r in rows
        ]
    # Series -> list
    if isinstance(pred, pd.Series):
        pred = pred.astype(object).tolist()
    # numpy -> list
    if isinstance(pred, np.ndarray):
        pred = pred.tolist()
    # dict spesial (punya "recommendations")
    if isinstance(pred, dict):
        if "recommendations" in pred and isinstance(pred["recommendations"], (list, tuple)):
            seq = pred["recommendations"]
        else:
            seq = [pred]
        return [
            r if isinstance(r, dict) else {"place": str(r), "score": None}
            for r in seq
        ]
    # list/tuple -> list of dicts
    if isinstance(pred, (list, tuple)):
        return [
            r if isinstance(r, dict) else {"place": str(r), "score": None}
            for r in pred
        ]
    # fallback scalar
    return [{"place": str(pred), "score": None}]

def _extract_catalog_from_model(loaded_model):
    """
    Ambil daftar user & place dari python_model (kalau trainer menyimpan user_seen/popular).
    Cocok dengan trainer recsys yang kita buat (punya .user_seen & .popular).
    """
    users, places, seen_map = set(), set(), {}
    try:
        impl = getattr(loaded_model, "_model_impl", None)
        py_model = getattr(impl, "python_model", None)
        if py_model is None:
            return users, places, seen_map

        if hasattr(py_model, "user_seen"):
            # dict[str -> set[str]]
            raw = py_model.user_seen
            seen_map = {str(k): set(map(str, v)) for k, v in raw.items()}
            users = set(seen_map.keys())
            for s in seen_map.values():
                places.update(s)
        if hasattr(py_model, "popular"):
            places.update(map(str, getattr(py_model, "popular")))
    except Exception as e:
        logging.warning(f"Catalog extraction skipped: {e}")
    return users, places, seen_map

def _load_mlflow_model():
    """Load model dari MLflow Model Registry ke global `model` + rebuild catalog."""
    global model, model_info, known_users, known_places, user_seen

    try:
        # MLflow tracking
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5001")
        mlflow.set_tracking_uri(tracking_uri)
        logging.info(f"MLflow tracking URI: {tracking_uri}")

        # S3/MinIO credentials (agar mlflow bisa fetch artifacts)
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        os.environ["AWS_EC2_METADATA_DISABLED"] = os.getenv("AWS_EC2_METADATA_DISABLED", "true")
        os.environ["AWS_S3_ADDRESSING_STYLE"] = os.getenv("AWS_S3_ADDRESSING_STYLE", "path")

        # Load via alias
        model_uri = f"models:/{model_info['name']}@{model_info['alias']}"
        logging.info(f"Loading model from: {model_uri}")

        loaded = mlflow.pyfunc.load_model(model_uri)
        model = loaded
        model_info.update({"status": "ready", "error": None})
        logging.info("✅ Model loaded successfully.")

        # Build catalog (opsional, tergantung trainer)
        u, p, s = _extract_catalog_from_model(model)
        known_users, known_places, user_seen = u, p, s
        logging.info(f"Catalog built: users={len(known_users)}, places={len(known_places)}")

        # Smoke test (tidak menggagalkan load)
        try:
            test_df = pd.DataFrame({"user_id": ["user_001"]})
            _ = model.predict(test_df)
            logging.info("✅ Smoke test prediction OK.")
        except Exception as e:
            logging.warning(f"⚠️ Smoke test failed: {e}")

    except MlflowException as e:
        msg = f"MLflow error: {e}"
        model_info.update({"status": "error", "error": msg})
        logging.error(f"❌ {msg}")
    except Exception as e:
        msg = f"General error loading model: {e}"
        model_info.update({"status": "error", "error": msg})
        logging.error(f"❌ {msg}")

# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    _load_mlflow_model()
    # Expose /metrics kalau lib tersedia (supaya Prometheus tidak 404)
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "api_status": "ok",
        "model_name": model_info["name"],
        "model_alias": model_info["alias"],
        "model_status": model_info["status"],
        "model_error": model_info["error"],
    }

@app.get("/health")
def health():
    is_ready = (model is not None) and (model_info["status"] == "ready")
    return {
        "healthy": is_ready,
        "model_ready": model is not None,
        "status": model_info["status"],
    }

@app.get("/model-info")
def get_model_info():
    return {
        **model_info,
        "num_known_users": len(known_users),
        "num_known_places": len(known_places),
    }

@app.post("/refresh-model")
def refresh_model():
    logging.info("Manual model refresh requested.")
    _load_mlflow_model()
    if model_info["status"] == "ready":
        return {"message": "Model refreshed successfully", "status": model_info["status"]}
    raise HTTPException(
        status_code=500,
        detail=f"Failed to refresh model. Status: {model_info['status']}, Error: {model_info.get('error')}",
    )

@app.get("/exists/user/{user_id}")
def exists_user(user_id: str = Path(...)):
    return {"user_id": user_id, "exists": user_id in known_users}

@app.get("/exists/place/{place_id}")
def exists_place(place_id: str = Path(...)):
    return {"place_id": place_id, "exists": place_id in known_places}

@app.post("/exists")
def exists_user_place(req: ExistenceReq = Body(...)):
    user_ok = req.user_id in known_users
    place_ok = (req.place_id in known_places) if req.place_id else None
    seen = None
    if user_ok and req.place_id is not None:
        seen = req.place_id in user_seen.get(req.user_id, set())
    return {
        "user_id": req.user_id,
        "user_exists": user_ok,
        "place_id": req.place_id,
        "place_exists": place_ok,
        "user_has_seen_or_rated_place": seen,
    }

@app.post("/predict", response_model=List[RecommendationResponse])
def predict(users: List[TourismInput]):
    """
    Input: [{"user_id":"123"}, {"user_id":"999"}]
    Output: List of RecommendationResponse (satu per user)
    """
    if model is None or model_info["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready. Status: {model_info['status']}, Error: {model_info.get('error','Unknown')}",
        )
    if not users:
        raise HTTPException(status_code=400, detail="No users provided")

    try:
        input_df = pd.DataFrame([{"user_id": u.user_id} for u in users])
        raw_preds = model.predict(input_df)

        # pastikan list sepanjang input
        if isinstance(raw_preds, np.ndarray):
            raw_preds = raw_preds.tolist()
        if not isinstance(raw_preds, list):
            raw_preds = [raw_preds] * len(users)

        responses: List[RecommendationResponse] = []
        for i, u in enumerate(users):
            try:
                pred_i = raw_preds[i] if i < len(raw_preds) else []
                recs = _normalize_pred_to_list_of_dict(pred_i)
                responses.append(
                    RecommendationResponse(
                        user_id=u.user_id,
                        recommendations=recs,
                        status="success",
                    )
                )
            except Exception as e:
                logging.exception(f"Prediction formatting error for user {u.user_id}: {e}")
                responses.append(
                    RecommendationResponse(
                        user_id=u.user_id,
                        recommendations=[],
                        status=f"error: {e}",
                    )
                )

        return responses

    except Exception as e:
        logging.exception(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

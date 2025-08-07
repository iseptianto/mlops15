import pickle
import numpy as np
import pandas as pd
import time
import logging
import os
from typing import List, Dict

from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from geopy.distance import geodesic
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Tourism Recommendation API", version="1.0.0")

# Security
security = HTTPBearer()

def is_valid_token(token: str) -> bool:
    """Simple token validation - implement proper JWT/OAuth in production"""
    # For demo purposes - use proper authentication in production
    valid_tokens = ["your-secret-token", "demo-token", "test-token"]
    return token in valid_tokens

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the provided token"""
    if not is_valid_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Prometheus metrics
prediction_counter = Counter('model_predictions_total', 'Total predictions made', ['endpoint'])
prediction_latency = Histogram('model_prediction_duration_seconds', 'Time spent on prediction', ['endpoint'])
model_accuracy = Gauge('model_current_accuracy', 'Current model accuracy')
data_drift_score = Gauge('data_drift_score', 'Data drift detection score')

# Global variables for models and data
prediction = None
user_id_map = None
place_id_map = None
content_sim = None
tourism_df = None
place_lookup = None
tfidf = None
tfidf_matrix = None
ratings_df = None
user_rated_items_train = {}
models_loaded = False

# Load models and data
MODELS_DIR = os.getenv("MODELS_DIR", "/app")  # ✅ Match dengan workdir

def load_models_and_data():
    """Load all required models and data"""
    global prediction, user_id_map, place_id_map, content_sim, tourism_df
    global place_lookup, tfidf, tfidf_matrix, ratings_df, user_rated_items_train, models_loaded
    
    try:
        logger.info("Loading models and data...")
        
        # Load models
        with open(os.path.join(MODELS_DIR, "prediction_matrix_best.pkl"), "rb") as f:
            prediction = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "user_id_map.pkl"), "rb") as f:
            user_id_map = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "place_id_map.pkl"), "rb") as f:
            place_id_map = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "content_similarity.pkl"), "rb") as f:
            content_sim = pickle.load(f)

        # Load tourism data
        tourism_df = pd.read_csv(os.path.join(MODELS_DIR, "tourism_with_id.csv"))
        
        # Handle missing values
        if 'Time_Minutes' in tourism_df.columns and tourism_df['Time_Minutes'].isnull().any():
            mean_time_minutes = tourism_df['Time_Minutes'].mean()
            tourism_df['Time_Minutes'].fillna(mean_time_minutes, inplace=True)

        # Create place lookup
        place_lookup = tourism_df.set_index("Place_Id")

        # Prepare TF-IDF
        place_metadata = tourism_df[["Place_Id", "Category", "City"]].drop_duplicates().set_index("Place_Id")
        place_metadata['text'] = place_metadata['Category'] + ' ' + place_metadata['City']
        
        reverse_place_map = {v: k for k, v in place_id_map.items()}
        ordered_place_ids = [reverse_place_map[i] for i in range(len(place_id_map))]
        ordered_place_metadata = place_metadata.loc[ordered_place_ids].reset_index()
        
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(ordered_place_metadata['text'])

        # Load ratings data
        try:
            ratings_df = pd.read_csv(os.path.join(MODELS_DIR, "tourism_rating.csv"))
            ratings_df = ratings_df.drop_duplicates()
            ratings_df['user'] = ratings_df['User_Id'].map(user_id_map)
            ratings_df['place'] = ratings_df['Place_Id'].map(place_id_map)

            # Create user rated items mapping
            valid_ratings = ratings_df.dropna(subset=['user', 'place'])
            for row in valid_ratings.itertuples():
                user_idx = int(row.user)
                place_idx = int(row.place)
                if user_idx in user_rated_items_train:
                    user_rated_items_train[user_idx].add(place_idx)
                else:
                    user_rated_items_train[user_idx] = {place_idx}
        
        except FileNotFoundError:
            logger.warning("Ratings data not found. Filtering may not work properly.")
            user_rated_items_train = {}

        models_loaded = True
        logger.info("Models and data loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models and data: {e}")
        models_loaded = False
        raise

# Load models on startup
load_models_and_data()

# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if v <= 0:
            raise ValueError('user_id must be positive')
        return v
    
    @validator('top_n')
    def validate_top_n(cls, v):
        if not 1 <= v <= 50:
            raise ValueError('top_n must be between 1 and 50')
        return v

class HybridRecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5
    alpha: float = 0.5
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if v <= 0:
            raise ValueError('user_id must be positive')
        return v
    
    @validator('top_n')
    def validate_top_n(cls, v):
        if not 1 <= v <= 50:
            raise ValueError('top_n must be between 1 and 50')
        return v
    
    @validator('alpha')
    def validate_alpha(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('alpha must be between 0.0 and 1.0')
        return v

class NearbyPlacesRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: int = 50
    
    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('latitude must be between -90 and 90')
        return v
    
    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('longitude must be between -180 and 180')
        return v

class SimilarPlacesRequest(BaseModel):
    place_name: str
    top_n: int = 5
    
    @validator('place_name')
    def validate_place_name(cls, v):
        if not v.strip():
            raise ValueError('place_name cannot be empty')
        return v.strip()

# Helper functions
def get_top_n_indices(user_idx: int, scores: np.ndarray, rated_items: Dict[int, set], top_n: int) -> List[int]:
    """Get top N recommendation indices, excluding already rated items"""
    scores_copy = scores.copy()
    
    if user_idx in rated_items:
        scores_copy[list(rated_items[user_idx])] = -np.inf
    
    if top_n < len(scores_copy):
        part_indices = np.argpartition(scores_copy, -top_n)[-top_n:]
        top_indices = part_indices[np.argsort(scores_copy[part_indices])][::-1]
    else:
        top_indices = np.argsort(scores_copy)[::-1]
    
    return top_indices.tolist()

# API Endpoints
@app.get("/")
async def read_root():
    """Root endpoint"""
    return {"message": "Tourism Recommendation System API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if not models_loaded:
            return {"status": "unhealthy", "reason": "Models not loaded"}
        
        if prediction is None or tourism_df is None:
            return {"status": "unhealthy", "reason": "Critical data missing"}
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "models_loaded": models_loaded,
            "data_loaded": tourism_df is not None,
            "prediction_matrix_shape": prediction.shape if prediction is not None else None
        }
    except Exception as e:
        return {"status": "unhealthy", "reason": f"Error: {str(e)}"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/recommend/user")
async def recommend_user_endpoint(
    request: RecommendationRequest,
    token: str = Depends(verify_token)
):
    """Get user-based collaborative filtering recommendations"""
    start_time = time.time()
    
    try:
        user_id = request.user_id
        top_n = request.top_n

        if user_id not in user_id_map:
            raise HTTPException(status_code=404, detail=f"User ID {user_id} not found")

        user_idx = user_id_map[user_id]
        
        if user_idx >= prediction.shape[0]:
            raise HTTPException(status_code=500, detail="User index out of bounds")

        cf_scores = prediction[user_idx]
        top_indices = get_top_n_indices(user_idx, cf_scores, user_rated_items_train, top_n)
        
        # Get place details
        reverse_place_map = {v: k for k, v in place_id_map.items()}
        top_place_ids = [reverse_place_map.get(idx) for idx in top_indices if reverse_place_map.get(idx) is not None]
        valid_place_ids = [id for id in top_place_ids if id in place_lookup.index]
        
        if not valid_place_ids:
            return {"user_id": user_id, "recommendations": []}
        
        recommendations = place_lookup.loc[valid_place_ids][['Place_Name', 'City', 'Category', 'Rating']].to_dict(orient="records")
        
        # Record metrics
        prediction_counter.labels(endpoint='user_recommend').inc()
        prediction_latency.labels(endpoint='user_recommend').observe(time.time() - start_time)
        
        return {"user_id": user_id, "recommendations": recommendations}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in user recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Add other endpoints similarly...
# (I'll include the key ones here)

@app.post("/places/nearby")
async def find_nearby_places_endpoint(
    request: NearbyPlacesRequest,
    token: str = Depends(verify_token)
):
    """Find nearby places"""
    start_time = time.time()
    
    try:
        lat, lon, radius_km = request.latitude, request.longitude, request.radius_km
        
        def calculate_distance(row):
            if pd.isna(row['Lat']) or pd.isna(row['Long']):
                return np.inf
            try:
                return geodesic((lat, lon), (row['Lat'], row['Long'])).km
            except Exception:
                return np.inf

        nearby_places_df = tourism_df.copy()
        nearby_places_df['Distance_km'] = nearby_places_df.apply(calculate_distance, axis=1)
        
        nearby_places_filtered = nearby_places_df[nearby_places_df['Distance_km'] <= radius_km]
        nearby_places_sorted = nearby_places_filtered.sort_values('Distance_km')
        
        recommendations = nearby_places_sorted[['Place_Name', 'City', 'Category', 'Distance_km', 'Rating']].to_dict(orient="records")
        
        prediction_counter.labels(endpoint='nearby_places').inc()
        prediction_latency.labels(endpoint='nearby_places').observe(time.time() - start_time)
        
        return {
            "latitude": lat,
            "longitude": lon,
            "radius_km": radius_km,
            "nearby_places": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error in nearby places: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
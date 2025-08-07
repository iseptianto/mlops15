import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from geopy.distance import geodesic
import os
from sklearn.metrics.pairwise import cosine_similarity # Import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer # Import TfidfVectorizer

# --- Load Models and Data ---
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models") # Default to /app/models inside container

try:
    # Load saved models and mappings
    with open(os.path.join(MODELS_DIR, "prediction_matrix_best.pkl"), "rb") as f:
        prediction = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "user_id_map.pkl"), "rb") as f:
        user_id_map = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "place_id_map.pkl"), "rb") as f:
        place_id_map = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "content_similarity.pkl"), "rb") as f:
        content_sim = pickle.load(f)

    # Create reverse mappings for quick lookups
    reverse_user_map = {v: k for k, v in user_id_map.items()}
    reverse_place_map = {v: k for k, v in place_id_map.items()}

    # Load tourism data for place lookup and content-based features
    # Assume tourism_with_id.csv is accessible, e.g., copied into the models directory
    TOURISM_DATA_PATH = os.path.join(MODELS_DIR, "tourism_with_id.csv")
    tourism_df = pd.read_csv(TOURISM_DATA_PATH)

    # Handle missing 'Time_Minutes' with mean as done in the notebook
    if 'Time_Minutes' in tourism_df.columns and tourism_df['Time_Minutes'].isnull().any():
         mean_time_minutes = tourism_df['Time_Minutes'].mean()
         tourism_df['Time_Minutes'].fillna(mean_time_minutes, inplace=True)

    # Create place lookup DataFrame
    place_lookup = tourism_df.set_index("Place_Id")

    # Prepare TF-IDF matrix and vectorizer for content-based part
    place_metadata = tourism_df[["Place_Id", "Category", "City"]].drop_duplicates().set_index("Place_Id")
    place_metadata['text'] = place_metadata['Category'] + ' ' + place_metadata['City']

    # Ensure the order of places for TF-IDF matrix is consistent with place_id_map indices
    # This is crucial for aligning content_sim and prediction matrices
    ordered_place_ids = [reverse_place_map[i] for i in range(len(place_id_map))]
    # Filter place_metadata to include only places present in place_id_map
    ordered_place_metadata = place_metadata.loc[ordered_place_ids].reset_index()

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(ordered_place_metadata['text'])

    # The content_sim matrix loaded should already be based on this ordered tfidf_matrix if saved correctly.
    # If not, it should be recalculated here:
    # content_sim = cosine_similarity(tfidf_matrix)

except FileNotFoundError as e:
    print(f"Error loading required files: {e}. Please ensure model files and tourism_with_id.csv are in the {MODELS_DIR} directory.")
    exit() # Exit if critical files cannot be loaded
except Exception as e:
    print(f"An error occurred during model or data loading: {e}")
    exit()

# Assume ratings data is needed to filter out already rated items for users
# Load and process ratings data
try:
    RATINGS_DATA_PATH = os.path.join(MODELS_DIR, "tourism_rating.csv") # Adjusted path
    ratings_df = pd.read_csv(RATINGS_DATA_PATH)
    ratings_df = ratings_df.drop_duplicates()
    # Map original IDs to numerical indices using the loaded maps
    ratings_df['user'] = ratings_df['User_Id'].map(user_id_map)
    ratings_df['place'] = ratings_df['Place_Id'].map(place_id_map)

    # Create a dictionary mapping user_idx to a set of rated place_idx for efficient lookup
    user_rated_items_train = {}
    # Only include ratings where both user and place were successfully mapped
    valid_ratings = ratings_df.dropna(subset=['user', 'place'])
    for row in valid_ratings.itertuples():
         user_idx = int(row.user) # Cast to int as map might return float
         place_idx = int(row.place) # Cast to int
         if user_idx in user_rated_items_train:
             user_rated_items_train[user_idx].add(place_idx)
         else:
             user_rated_items_train[user_idx] = {place_idx}

except FileNotFoundError:
     print("Error loading ratings data. Recommendation functions filtering rated items might not work correctly.")
     user_rated_items_train = {} # Fallback to empty, but filtering won't work
except Exception as e:
     print(f"An error occurred during ratings data processing: {e}")
     user_rated_items_train = {} # Fallback in case of other errors


# --- FastAPI App ---
app = FastAPI()

# --- Pydantic Models for Request Body Validation ---
class RecommendationRequest(BaseModel):
    user_id: int # Assuming the original integer User_Id
    top_n: int = 5

class HybridRecommendationRequest(BaseModel):
    user_id: int # Assuming the original integer User_Id
    top_n: int = 5
    alpha: float = 0.5 # Weight for Collaborative Filtering (0.0 to 1.0)

class NearbyPlacesRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: int = 50

class SimilarPlacesRequest(BaseModel):
    place_name: str
    top_n: int = 5

# --- Helper Functions ---

# Helper function to get top N recommendations indices, excluding rated items
def get_top_n_indices(user_idx: int, scores: np.ndarray, rated_items: Dict[int, set], top_n: int) -> List[int]:
    """
    Gets indices of top N items based on scores, excluding items in rated_items for the given user index.
    """
    # Create a copy to avoid modifying the original scores array
    scores_copy = scores.copy()

    # Set scores for rated items to a very low value
    if user_idx in rated_items:
        scores_copy[list(rated_items[user_idx])] = -np.inf

    # Get indices of top_n items with highest scores
    # Use argpartition for efficiency if only top_n are needed, then sort top_n
    if top_n < len(scores_copy):
        # Get indices of the top_n elements
        part_indices = np.argpartition(scores_copy, -top_n)[-top_n:]
        # Sort these top_n indices by their scores in descending order
        top_indices = part_indices[np.argsort(scores_copy[part_indices])][::-1]
    else:
        # If top_n is greater than or equal to the total number of items, just sort all
        top_indices = np.argsort(scores_copy)[::-1]

    return top_indices.tolist()


# Helper function to get user profile for content-based part
def get_user_profile(user_id: int, tfidf_matrix, user_id_map: Dict[int, int], ratings_data_mapped: pd.DataFrame, tfidf_vectorizer) -> np.ndarray:
    """
    Creates a content-based profile for a user based on their rated items.
    ratings_data_mapped should contain 'user', 'place', and 'Place_Ratings' columns.
    tfidf_vectorizer is the fitted TfidfVectorizer instance.
    """
    if user_id not in user_id_map:
        return np.zeros(tfidf_matrix.shape[1]) # Return zero profile if user not found

    user_idx = user_id_map[user_id]

    # Filter ratings data for the specific user and ensure valid indices
    user_ratings = ratings_data_mapped[(ratings_data_mapped['user'] == user_idx) &
                                       (ratings_data_mapped['place'] < tfidf_matrix.shape[0]) &
                                       (ratings_data_mapped['place'] >= 0)]


    profile = np.zeros(tfidf_matrix.shape[1])

    if user_ratings.empty:
        return profile # Return zero profile if no valid ratings for the user

    # Accumulate weighted TF-IDF vectors of rated items
    for row in user_ratings.itertuples():
        # tfidf_matrix is based on ordered_place_metadata which is ordered by place_id_map indices
        tfidf_vec = tfidf_matrix[int(row.place)].toarray()[0] # Ensure index is int
        profile += row.Place_Ratings * tfidf_vec

    # Normalize the profile (optional but often helpful)
    # norm = np.linalg.norm(profile)
    # if norm > 0:
    #     profile /= norm

    return profile


# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Recommendation System API is running!"}

@app.post("/recommend/user")
async def recommend_user_endpoint(request: RecommendationRequest):
    user_id_str = request.user_id # Original User_Id (int)
    top_n = request.top_n

    if user_id_str not in user_id_map:
        raise HTTPException(status_code=404, detail=f"User ID {user_id_str} not found.")

    user_idx = user_id_map[user_id_str]

    # Get collaborative filtering scores from the prediction matrix
    if user_idx >= prediction.shape[0]:
         raise HTTPException(status_code=500, detail=f"Internal error: User index {user_idx} out of bounds for prediction matrix.")
    cf_scores = prediction[user_idx]

    # Get top N indices, excluding items rated in training
    # Use the preloaded user_rated_items_train dictionary
    top_indices = get_top_n_indices(user_idx, cf_scores, user_rated_items_train, top_n)

    # Get original Place_Ids from indices
    top_place_ids = [reverse_place_map.get(idx) for idx in top_indices]
    # Filter out any potential None if reverse_place_map is incomplete (shouldn't be if maps are correct)
    top_place_ids = [id for id in top_place_ids if id is not None]


    # Get place details from place_lookup
    # Ensure Place_Ids exist in place_lookup index
    valid_place_ids = [id for id in top_place_ids if id in place_lookup.index]

    if not valid_place_ids:
         return {"user_id": user_id_str, "recommendations": []} # Return empty list if no valid places found

    recommendations = place_lookup.loc[valid_place_ids][['Place_Name', 'City', 'Category', 'Rating']].to_dict(orient="records")

    return {"user_id": user_id_str, "recommendations": recommendations}


@app.post("/recommend/hybrid")
async def hybrid_recommend_endpoint(request: HybridRecommendationRequest):
    user_id_str = request.user_id # Original User_Id (int)
    top_n = request.top_n
    alpha = request.alpha

    if user_id_str not in user_id_map:
        raise HTTPException(status_code=404, detail=f"User ID {user_id_str} not found.")

    user_idx = user_id_map[user_id_str]

    # Collaborative Filtering scores
    if user_idx >= prediction.shape[0]:
         raise HTTPException(status_code=500, detail=f"Internal error: User index {user_idx} out of bounds for prediction matrix.")
    cf_scores = prediction[user_idx]

    # Content-Based scores
    # Recalculate user profile using the loaded ratings_df and tfidf_matrix/vectorizer
    user_profile = get_user_profile(user_id_str, tfidf_matrix, user_id_map, ratings_df, tfidf)

    if user_profile is None or np.sum(user_profile) == 0:
        # If profile is empty (user not found or no valid ratings), use only CF scores
        cb_scores = np.zeros(len(place_id_map))
        print(f"Warning: Could not create content profile for user {user_id_str}. Using only CF.")
    else:
        # Calculate cosine similarity between user profile and all item TF-IDF vectors
        # tfidf_matrix is ordered by place_id_map indices, which matches prediction matrix rows
        cb_scores = cosine_similarity(tfidf_matrix, user_profile.reshape(1, -1)).flatten()

    # Combine scores
    # Ensure scores have the same length as the number of places
    if len(cf_scores) != len(place_id_map) or len(cb_scores) != len(place_id_map):
         raise HTTPException(status_code=500, detail="Internal error: Score matrix dimension mismatch.")

    hybrid_scores = alpha * cf_scores + (1 - alpha) * cb_scores

    # Get top N indices, excluding items rated in training
    rated_items = user_rated_items_train # Use the preloaded dictionary
    top_indices = get_top_n_indices(user_idx, hybrid_scores, rated_items, top_n)


    # Get original Place_Ids from indices
    top_place_ids = [reverse_place_map.get(idx) for idx in top_indices]
    top_place_ids = [id for id in top_place_ids if id is not None]


    # Get place details from place_lookup
    # Ensure Place_Ids exist in place_lookup index
    valid_place_ids = [id for id in top_place_ids if id in place_lookup.index]

    if not valid_place_ids:
         return {"user_id": user_id_str, "recommendations": []}

    recommendations = place_lookup.loc[valid_place_ids][['Place_Name', 'City', 'Category', 'Rating']].to_dict(orient="records")

    return {"user_id": user_id_str, "alpha": alpha, "recommendations": recommendations}


@app.post("/places/nearby")
async def find_nearby_places_endpoint(request: NearbyPlacesRequest):
    lat = request.latitude
    lon = request.longitude
    radius_km = request.radius_km

    def calculate_distance(row):
        # Check for valid Lat/Long before calculating distance
        if pd.isna(row['Lat']) or pd.isna(row['Long']):
             return np.inf # Treat as infinitely far if coordinates are missing
        try:
            return geodesic((lat, lon), (row['Lat'], row['Long'])).km
        except Exception as e:
            print(f"Error calculating distance for place {row["Place_Id"]}: {e}")
            return np.inf


    # Apply distance calculation to the loaded tourism_df
    # Work on a copy to avoid modifying the original df with Distance_km column persistently
    nearby_places_df = tourism_df.copy()
    nearby_places_df['Distance_km'] = nearby_places_df.apply(calculate_distance, axis=1)

    # Filter by radius and sort
    nearby_places_filtered = nearby_places_df[nearby_places_df['Distance_km'] <= radius_km]
    nearby_places_sorted = nearby_places_filtered.sort_values('Distance_km')

    # Select and format output columns
    output_cols = ['Place_Name', 'City', 'Category', 'Distance_km', 'Rating'] # Include Rating
    recommendations = nearby_places_sorted[output_cols].to_dict(orient="records")

    return {"latitude": lat, "longitude": lon, "radius_km": radius_km, "nearby_places": recommendations}


@app.post("/places/similar")
async def show_similar_places_endpoint(request: SimilarPlacesRequest):
    place_name = request.place_name
    top_n = request.top_n

    # Find the place in place_lookup by name (case-insensitive)
    matches = place_lookup[place_lookup['Place_Name'].str.lower() == place_name.lower()]

    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Place '{place_name}' not found.")

    # Get the Place_Id of the matched place (assuming the first match if multiple)
    target_place_id = matches.index[0]

    # Get the index of the target place in the place_id_map for similarity matrix lookup
    if target_place_id not in place_id_map:
         # This should not happen if place_lookup and place_id_map are consistent
         raise HTTPException(status_code=500, detail=f"Internal error: Place ID {target_place_id} not found in place_id_map.")

    target_place_idx = place_id_map[target_place_id]

    # Ensure the index is within the bounds of content_sim matrix
    if target_place_idx >= content_sim.shape[0]:
         raise HTTPException(status_code=500, detail=f"Internal error: Place index {target_place_idx} out of bounds for similarity matrix.")


    # Get similarity scores for the target place from the content_sim matrix
    # content_sim matrix is indexed by place_id_map indices
    sim_scores = list(enumerate(content_sim[target_place_idx]))

    # Sort scores in descending order, excluding the place itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get indices of the top_n similar places (excluding the first element which is the place itself)
    top_indices = [i for i, _ in sim_scores[1:top_n+1]]

    # Get original Place_Ids of similar places
    top_place_ids = [reverse_place_map.get(idx) for idx in top_indices]
    top_place_ids = [id for id in top_place_ids if id is not None]


    # Get place details from place_lookup
    # Ensure Place_Ids exist in place_lookup index
    valid_place_ids = [id for id in top_place_ids if id in place_lookup.index]

    if not valid_place_ids:
         return {"target_place": place_name, "similar_places": []}

    recommendations = place_lookup.loc[valid_place_ids][['Place_Name', 'City', 'Category', 'Rating']].to_dict(orient="records")

    return {"target_place": place_name, "similar_places": recommendations}

@app.get("/user/profile/{user_id}")
async def get_user_profile_endpoint(user_id: int): # Original User_Id (int)
    user_id_str = user_id # Use the original ID for the map lookup

    if user_id_str not in user_id_map:
        raise HTTPException(status_code=404, detail=f"User ID {user_id_str} not found.")

    # Recalculate user profile using the loaded ratings_df and tfidf_matrix/vectorizer
    # Note: This recalculates on every call. For performance, you might pre-calculate profiles or use a cache.
    user_profile = get_user_profile(user_id_str, tfidf_matrix, user_id_map, ratings_df, tfidf)

    if user_profile is None or np.sum(user_profile) == 0:
         # Return empty profile or error if profile cannot be created
         return {"user_id": user_id_str, "top_profile_features": [], "message": "Could not create user profile based on available ratings."}

    # Get the TF-IDF feature names (which correspond to words/terms like categories and cities)
    feature_names = tfidf.get_feature_names_out()

    # Create a dictionary of feature names and their scores in the user profile
    profile_scores = dict(zip(feature_names, user_profile))

    # Sort the scores and get the top N features (e.g., top 10 or 20)
    # Filtering for scores > 0 might be useful to only show terms the user has a positive preference for
    sorted_profile_scores = sorted(profile_scores.items(), key=lambda item: item[1], reverse=True)
    top_profile_features = [(term, score) for term, score in sorted_profile_scores if score > 0][:10] # Get top 10 with positive scores

    return {"user_id": user_id_str, "top_profile_features": top_profile_features}


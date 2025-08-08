
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

print("🚀 Starting dummy model generation...")

def wait_for_mlflow():
    """Wait for MLflow server to be ready"""
    import requests
    
    mlflow_url = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5001")
    max_retries = 15
    retry_count = 0
    
    print(f"⏳ Waiting for MLflow server at {mlflow_url}...")
    
    while retry_count < max_retries:
        try:
            response = requests.get(f"{mlflow_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ MLflow server is ready!")
                return True
        except:
            pass
        
        retry_count += 1
        print(f"⏳ MLflow not ready, retrying... ({retry_count}/{max_retries})")
        time.sleep(3)
    
    print("❌ MLflow server not ready, proceeding anyway...")
    return False

# Wait for MLflow to be ready
wait_for_mlflow()

# Create models directory
os.makedirs('/app/models', exist_ok=True)
print("📁 Created models directory")

print("🔄 Generating dummy tourism data...")

# Generate dummy tourism data
np.random.seed(42)

# Create user and place mappings
n_users = 100
n_places = 50

user_ids = [f"user_{i:03d}" for i in range(1, n_users + 1)]
place_ids = [f"place_{i:03d}" for i in range(1, n_places + 1)]

user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
place_id_map = {place_id: idx for idx, place_id in enumerate(place_ids)}

print(f"👥 Created {len(user_id_map)} users and {len(place_id_map)} places")

# Generate collaborative filtering matrix (user-item predictions)
prediction_matrix = np.random.random((n_users, n_places))

# Add some structure to make it more realistic
for i in range(n_users):
    # Some users prefer certain types of places
    preference_type = i % 3
    if preference_type == 0:  # Nature lovers
        prediction_matrix[i, :20] += 0.3
    elif preference_type == 1:  # Cultural enthusiasts  
        prediction_matrix[i, 20:35] += 0.3
    else:  # Adventure seekers
        prediction_matrix[i, 35:] += 0.3

# Normalize to [0, 1]
prediction_matrix = np.clip(prediction_matrix, 0, 1)

print("🤖 Generated collaborative filtering matrix")

# Generate content similarity matrix
categories = ['Nature', 'Cultural', 'Adventure', 'Beach', 'Mountain']
cities = ['Jakarta', 'Bali', 'Yogyakarta', 'Bandung', 'Surabaya']

# Create place metadata
place_metadata = []
for i, place_id in enumerate(place_ids):
    place_metadata.append({
        'Place_Id': place_id,
        'Place_Name': f'Tourism Spot {i+1}',
        'Category': np.random.choice(categories),
        'City': np.random.choice(cities),
        'Rating': np.random.uniform(3.0, 5.0),
        'Lat': np.random.uniform(-10, 5),
        'Long': np.random.uniform(95, 141)
    })

tourism_df = pd.DataFrame(place_metadata)

# Generate content similarity based on category and city
content_features = []
for _, row in tourism_df.iterrows():
    # Simple feature: one-hot encoding of category + city
    feature_vector = [0] * (len(categories) + len(cities))
    
    if row['Category'] in categories:
        cat_idx = categories.index(row['Category'])
        feature_vector[cat_idx] = 1
    
    if row['City'] in cities:
        city_idx = cities.index(row['City'])
        feature_vector[len(categories) + city_idx] = 1
    
    content_features.append(feature_vector)

content_similarity = cosine_similarity(content_features)

print("📊 Generated content similarity matrix")

# Save tourism dataset
tourism_df.to_csv('/app/models/tourism_with_id.csv', index=False)
print("💾 Saved tourism dataset")

# Generate dummy ratings data
ratings_data = []
for user_id in user_ids[:20]:  # Only some users have ratings
    n_ratings = np.random.randint(5, 15)
    rated_places = np.random.choice(place_ids, size=n_ratings, replace=False)
    
    for place_id in rated_places:
        rating = np.random.randint(1, 6)  # 1-5 star rating
        ratings_data.append({
            'User_Id': user_id,
            'Place_Id': place_id, 
            'Rating': rating
        })

ratings_df = pd.DataFrame(ratings_data)
ratings_df.to_csv('/app/models/tourism_rating.csv', index=False)
print("⭐ Generated ratings dataset")

# Save all pickle files
print("💾 Saving model artifacts...")

with open('/app/models/prediction_matrix_best.pkl', 'wb') as f:
    pickle.dump(prediction_matrix, f)
print("✅ Saved prediction matrix")

with open('/app/models/user_id_map.pkl', 'wb') as f:
    pickle.dump(user_id_map, f)
print("✅ Saved user ID mapping")

with open('/app/models/place_id_map.pkl', 'wb') as f:
    pickle.dump(place_id_map, f)
print("✅ Saved place ID mapping")

with open('/app/models/content_similarity.pkl', 'wb') as f:
    pickle.dump(content_similarity, f)
print("✅ Saved content similarity matrix")

print("🎉 All model artifacts generated successfully!")
print(f"📁 Files saved in /app/models/:")
for file in os.listdir('/app/models'):
    file_path = os.path.join('/app/models', file)
    size = os.path.getsize(file_path) / 1024  # KB
    print(f"  - {file} ({size:.1f} KB)")
EOF
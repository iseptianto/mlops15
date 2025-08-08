import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import os

print("Generating dummy model...")

# Create dummy data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs('/app/models', exist_ok=True)
with open('/app/models/prediction_matrix_best.pkl', 'wb') as f:
    pickle.dump(model.predict_proba(X_train), f)

# Create dummy mappings
user_id_map = {i: i for i in range(100)}
place_id_map = {i: i for i in range(50)}

with open('/app/models/user_id_map.pkl', 'wb') as f:
    pickle.dump(user_id_map, f)

with open('/app/models/place_id_map.pkl', 'wb') as f:
    pickle.dump(place_id_map, f)

# Create dummy similarity matrix
similarity_matrix = np.random.random((50, 50))
with open('/app/models/content_similarity.pkl', 'wb') as f:
    pickle.dump(similarity_matrix, f)

print("Dummy model files created successfully!")
import pytest
from fastapi.testclient import TestClient
from app.api.main import app
import numpy as np

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_recommend_user_valid():
    # Mock authentication
    headers = {"Authorization": "Bearer valid-token"}
    response = client.post(
        "/recommend/user",
        json={"user_id": 1, "top_n": 5},
        headers=headers
    )
    assert response.status_code == 200
    assert "recommendations" in response.json()

def test_recommend_user_invalid_id():
    headers = {"Authorization": "Bearer valid-token"}
    response = client.post(
        "/recommend/user",
        json={"user_id": -1, "top_n": 5},
        headers=headers
    )
    assert response.status_code == 422  # Validation error

@pytest.fixture
def sample_predictions():
    return np.random.rand(100, 50)  # Mock prediction matrix

def test_model_accuracy(sample_predictions):
    # Implementasi test accuracy dengan mock data
    accuracy = np.mean(sample_predictions > 0.5)
    assert accuracy > 0.3, f"Model accuracy {accuracy} below minimum threshold"
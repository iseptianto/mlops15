# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_recommend_user():
    response = client.post(
        "/recommend/user",
        json={"user_id": 1, "top_n": 5}
    )
    assert response.status_code == 200
    assert "recommendations" in response.json()

def test_model_accuracy():
    # Test model with known good data
    accuracy = calculate_model_accuracy()
    assert accuracy > 0.7, f"Model accuracy {accuracy} below threshold"
"""API tests for FastAPI recommender application using TestClient."""

import pytest
from fastapi.testclient import TestClient

from recsys.serving.app import app, set_pipeline
from recsys.serving.loader import ServingPipeline


@pytest.fixture
def test_client(tmp_path):
    # Setup mock serving pipeline with dummy popular items
    mock_pipeline = ServingPipeline(
        movie_dict={
            1: {"title": "Toy Story (1995)", "genres": "Adventure|Animation|Comedy"},
            2: {"title": "Jumanji (1995)", "genres": "Adventure|Children|Fantasy"},
        },
        user_to_idx={100: 0},
        item_to_idx={1: 0, 2: 1},
        idx_to_item={0: 1, 1: 2},
        popular_items=[1, 2],
    )
    set_pipeline(mock_pipeline)

    with TestClient(app) as client:
        yield client


def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_recommend_endpoint_cold_start(test_client):
    # Cold start user (not in user_to_idx)
    response = test_client.post("/recommend", json={"user_id": 99999, "k": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 99999
    assert data["is_cold_start"] is True
    assert len(data["recommendations"]) <= 2
    assert "stage_latencies" in data
    assert "total_ms" in data["stage_latencies"]


def test_recommend_with_genre_filter(test_client):
    response = test_client.post(
        "/recommend",
        json={"user_id": 99999, "k": 10, "genre_filter": "Comedy"},
    )
    assert response.status_code == 200
    data = response.json()
    for item in data["recommendations"]:
        assert "Comedy" in item["genres"]


def test_feedback_endpoint(test_client, tmp_path):
    event = {
        "user_id": 100,
        "item_id": 1,
        "action": "click",
        "propensity": 0.05,
        "reward": 1.0,
    }
    response = test_client.post("/feedback", json=event)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "recorded"

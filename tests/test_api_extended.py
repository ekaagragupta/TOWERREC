"""Extended integration tests for production FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from recsys.serving.app import app, set_pipeline
from recsys.serving.loader import init_production_pipeline


@pytest.fixture(scope="module")
def prod_client():
    prod_pipe = init_production_pipeline()
    set_pipeline(prod_pipe)
    with TestClient(app) as client:
        yield client
    set_pipeline(None)


def test_prod_health(prod_client):
    res = prod_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["models_loaded"]["two_tower"] is True
    assert data["models_loaded"]["faiss_index"] is True
    assert data["models_loaded"]["lgbm_ranker"] is True
    assert data["num_users"] == 600
    assert data["num_items"] == 1500


def test_prod_recommend_strategies(prod_client):
    for strategy in ["two_tower_lgbm", "two_tower_retrieval", "popularity", "item_knn"]:
        res = prod_client.post(
            "/recommend",
            json={"user_id": 42, "k": 5, "model_type": strategy},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["user_id"] == 42
        assert len(data["recommendations"]) == 5
        assert data["stage_latencies"]["total_ms"] > 0
        assert data["is_cold_start"] is False
        assert data["recommendations"][0]["explanation"] is not None


def test_prod_recommend_genre_filter(prod_client):
    res = prod_client.post(
        "/recommend",
        json={"user_id": 1, "k": 5, "genre_filter": "Animation"},
    )
    assert res.status_code == 200
    data = res.json()
    for item in data["recommendations"]:
        assert "Animation" in item["genres"]


def test_prod_movies_search(prod_client):
    res = prod_client.get("/movies?q=Godfather&limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] >= 1
    assert any("Godfather" in m["title"] for m in data["movies"])


def test_prod_genres(prod_client):
    res = prod_client.get("/genres")
    assert res.status_code == 200
    genres = res.json()
    assert len(genres) >= 15
    assert any(g["name"] == "Sci-Fi" for g in genres)


def test_prod_users_and_profile(prod_client):
    users_res = prod_client.get("/users")
    assert users_res.status_code == 200
    users = users_res.json()
    assert len(users) >= 5

    profile_res = prod_client.get("/users/42/profile")
    assert profile_res.status_code == 200
    profile = profile_res.json()
    assert profile["user_id"] == 42
    assert profile["total_ratings"] > 0
    assert len(profile["history"]) > 0


def test_prod_similar_movies(prod_client):
    res = prod_client.get("/movies/1/similar?k=4")
    assert res.status_code == 200
    similar = res.json()
    assert len(similar) == 4
    for item in similar:
        assert item["item_id"] != 1
        assert "similarity" in item


def test_prod_metrics_summary(prod_client):
    res = prod_client.get("/metrics/summary")
    assert res.status_code == 200
    data = res.json()
    assert "offline_metrics" in data
    assert "significance" in data
    assert "ann_benchmarks" in data
    assert "policy_values" in data
    assert "ablations" in data
    assert "serving_latency" in data
    assert len(data["offline_metrics"]) >= 5


def test_prod_feedback_flow(prod_client):
    payload = {
        "user_id": 42,
        "item_id": 1,
        "action": "click",
        "propensity": 0.08,
        "reward": 1.0,
    }
    res = prod_client.post("/feedback", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "recorded"

    stats_res = prod_client.get("/feedback/stats")
    assert stats_res.status_code == 200
    assert stats_res.json()["total_events"] >= 1

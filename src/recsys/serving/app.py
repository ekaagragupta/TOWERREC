"""FastAPI recommendation and feedback application."""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status

from recsys.serving.loader import ServingPipeline
from recsys.serving.schemas import (
    FeedbackEvent,
    FeedbackResponse,
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
)
from recsys.utils.logger import get_logger

logger = get_logger("recsys.serving.app")

# Global pipeline instance
pipeline: ServingPipeline | None = None
FEEDBACK_LOG_PATH = Path("data/logged_feedback.jsonl")
START_TIME = time.time()


def set_pipeline(new_pipeline: ServingPipeline) -> None:
    """Sets the active serving pipeline instance."""
    global pipeline
    pipeline = new_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models and resources on startup."""
    global pipeline
    logger.info("Initializing FastAPI recommendation server...")
    if pipeline is None:
        # Default lightweight mock pipeline if uninitialized
        pipeline = ServingPipeline(popular_items=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Shutting down FastAPI recommendation server...")


app = FastAPI(
    title="Two-Tower Retrieval & LightGBM Ranking Recommender API",
    version="0.1.0",
    description="Production-grade two-stage recommender service with counterfactual feedback logging.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Returns service health and loaded models status."""
    is_ready = pipeline is not None
    models_status = {
        "two_tower": bool(pipeline and pipeline.two_tower_model is not None),
        "faiss_index": bool(pipeline and pipeline.item_index is not None),
        "lgbm_ranker": bool(pipeline and pipeline.ranker_model is not None),
    }
    return HealthResponse(
        status="ok" if is_ready else "initializing",
        models_loaded=models_status,
        num_users=len(pipeline.user_to_idx) if pipeline else 0,
        num_items=len(pipeline.item_to_idx) if pipeline else 0,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    """Returns top-k recommendations with per-stage latency telemetry."""
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation pipeline not initialized.",
        )
    return pipeline.recommend(
        user_id=request.user_id,
        k=request.k,
        genre_filter=request.genre_filter,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def log_feedback(event: FeedbackEvent) -> FeedbackResponse:
    """Logs impression or click feedback along with propensity for counterfactual OPE."""
    try:
        record = event.model_dump()
        if record.get("timestamp") is None:
            record["timestamp"] = time.time()

        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return FeedbackResponse(status="recorded", logged_event_count=1)
    except Exception as e:
        logger.error("Failed to log feedback: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to persist feedback event.")

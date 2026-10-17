"""Pydantic schemas for the FastAPI recommendation and feedback service."""


from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="Unique user identifier.")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations to return.")
    genre_filter: str | None = Field(None, description="Optional genre name filter.")


class RecommendationItem(BaseModel):
    item_id: int
    score: float
    title: str | None = None
    genres: str | None = None
    rank: int


class StageLatency(BaseModel):
    feature_fetch_ms: float
    retrieval_ms: float
    ranking_ms: float
    total_ms: float


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendationItem]
    stage_latencies: StageLatency
    is_cold_start: bool


class FeedbackEvent(BaseModel):
    user_id: int = Field(..., description="User identifier.")
    item_id: int = Field(..., description="Item shown or clicked.")
    action: str = Field(..., description="Action type: impression, click, or rating.")
    propensity: float = Field(..., ge=0.0, le=1.0, description="Logging policy propensity p0(a|x).")
    reward: float | None = Field(None, description="Observed reward if applicable.")
    timestamp: float | None = Field(None, description="Event epoch timestamp.")


class FeedbackResponse(BaseModel):
    status: str
    logged_event_count: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    num_users: int
    num_items: int

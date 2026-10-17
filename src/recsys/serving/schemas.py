"""Pydantic schemas for the FastAPI recommendation and feedback service."""

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="Unique user identifier.")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations to return.")
    genre_filter: str | None = Field(None, description="Optional genre name filter.")
    model_type: str | None = Field("two_tower_lgbm", description="Model strategy: two_tower_lgbm, two_tower_retrieval, popularity, item_knn, bpr_mf")


class RecommendationItem(BaseModel):
    item_id: int
    score: float
    title: str | None = None
    genres: str | None = None
    year: int | None = None
    rank: int
    retrieval_score: float | None = None
    explanation: str | None = None


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
    model_used: str = "two_tower_lgbm"


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


class MovieItem(BaseModel):
    item_id: int
    title: str
    genres: str
    year: int | None = None
    avg_rating: float = 0.0
    rating_count: int = 0


class MovieSearchResponse(BaseModel):
    movies: list[MovieItem]
    total: int
    page: int
    limit: int
    total_pages: int


class GenreCount(BaseModel):
    name: str
    count: int


class SampleUser(BaseModel):
    user_id: int
    activity_tier: str
    total_ratings: int
    top_genres: list[str]


class UserHistoryItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    rating: float
    timestamp: float


class GenreAffinity(BaseModel):
    genre: str
    count: int


class UserProfileResponse(BaseModel):
    user_id: int
    activity_tier: str
    total_ratings: int
    avg_rating: float
    favorite_genres: list[GenreAffinity]
    history: list[UserHistoryItem]


class SimilarMovieItem(BaseModel):
    item_id: int
    title: str
    genres: str
    similarity: float

export interface RecommendationItem {
  item_id: int;
  score: number;
  title: string;
  genres: string;
  year?: number | null;
  rank: number;
  retrieval_score?: number | null;
  explanation?: string | null;
}

export type int = number;

export interface StageLatency {
  feature_fetch_ms: number;
  retrieval_ms: number;
  ranking_ms: number;
  total_ms: number;
}

export interface RecommendResponse {
  user_id: number;
  recommendations: RecommendationItem[];
  stage_latencies: StageLatency;
  is_cold_start: boolean;
  model_used: string;
}

export interface MovieItem {
  item_id: number;
  title: string;
  genres: string;
  year?: number | null;
  avg_rating: number;
  rating_count: number;
}

export interface MovieSearchResponse {
  movies: MovieItem[];
  total: number;
  page: number;
  limit: number;
  total_pages: number;
}

export interface GenreCount {
  name: string;
  count: number;
}

export interface SampleUser {
  user_id: number;
  activity_tier: string;
  total_ratings: number;
  top_genres: string[];
}

export interface UserHistoryItem {
  movie_id: number;
  title: string;
  genres: string;
  rating: number;
  timestamp: number;
}

export interface GenreAffinity {
  genre: string;
  count: number;
}

export interface UserProfileResponse {
  user_id: number;
  activity_tier: string;
  total_ratings: number;
  avg_rating: number;
  favorite_genres: GenreAffinity[];
  history: UserHistoryItem[];
}

export interface SimilarMovieItem {
  item_id: number;
  title: string;
  genres: string;
  similarity: number;
}

export interface HealthResponse {
  status: string;
  models_loaded: {
    two_tower: boolean;
    faiss_index: boolean;
    lgbm_ranker: boolean;
  };
  num_users: number;
  num_items: number;
}

export interface OfflineMetricRow {
  model: string;
  ndcg_10: number;
  ndcg_10_ci_lower: number;
  ndcg_10_ci_upper: number;
  recall_10: number;
  precision_10: number;
  map_10: number;
  mrr_10: number;
  hit_rate_10: number;
  coverage: number;
  gini: number;
}

export interface AnnBenchmarkRow {
  index_type: string;
  recall_50: number;
  recall_loss: number;
  p50_ms: number;
  p95_ms: number;
  qps: number;
}

export interface SignificanceRow {
  model: string;
  diff: string;
  t_stat: number;
  p_val: string;
  significant: boolean;
}

export interface AblationRow {
  feature: string;
  with_ndcg: number;
  without_ndcg: number;
  gain: string;
  insight: string;
}

export interface PolicyValueRow {
  model: string;
  true_value: number;
  ips: number;
  snips: number;
  dr: number;
}

export interface MetricsSummaryResponse {
  offline_metrics: OfflineMetricRow[];
  significance: SignificanceRow[];
  sliced_user_activity: Array<{ model: string; cold: number; warm: number; heavy: number }>;
  ann_benchmarks: AnnBenchmarkRow[];
  policy_values: PolicyValueRow[];
  ablations: AblationRow[];
  serving_latency: {
    p50_total_ms: number;
    p95_total_ms: number;
    p99_total_ms: number;
    throughput_qps: number;
    feature_fetch_ms: number;
    retrieval_ms: number;
    ranking_ms: number;
  };
  ope_sweeps: {
    temperature: any[];
    clipping: any[];
    sample_size: any[];
  };
}

export interface FeedbackEventPayload {
  user_id: number;
  item_id: number;
  action: 'impression' | 'click' | 'rating' | 'watchlist';
  propensity: number;
  reward?: number;
  timestamp?: number;
}

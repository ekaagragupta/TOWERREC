import {
  FeedbackEventPayload,
  GenreCount,
  HealthResponse,
  MetricsSummaryResponse,
  MovieSearchResponse,
  RecommendResponse,
  SampleUser,
  SimilarMovieItem,
  UserProfileResponse,
} from '../types/api'

const API_BASE = (import.meta as any).env?.VITE_API_URL || ''

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const fullUrl = `${API_BASE}${url}`
  const response = await fetch(fullUrl, {
    ...options,
    headers: {
      'Content-Control': 'no-cache',
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    let errorDetail = `HTTP Error ${response.status}: ${response.statusText}`
    try {
      const errorJson = await response.json()
      if (errorJson.detail) {
        errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail)
      }
    } catch {
      // ignore json parse failure on error body
    }
    throw new Error(errorDetail)
  }

  return response.json()
}

export const api = {
  getHealth: () => fetchJson<HealthResponse>('/api/health'),

  getRecommendations: (
    userId: number,
    k: number = 10,
    genreFilter?: string,
    modelType: string = 'two_tower_lgbm'
  ) =>
    fetchJson<RecommendResponse>('/api/recommend', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        k,
        genre_filter: genreFilter || null,
        model_type: modelType,
      }),
    }),

  searchMovies: (
    q: string = '',
    genre: string = '',
    sortBy: string = 'popularity',
    page: number = 1,
    limit: number = 24
  ) => {
    const params = new URLSearchParams()
    if (q) params.set('q', q)
    if (genre) params.set('genre', genre)
    if (sortBy) params.set('sort_by', sortBy)
    params.set('page', page.toString())
    params.set('limit', limit.toString())
    return fetchJson<MovieSearchResponse>(`/api/movies?${params.toString()}`)
  },

  getSimilarMovies: (movieId: number, k: number = 6) =>
    fetchJson<SimilarMovieItem[]>(`/api/movies/${movieId}/similar?k=${k}`),

  getGenres: () => fetchJson<GenreCount[]>('/api/genres'),

  getSampleUsers: () => fetchJson<SampleUser[]>('/api/users'),

  getUserProfile: (userId: number) =>
    fetchJson<UserProfileResponse>(`/api/users/${userId}/profile`),

  getMetricsSummary: () => fetchJson<MetricsSummaryResponse>('/api/metrics/summary'),

  logFeedback: (payload: FeedbackEventPayload) =>
    fetchJson<{ status: string; logged_event_count: number }>('/api/feedback', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getFeedbackStats: () =>
    fetchJson<{ total_events: number; recent_events: any[] }>('/api/feedback/stats'),
}

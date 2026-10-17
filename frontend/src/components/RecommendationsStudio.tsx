import React, { useEffect, useState } from 'react'
import {
  Sparkles,
  Layers,
  Filter,
  Sliders,
  Users,
  Cpu,
  TrendingUp,
  AlertCircle,
  Loader2,
  HelpCircle,
  ArrowRight,
  Flame,
} from 'lucide-react'
import { api } from '../api/client'
import { RecommendResponse } from '../types/api'
import { LatencyWaterfall } from './LatencyWaterfall'
import { MovieCard } from './MovieCard'

interface RecommendationsStudioProps {
  onSelectMovie: (id: number) => void
  onFeedbackEvent: (msg: string) => void
}

const PRESET_USERS = [
  { id: 42, label: 'User 42', desc: 'Heavy Action & Sci-Fi', tier: 'Heavy' },
  { id: 10, label: 'User 10', desc: 'Crime & Film-Noir Enthusiast', tier: 'Warm' },
  { id: 100, label: 'User 100', desc: 'Animation & Family Fan', tier: 'Warm' },
  { id: 1, label: 'User 1', desc: 'Drama & Mystery Lover', tier: 'Heavy' },
  { id: 250, label: 'User 250', desc: 'Western & Adventure Explorer', tier: 'Cold' },
  { id: 9999, label: 'User 9999', desc: 'Cold Start (Unseen Profile)', tier: 'Cold-Start' },
]

const STRATEGIES = [
  {
    id: 'two_tower_lgbm',
    name: 'Two-Tower + LightGBM',
    tag: 'Production 2-Stage',
    icon: Sparkles,
    color: 'from-indigo-600 to-cyan-500',
    desc: 'FAISS ANN retrieval followed by 10-feature LightGBM LambdaRank re-ranking.',
  },
  {
    id: 'two_tower_retrieval',
    name: 'Two-Tower Retrieval',
    tag: 'Stage 1 Only',
    icon: Layers,
    color: 'from-cyan-600 to-blue-500',
    desc: 'Pure dot-product vector search across 64-dimensional user and item embeddings.',
  },
  {
    id: 'popularity',
    name: 'Global Popularity',
    tag: 'Heuristic Baseline',
    icon: TrendingUp,
    color: 'from-amber-600 to-orange-500',
    desc: 'Frequency-ranked baseline. Suffer from high exposure bias in counterfactual OPE.',
  },
  {
    id: 'item_knn',
    name: 'Item-KNN Cosine',
    tag: 'Collaborative Baseline',
    icon: Users,
    color: 'from-emerald-600 to-teal-500',
    desc: 'Item-to-item co-occurrence matrix matching user recent interactions.',
  },
]

export const RecommendationsStudio: React.FC<RecommendationsStudioProps> = ({
  onSelectMovie,
  onFeedbackEvent,
}) => {
  const [selectedUser, setSelectedUser] = useState<number>(42)
  const [customUser, setCustomUser] = useState<string>('')
  const [strategy, setStrategy] = useState<string>('two_tower_lgbm')
  const [k, setK] = useState<number>(10)
  const [selectedGenre, setSelectedGenre] = useState<string>('')
  const [genres, setGenres] = useState<string[]>([])
  const [recData, setRecData] = useState<RecommendResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  // Load genres list
  useEffect(() => {
    api
      .getGenres()
      .then((data) => setGenres(data.map((g) => g.name)))
      .catch(console.error)
  }, [])

  // Fetch recommendations whenever controls change
  const fetchRecommendations = () => {
    setLoading(true)
    setError(null)
    const targetUserId = customUser ? parseInt(customUser, 10) || selectedUser : selectedUser

    api
      .getRecommendations(targetUserId, k, selectedGenre || undefined, strategy)
      .then((data) => {
        setRecData(data)
      })
      .catch((err) => {
        console.error('Failed to get recommendations:', err)
        setError(err.message || 'Error communicating with recommendation server.')
      })
      .finally(() => {
        setLoading(false)
      })
  }

  useEffect(() => {
    fetchRecommendations()
  }, [selectedUser, strategy, k, selectedGenre])

  const handleLike = (itemId: number) => {
    const targetUserId = customUser ? parseInt(customUser, 10) || selectedUser : selectedUser
    api
      .logFeedback({
        user_id: targetUserId,
        item_id: itemId,
        action: 'click',
        propensity: 0.05,
        reward: 1.0,
      })
      .then(() => {
        onFeedbackEvent(`Logged click interaction for Movie #${itemId} (Reward: 1.0, Propensity: 0.05)`)
      })
      .catch(console.error)
  }

  const handleRate = (itemId: number, rating: number) => {
    const targetUserId = customUser ? parseInt(customUser, 10) || selectedUser : selectedUser
    const rewardNormalized = rating / 5.0
    api
      .logFeedback({
        user_id: targetUserId,
        item_id: itemId,
        action: 'rating',
        propensity: 0.08,
        reward: rewardNormalized,
      })
      .then(() => {
        onFeedbackEvent(`Logged rating of ${rating}★ for Movie #${itemId} (Reward: ${rewardNormalized})`)
      })
      .catch(console.error)
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Studio Header & Model Architecture Pill */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 p-5 rounded-2xl border border-slate-800 bg-slate-900/60 backdrop-blur-md">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-semibold">
              Interactive Experimentation Studio
            </span>
            {recData?.is_cold_start && (
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 font-semibold flex items-center gap-1">
                <Flame className="h-3 w-3" /> Cold-Start Active
              </span>
            )}
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Personalized Two-Stage Recommendations
          </h1>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">
            Test how different architectures retrieve and rank candidate items. Compare full Two-Stage (Retrieval + LambdaRank) against Stage 1 ANN vector retrieval, Popularity, and Collaborative Item-KNN.
          </p>
        </div>

        {/* Funnel Visualizer */}
        <div className="flex items-center gap-2 p-2.5 rounded-xl bg-slate-950/80 border border-slate-800 text-xs font-mono shrink-0">
          <div className="text-center px-2">
            <div className="text-slate-400 text-[10px]">Catalog</div>
            <div className="text-slate-200 font-bold">1,500</div>
          </div>
          <ArrowRight className="h-3 w-3 text-slate-600" />
          <div className="text-center px-2">
            <div className="text-cyan-400 text-[10px]">Stage 1 ANN</div>
            <div className="text-cyan-300 font-bold">Top 50</div>
          </div>
          <ArrowRight className="h-3 w-3 text-slate-600" />
          <div className="text-center px-2">
            <div className="text-indigo-400 text-[10px]">Stage 2 Ranker</div>
            <div className="text-indigo-300 font-bold">Top {k}</div>
          </div>
        </div>
      </div>

      {/* Control Panel: User Persona & Strategy Selection */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column: User Persona Selector */}
        <div className="lg:col-span-5 space-y-4 p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Users className="h-4 w-4 text-indigo-400" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                1. Select User Profile
              </h3>
            </div>
            <span className="text-[11px] font-mono text-slate-500">
              User ID: #{customUser || selectedUser}
            </span>
          </div>

          {/* Persona Buttons */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {PRESET_USERS.map((u) => {
              const isSelected = selectedUser === u.id && !customUser
              return (
                <button
                  key={u.id}
                  onClick={() => {
                    setSelectedUser(u.id)
                    setCustomUser('')
                  }}
                  className={`p-2.5 rounded-lg border text-left transition ${
                    isSelected
                      ? 'border-indigo-500 bg-indigo-500/15 text-white shadow-sm shadow-indigo-500/10'
                      : 'border-slate-800 bg-slate-800/40 hover:bg-slate-800 text-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs font-semibold">
                    <span>{u.label}</span>
                    <span
                      className={`text-[9px] px-1 py-0.2 rounded font-mono ${
                        u.tier === 'Heavy'
                          ? 'bg-purple-500/20 text-purple-300'
                          : u.tier === 'Warm'
                          ? 'bg-blue-500/20 text-blue-300'
                          : u.tier === 'Cold-Start'
                          ? 'bg-amber-500/20 text-amber-300'
                          : 'bg-slate-700 text-slate-300'
                      }`}
                    >
                      {u.tier}
                    </span>
                  </div>
                  <p className="text-[10px] text-slate-400 truncate mt-1">{u.desc}</p>
                </button>
              )
            })}
          </div>

          {/* Custom ID Input */}
          <div className="flex items-center gap-2 pt-2 border-t border-slate-800/60">
            <span className="text-xs text-slate-400 shrink-0">Custom User ID:</span>
            <input
              type="number"
              placeholder="e.g. 7, 50, 600"
              value={customUser}
              onChange={(e) => setCustomUser(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 px-3 py-1.5 rounded-lg text-xs text-white placeholder-slate-600 focus:outline-none transition font-mono"
            />
            {customUser && (
              <button
                onClick={() => setCustomUser('')}
                className="text-[11px] text-slate-400 hover:text-white px-2 py-1 rounded bg-slate-800"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Right Column: Model Architecture Selection */}
        <div className="lg:col-span-7 space-y-4 p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-cyan-400" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                2. Recommendation Model Strategy
              </h3>
            </div>
            <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded">
              Active: {strategy}
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            {STRATEGIES.map((s) => {
              const Icon = s.icon
              const isSelected = strategy === s.id
              return (
                <button
                  key={s.id}
                  onClick={() => setStrategy(s.id)}
                  className={`p-3 rounded-xl border text-left transition relative flex flex-col justify-between ${
                    isSelected
                      ? 'border-indigo-500/80 bg-gradient-to-br from-indigo-950/40 via-slate-900 to-slate-900 text-white shadow-md shadow-indigo-500/10'
                      : 'border-slate-800 bg-slate-800/30 hover:bg-slate-800/60 text-slate-300'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2 mb-1.5">
                    <div className="flex items-center gap-2">
                      <div className={`p-1.5 rounded-lg bg-slate-800 border border-slate-700/60 ${isSelected ? 'text-indigo-400' : 'text-slate-400'}`}>
                        <Icon className="h-4 w-4" />
                      </div>
                      <span className="text-xs font-semibold text-slate-100">{s.name}</span>
                    </div>
                    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                      {s.tag}
                    </span>
                  </div>
                  <p className="text-[10px] text-slate-400 leading-snug">{s.desc}</p>
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* Filter Bar: Genre Pills & Top-K Slider */}
      <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Genre Filter Chips */}
        <div className="flex items-center gap-2 overflow-x-auto no-scrollbar py-1">
          <span className="text-xs text-slate-400 font-medium flex items-center gap-1.5 shrink-0 pr-2 border-r border-slate-800">
            <Filter className="h-3.5 w-3.5 text-indigo-400" />
            Genre:
          </span>

          <button
            onClick={() => setSelectedGenre('')}
            className={`px-3 py-1 rounded-full text-xs font-medium whitespace-nowrap transition ${
              selectedGenre === ''
                ? 'bg-indigo-600 text-white shadow'
                : 'bg-slate-800/80 text-slate-400 hover:text-slate-200'
            }`}
          >
            All Genres
          </button>

          {genres.slice(0, 10).map((g) => (
            <button
              key={g}
              onClick={() => setSelectedGenre(g === selectedGenre ? '' : g)}
              className={`px-3 py-1 rounded-full text-xs font-medium whitespace-nowrap transition ${
                selectedGenre === g
                  ? 'bg-indigo-600 text-white shadow'
                  : 'bg-slate-800/80 text-slate-400 hover:text-slate-200'
              }`}
            >
              {g}
            </button>
          ))}
        </div>

        {/* Top-K Slider */}
        <div className="flex items-center gap-3 shrink-0 self-end md:self-auto">
          <span className="text-xs text-slate-400 flex items-center gap-1">
            <Sliders className="h-3.5 w-3.5 text-indigo-400" />
            Top-K:
          </span>
          <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 font-mono text-xs">
            {[5, 10, 15, 20].map((num) => (
              <button
                key={num}
                onClick={() => setK(num)}
                className={`px-2.5 py-1 rounded font-semibold transition ${
                  k === num
                    ? 'bg-indigo-600 text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {num}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Latency Waterfall Banner */}
      {recData?.stage_latencies && (
        <LatencyWaterfall
          latencies={recData.stage_latencies}
          isColdStart={recData.is_cold_start}
          modelUsed={recData.model_used}
        />
      )}

      {/* Error Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-800 text-red-200 text-xs flex items-center gap-3">
          <AlertCircle className="h-5 w-5 text-red-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Recommendations Cards Grid */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-indigo-400" />
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
              Recommended Movies for User #{customUser || selectedUser}
            </h3>
            <span className="text-xs text-slate-500 font-mono">
              ({recData?.recommendations.length || 0} items)
            </span>
          </div>

          <span className="text-xs text-slate-400 hidden sm:block">
            Click any movie card to inspect 64-d vector space nearest neighbors
          </span>
        </div>

        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {Array.from({ length: k }).map((_, idx) => (
              <div
                key={idx}
                className="h-64 rounded-xl border border-slate-800/60 bg-slate-900/40 animate-pulse flex flex-col justify-between p-4"
              >
                <div className="h-20 bg-slate-800/40 rounded-lg" />
                <div className="space-y-2 mt-4">
                  <div className="h-4 bg-slate-800/60 rounded w-3/4" />
                  <div className="h-3 bg-slate-800/40 rounded w-1/2" />
                </div>
                <div className="h-6 bg-slate-800/40 rounded mt-4" />
              </div>
            ))}
          </div>
        ) : recData?.recommendations.length === 0 ? (
          <div className="text-center py-16 rounded-xl border border-slate-800 bg-slate-900/30">
            <p className="text-slate-400 text-sm">No recommendations match the selected genre filter.</p>
            <button
              onClick={() => setSelectedGenre('')}
              className="mt-3 px-4 py-1.5 rounded-lg bg-indigo-600 text-white text-xs font-medium"
            >
              Reset Genre Filter
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {recData?.recommendations.map((item) => (
              <MovieCard
                key={item.item_id}
                movie={item}
                isRecommendation={true}
                onLike={handleLike}
                onRate={handleRate}
                onSelectMovie={onSelectMovie}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

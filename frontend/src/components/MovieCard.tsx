import React, { useState } from 'react'
import { Heart, Star, Sparkles, Compass, Check, Info } from 'lucide-react'
import { MovieItem, RecommendationItem } from '../types/api'

interface MovieCardProps {
  movie: RecommendationItem | MovieItem
  isRecommendation?: boolean
  onLike?: (itemId: number) => void
  onRate?: (itemId: number, rating: number) => void
  onSelectMovie?: (movieId: number) => void
}

const GENRE_GRADIENTS: Record<string, string> = {
  Action: 'from-amber-600/30 via-red-900/40 to-slate-900',
  Adventure: 'from-emerald-600/30 via-teal-900/40 to-slate-900',
  Animation: 'from-cyan-600/30 via-sky-900/40 to-slate-900',
  Comedy: 'from-yellow-600/30 via-orange-900/40 to-slate-900',
  Crime: 'from-slate-700/40 via-red-950/40 to-slate-900',
  Documentary: 'from-blue-600/30 via-slate-900 to-slate-900',
  Drama: 'from-indigo-600/30 via-purple-950/40 to-slate-900',
  Fantasy: 'from-fuchsia-600/30 via-purple-900/40 to-slate-900',
  Horror: 'from-rose-700/40 via-neutral-950 to-slate-900',
  Mystery: 'from-violet-600/30 via-indigo-950/40 to-slate-900',
  Romance: 'from-pink-600/30 via-rose-950/40 to-slate-900',
  'Sci-Fi': 'from-cyan-500/30 via-indigo-900/40 to-slate-900',
  Thriller: 'from-red-600/30 via-slate-950 to-slate-900',
  Western: 'from-amber-700/30 via-stone-900/40 to-slate-900',
}

export const MovieCard: React.FC<MovieCardProps> = ({
  movie,
  isRecommendation = false,
  onLike,
  onRate,
  onSelectMovie,
}) => {
  const [liked, setLiked] = useState(false)
  const [rated, setRated] = useState<number | null>(null)
  const [showRatingMenu, setShowRatingMenu] = useState(false)

  const recItem = isRecommendation ? (movie as RecommendationItem) : null
  const movieId = movie.item_id
  const title = movie.title
  const genres = movie.genres ? movie.genres.split('|') : []
  const primaryGenre = genres[0] || 'Drama'
  const gradient = GENRE_GRADIENTS[primaryGenre] || 'from-indigo-900/30 to-slate-900'

  const handleLikeClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    setLiked(!liked)
    if (onLike) {
      onLike(movieId)
    }
  }

  const handleRate = (e: React.MouseEvent, star: number) => {
    e.stopPropagation()
    setRated(star)
    setShowRatingMenu(false)
    if (onRate) {
      onRate(movieId, star)
    }
  }

  return (
    <div
      onClick={() => onSelectMovie && onSelectMovie(movieId)}
      className="group relative rounded-xl border border-slate-800 bg-slate-900/80 hover:border-indigo-500/50 hover:shadow-lg hover:shadow-indigo-500/10 transition-all duration-200 overflow-hidden cursor-pointer flex flex-col justify-between"
    >
      {/* Visual Backdrop Banner */}
      <div className={`h-24 w-full bg-gradient-to-br ${gradient} p-3 flex flex-col justify-between relative`}>
        {/* Subtle grid pattern */}
        <div className="absolute inset-0 opacity-10 bg-[radial-gradient(#fff_1px,transparent_1px)] [background-size:12px_12px]" />

        <div className="flex items-center justify-between relative z-10">
          {/* Rank Badge for recommendations */}
          {recItem ? (
            <span className="flex items-center gap-1 font-mono text-xs font-bold px-2 py-0.5 rounded-full bg-slate-950/80 text-indigo-400 border border-indigo-500/30 shadow">
              #{recItem.rank}
            </span>
          ) : (
            <span className="text-[10px] font-mono text-slate-400 bg-slate-950/70 px-2 py-0.5 rounded border border-slate-800">
              ID #{movieId}
            </span>
          )}

          {/* Quick Actions (Like & Rate) */}
          <div className="flex items-center gap-1">
            <button
              onClick={handleLikeClick}
              title={liked ? 'Liked (Logged to Counterfactual OPE)' : 'Like this movie'}
              className={`p-1.5 rounded-full backdrop-blur-md transition ${
                liked
                  ? 'bg-rose-500 text-white shadow-md shadow-rose-500/30'
                  : 'bg-slate-950/60 text-slate-300 hover:text-rose-400 hover:bg-slate-900'
              }`}
            >
              <Heart className={`h-3.5 w-3.5 ${liked ? 'fill-current' : ''}`} />
            </button>

            <div className="relative">
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  setShowRatingMenu(!showRatingMenu)
                }}
                title="Rate movie (1-5)"
                className={`p-1.5 rounded-full backdrop-blur-md transition ${
                  rated
                    ? 'bg-amber-500 text-slate-950 font-bold'
                    : 'bg-slate-950/60 text-slate-300 hover:text-amber-400 hover:bg-slate-900'
                }`}
              >
                <Star className={`h-3.5 w-3.5 ${rated ? 'fill-current' : ''}`} />
              </button>

              {/* Rating Popup Menu */}
              {showRatingMenu && (
                <div className="absolute right-0 top-8 z-30 flex gap-1 p-1 rounded-lg bg-slate-900 border border-slate-700 shadow-xl">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      onClick={(e) => handleRate(e, star)}
                      className="p-1 hover:text-amber-400 text-slate-400 transition"
                    >
                      <Star className={`h-3.5 w-3.5 ${rated && rated >= star ? 'text-amber-400 fill-current' : ''}`} />
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Primary Genre Tag */}
        <div className="relative z-10 flex items-center justify-between">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-300/90 font-mono bg-slate-950/40 px-2 py-0.5 rounded">
            {primaryGenre}
          </span>
          {movie.year && (
            <span className="text-[11px] font-mono text-slate-400 bg-slate-950/40 px-1.5 py-0.5 rounded">
              {movie.year}
            </span>
          )}
        </div>
      </div>

      {/* Movie Details Body */}
      <div className="p-3.5 flex-1 flex flex-col justify-between">
        <div>
          <h4 className="text-sm font-semibold text-slate-100 group-hover:text-indigo-300 transition-colors line-clamp-1">
            {title}
          </h4>

          {/* Genre Chips */}
          <div className="flex flex-wrap gap-1 mt-2">
            {genres.slice(0, 3).map((g) => (
              <span
                key={g}
                className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700/60"
              >
                {g}
              </span>
            ))}
          </div>
        </div>

        {/* Bottom Recommendation Score & Explanation */}
        <div className="mt-3 pt-2.5 border-t border-slate-800/80">
          {recItem ? (
            <div>
              <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 mb-1">
                <span>Model Score:</span>
                <span className="text-indigo-400 font-bold">
                  {recItem.score.toFixed(4)}
                </span>
              </div>

              {recItem.explanation && (
                <p className="text-[10px] text-slate-400 leading-tight line-clamp-2 mt-1 bg-slate-800/30 p-1.5 rounded border border-slate-800">
                  {recItem.explanation}
                </p>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span className="flex items-center gap-1 font-mono">
                <Star className="h-3 w-3 text-amber-400 fill-amber-400" />
                {'avg_rating' in movie ? (movie.avg_rating as number).toFixed(1) : '3.5'}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">
                {'rating_count' in movie ? `${movie.rating_count} ratings` : ''}
              </span>
            </div>
          )}

          {/* Find Similar Button */}
          <button
            onClick={(e) => {
              e.stopPropagation()
              onSelectMovie && onSelectMovie(movieId)
            }}
            className="w-full mt-2.5 py-1 px-2 rounded-lg bg-slate-800/60 hover:bg-indigo-600/20 hover:text-indigo-300 border border-slate-700/50 hover:border-indigo-500/40 text-[11px] text-slate-300 flex items-center justify-center gap-1.5 transition"
          >
            <Compass className="h-3 w-3" />
            <span>Similar in Embedding Space</span>
          </button>
        </div>
      </div>
    </div>
  )
}

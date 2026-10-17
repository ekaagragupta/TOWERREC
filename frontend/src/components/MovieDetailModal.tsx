import React, { useEffect, useState } from 'react'
import { X, Sparkles, Compass, Star, Film, ExternalLink, Loader2 } from 'lucide-react'
import { api } from '../api/client'
import { SimilarMovieItem } from '../types/api'

interface MovieDetailModalProps {
  movieId: number | null
  onClose: () => void
  onSelectMovie: (id: number) => void
}

export const MovieDetailModal: React.FC<MovieDetailModalProps> = ({
  movieId,
  onClose,
  onSelectMovie,
}) => {
  const [similarMovies, setSimilarMovies] = useState<SimilarMovieItem[]>([])
  const [loading, setLoading] = useState(false)
  const [movieInfo, setMovieInfo] = useState<{ title: string; genres: string; year?: number | null } | null>(null)

  useEffect(() => {
    if (!movieId) return

    setLoading(true)
    api
      .getSimilarMovies(movieId, 6)
      .then((data) => {
        setSimilarMovies(data)
      })
      .catch((err) => {
        console.error('Failed to fetch similar movies:', err)
      })
      .finally(() => {
        setLoading(false)
      })

    // Also fetch movie basic info via search
    api
      .searchMovies('', '', 'popularity', 1, 50)
      .then((res) => {
        const found = res.movies.find((m) => m.item_id === movieId)
        if (found) {
          setMovieInfo({
            title: found.title,
            genres: found.genres,
            year: found.year,
          })
        }
      })
      .catch(console.error)
  }, [movieId])

  if (!movieId) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fadeIn">
      <div className="relative w-full max-w-2xl rounded-2xl border border-slate-700 bg-slate-900 shadow-2xl overflow-hidden">
        {/* Header Banner */}
        <div className="p-6 bg-gradient-to-r from-indigo-900/60 via-purple-900/40 to-slate-900 border-b border-slate-800 flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[11px] font-mono uppercase px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-semibold">
                MovieLens Catalog #{movieId}
              </span>
              {movieInfo?.year && (
                <span className="text-[11px] font-mono text-slate-400">
                  Released in {movieInfo.year}
                </span>
              )}
            </div>
            <h2 className="text-xl font-bold text-white">
              {movieInfo?.title || `Movie #${movieId}`}
            </h2>
            <div className="flex flex-wrap gap-1.5 mt-2">
              {movieInfo?.genres.split('|').map((g) => (
                <span
                  key={g}
                  className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700 font-medium"
                >
                  {g}
                </span>
              ))}
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800/80 transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content Section: Two-Tower Vector Space Similar Movies */}
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Compass className="h-4 w-4 text-cyan-400" />
              <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
                Nearest Neighbors in 64-D Latent Embedding Space
              </h3>
            </div>
            <span className="text-[11px] font-mono text-slate-400 bg-slate-800/60 px-2 py-0.5 rounded">
              Cosine Similarity Dot-Product
            </span>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-12 text-slate-400">
              <Loader2 className="h-6 w-6 animate-spin text-indigo-400 mr-2" />
              <span className="text-sm">Calculating vector distance in embedding space...</span>
            </div>
          ) : similarMovies.length === 0 ? (
            <div className="text-center py-8 text-slate-500 text-sm">
              No nearest neighbor items found for this ID.
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {similarMovies.map((sim) => (
                <div
                  key={sim.item_id}
                  onClick={() => onSelectMovie(sim.item_id)}
                  className="p-3 rounded-xl border border-slate-800/80 bg-slate-800/30 hover:border-cyan-500/40 hover:bg-slate-800/70 transition cursor-pointer flex items-center justify-between group"
                >
                  <div className="min-w-0 pr-2">
                    <p className="text-xs font-semibold text-slate-200 group-hover:text-cyan-300 transition truncate">
                      {sim.title}
                    </p>
                    <p className="text-[10px] text-slate-400 truncate mt-0.5">
                      {sim.genres.replace(/\|/g, ', ')}
                    </p>
                  </div>
                  <div className="shrink-0 text-right">
                    <span className="text-xs font-mono font-bold text-cyan-400">
                      {(sim.similarity * 100).toFixed(1)}%
                    </span>
                    <p className="text-[9px] text-slate-500 font-mono">similarity</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Explanation Box for Professor */}
          <div className="mt-6 p-3.5 rounded-xl bg-slate-800/40 border border-slate-800 text-xs text-slate-400 leading-relaxed">
            <span className="font-semibold text-indigo-400">Two-Tower Retrieval Logic: </span>
            This film&apos;s 64-dimensional latent vector is generated by the PyTorch Item Tower combining ID embedding, multi-hot genre projection, and normalized release year. Nearest items represent pure semantic proximity before LightGBM ranking.
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 bg-slate-950/80 border-t border-slate-800/80 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition"
          >
            Close Details
          </button>
        </div>
      </div>
    </div>
  )
}

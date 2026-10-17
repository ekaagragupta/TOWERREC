import React, { useEffect, useState } from 'react'
import { Users, Star, Clock, Sparkles, Film, ArrowRight, ShieldCheck, Flame, Loader2 } from 'lucide-react'
import { api } from '../api/client'
import { SampleUser, UserProfileResponse } from '../types/api'

interface UserProfilesProps {
  onSelectUserForRec: (userId: number) => void
  onSelectMovie: (movieId: number) => void
}

export const UserProfiles: React.FC<UserProfilesProps> = ({
  onSelectUserForRec,
  onSelectMovie,
}) => {
  const [sampleUsers, setSampleUsers] = useState<SampleUser[]>([])
  const [selectedUserId, setSelectedUserId] = useState<number>(42)
  const [profile, setProfile] = useState<UserProfileResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(false)

  // Load sample users list
  useEffect(() => {
    api.getSampleUsers().then(setSampleUsers).catch(console.error)
  }, [])

  // Load profile when user changes
  useEffect(() => {
    setLoading(true)
    api
      .getUserProfile(selectedUserId)
      .then(setProfile)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [selectedUserId])

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/60 backdrop-blur-md">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 font-semibold">
            User Persona & Temporal History
          </span>
          <span className="text-[11px] font-mono text-slate-500">
            600 MovieLens Users
          </span>
        </div>
        <h1 className="text-2xl font-bold tracking-tight text-white">
          User Interaction Sequences & Profiles
        </h1>
        <p className="text-xs text-slate-400 mt-1 max-w-2xl">
          Inspect how user histories are constructed for Two-Tower encoding. The User Tower processes the user ID alongside a recency-weighted sequence of their last 20 watched items to generate personalized retrieval vectors.
        </p>
      </div>

      {/* User Selector Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
        {sampleUsers.map((u) => {
          const isSelected = selectedUserId === u.user_id
          const isColdStart = u.user_id === 9999
          return (
            <button
              key={u.user_id}
              onClick={() => setSelectedUserId(u.user_id)}
              className={`p-3 rounded-xl border text-left transition flex flex-col justify-between ${
                isSelected
                  ? 'border-indigo-500 bg-indigo-500/15 text-white shadow-md shadow-indigo-500/10'
                  : 'border-slate-800 bg-slate-900/40 hover:bg-slate-800/60 text-slate-300'
              }`}
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold font-mono">#{u.user_id}</span>
                  {isColdStart && <Flame className="h-3 w-3 text-amber-400" />}
                </div>
                <span className="text-[10px] text-slate-400 block mt-1 line-clamp-1">
                  {u.activity_tier.split(' ')[0]}
                </span>
              </div>
              <div className="mt-2 text-[10px] text-slate-400 font-mono">
                {u.total_ratings} ratings
              </div>
            </button>
          )
        })}
      </div>

      {/* Selected User Details Panel */}
      {loading ? (
        <div className="p-12 text-center text-slate-400">
          <Loader2 className="h-6 w-6 animate-spin text-indigo-400 mx-auto mb-2" />
          <span className="text-sm">Fetching user temporal interaction sequence...</span>
        </div>
      ) : profile ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* User Profile Card */}
          <div className="lg:col-span-4 space-y-4">
            <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/60 space-y-4">
              <div className="flex items-start justify-between pb-4 border-b border-slate-800">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-xl font-bold text-white">User #{profile.user_id}</h2>
                    <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                      {profile.activity_tier}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Temporal Profile Segment
                  </p>
                </div>
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                  <span className="text-[10px] uppercase font-mono text-slate-500 block">
                    Total Ratings
                  </span>
                  <span className="text-lg font-bold text-white font-mono">
                    {profile.total_ratings}
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                  <span className="text-[10px] uppercase font-mono text-slate-500 block">
                    Avg Rating Given
                  </span>
                  <span className="text-lg font-bold text-amber-400 font-mono flex items-center gap-1">
                    {profile.avg_rating} <Star className="h-4 w-4 fill-amber-400" />
                  </span>
                </div>
              </div>

              {/* Top Favorite Genres */}
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">
                  Top Genre Affinities
                </h4>
                {profile.favorite_genres.length === 0 ? (
                  <p className="text-xs text-slate-500 italic">No historical ratings recorded.</p>
                ) : (
                  <div className="space-y-2">
                    {profile.favorite_genres.map((g) => {
                      const maxCount = profile.favorite_genres[0]?.count || 1
                      const pct = Math.round((g.count / maxCount) * 100)
                      return (
                        <div key={g.genre} className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-slate-300 font-medium">{g.genre}</span>
                            <span className="font-mono text-slate-400">{g.count} films</span>
                          </div>
                          <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                            <div
                              style={{ width: `${pct}%` }}
                              className="h-full bg-indigo-500 rounded-full"
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>

              {/* Action Button: Personalize for this user */}
              <button
                onClick={() => onSelectUserForRec(profile.user_id)}
                className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white font-semibold text-xs flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/20 transition"
              >
                <Sparkles className="h-4 w-4" />
                <span>Generate Recommendations for User #{profile.user_id}</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {/* User Watch & Rating Timeline */}
          <div className="lg:col-span-8 p-5 rounded-2xl border border-slate-800 bg-slate-900/60 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-indigo-400" />
                <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
                  Recent Interaction Sequence (Fed to User Tower)
                </h3>
              </div>
              <span className="text-xs font-mono text-slate-400">
                Recency-Weighted Timeline
              </span>
            </div>

            {profile.history.length === 0 ? (
              <div className="text-center py-16 text-slate-500 text-sm">
                No past ratings found for this user (Cold-Start Persona). The system will automatically fall back to Global Popularity with high-temperature exploration.
              </div>
            ) : (
              <div className="divide-y divide-slate-800/80 max-h-[500px] overflow-y-auto pr-2">
                {profile.history.map((item, index) => (
                  <div
                    key={`${item.movie_id}-${item.timestamp}-${index}`}
                    onClick={() => onSelectMovie(item.movie_id)}
                    className="py-3 flex items-center justify-between hover:bg-slate-800/30 px-3 rounded-xl transition cursor-pointer group"
                  >
                    <div className="min-w-0 pr-3">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-slate-200 group-hover:text-indigo-300 transition truncate">
                          {item.title}
                        </span>
                        <span className="text-[10px] font-mono text-slate-500">
                          #{item.movie_id}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 mt-0.5 truncate">
                        {item.genres.replace(/\|/g, ', ')}
                      </p>
                    </div>

                    <div className="flex items-center gap-4 shrink-0 font-mono">
                      <div className="flex items-center gap-1 text-amber-400 font-bold text-xs bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded">
                        <span>{item.rating}</span>
                        <Star className="h-3 w-3 fill-amber-400" />
                      </div>
                      <span className="text-[10px] text-slate-500 hidden sm:inline">
                        {new Date(item.timestamp * 1000).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : null}
    </div>
  )
}

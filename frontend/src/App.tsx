import React, { useEffect, useState } from 'react'
import { Navbar } from './components/Navbar'
import { RecommendationsStudio } from './components/RecommendationsStudio'
import { MovieExplorer } from './components/MovieExplorer'
import { UserProfiles } from './components/UserProfiles'
import { MetricsDashboard } from './components/MetricsDashboard'
import { ArchitectureView } from './components/ArchitectureView'
import { MovieDetailModal } from './components/MovieDetailModal'
import { FeedbackToast } from './components/FeedbackToast'
import { api } from './api/client'
import { HealthResponse } from './types/api'
import { GitBranch, Terminal, Sparkles, Cpu, Layers } from 'lucide-react'

export function App() {
  const [activeTab, setActiveTab] = useState<string>('recommend')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthLoading, setHealthLoading] = useState<boolean>(false)
  const [selectedMovieId, setSelectedMovieId] = useState<number | null>(null)
  const [toastMessage, setToastMessage] = useState<string | null>(null)

  const checkHealth = () => {
    setHealthLoading(true)
    api
      .getHealth()
      .then(setHealth)
      .catch((err) => {
        console.error('Health check failed:', err)
        setHealth(null)
      })
      .finally(() => setHealthLoading(false))
  }

  useEffect(() => {
    checkHealth()
  }, [])

  const handleSelectUserForRec = (userId: number) => {
    setActiveTab('recommend')
  }

  return (
    <div className="min-h-screen bg-[#0b0f17] text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        healthLoading={healthLoading}
        onRefreshHealth={checkHealth}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'recommend' && (
          <RecommendationsStudio
            onSelectMovie={(id) => setSelectedMovieId(id)}
            onFeedbackEvent={(msg) => setToastMessage(msg)}
          />
        )}

        {activeTab === 'movies' && (
          <MovieExplorer
            onSelectMovie={(id) => setSelectedMovieId(id)}
            onFeedbackEvent={(msg) => setToastMessage(msg)}
          />
        )}

        {activeTab === 'users' && (
          <UserProfiles
            onSelectUserForRec={handleSelectUserForRec}
            onSelectMovie={(id) => setSelectedMovieId(id)}
          />
        )}

        {activeTab === 'metrics' && <MetricsDashboard />}

        {activeTab === 'architecture' && <ArchitectureView />}
      </main>

      {/* Movie Details Modal */}
      <MovieDetailModal
        movieId={selectedMovieId}
        onClose={() => setSelectedMovieId(null)}
        onSelectMovie={(id) => setSelectedMovieId(id)}
      />

      {/* Real-Time Feedback Toast */}
      <FeedbackToast
        message={toastMessage}
        onClose={() => setToastMessage(null)}
      />

      {/* Footer */}
      <footer className="w-full border-t border-slate-800/80 bg-slate-950/60 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-slate-300">
              Two-Tower Recommender &amp; LightGBM LambdaRank
            </span>
            <span>•</span>
            <span className="font-mono text-[11px] text-slate-400">
              Counterfactual Off-Policy Evaluation (OPE)
            </span>
          </div>

          <div className="flex items-center gap-4 text-slate-400 font-mono text-[11px]">
            <span className="flex items-center gap-1">
              <Cpu className="h-3.5 w-3.5 text-indigo-400" />
              <span>Apple CPU (16 GB Friendly)</span>
            </span>
            <span>•</span>
            <a
              href="https://github.com/ekaagragupta/rs"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 hover:text-white transition"
            >
              <GitBranch className="h-3.5 w-3.5" />
              <span>GitHub Repo</span>
            </a>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App

import React from 'react'
import { Sparkles, Film, Users, BarChart3, Cpu, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react'
import { HealthResponse } from '../types/api'

interface NavbarProps {
  activeTab: string
  setActiveTab: (tab: string) => void
  health: HealthResponse | null
  healthLoading: boolean
  onRefreshHealth: () => void
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  health,
  healthLoading,
  onRefreshHealth,
}) => {
  const navItems = [
    { id: 'recommend', label: 'Recommendations', icon: Sparkles, desc: 'Two-Stage Personalization' },
    { id: 'movies', label: 'Catalog Search', icon: Film, desc: '1,500 Movies' },
    { id: 'users', label: 'User Profiles', icon: Users, desc: 'History & Personas' },
    { id: 'metrics', label: 'Model Benchmarks', icon: BarChart3, desc: 'Offline & OPE Research' },
    { id: 'architecture', label: 'Architecture & Live Logs', icon: Cpu, desc: 'System Funnel' },
  ]

  const isHealthy = health?.status === 'ok'

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800/80 bg-[#0b0f17]/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo & Title */}
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 p-0.5 shadow-lg shadow-indigo-500/20 flex items-center justify-center">
              <div className="h-full w-full bg-[#0b0f17] rounded-[10px] flex items-center justify-center">
                <Sparkles className="h-5 w-5 text-indigo-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-base tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                  TOWERREC
                </span>
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-semibold">
                  v2.0 CPU
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                Two-Tower PyTorch + FAISS + LightGBM LambdaRank
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = activeTab === item.id
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-150 ${
                    isActive
                      ? 'bg-indigo-600/15 text-indigo-300 border border-indigo-500/30 shadow-sm shadow-indigo-500/10'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className={`h-4 w-4 ${isActive ? 'text-indigo-400' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </button>
              )
            })}
          </nav>

          {/* Live System Status Pill */}
          <div className="flex items-center gap-2">
            <button
              onClick={onRefreshHealth}
              title="Refresh backend status"
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 transition"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${healthLoading ? 'animate-spin text-indigo-400' : ''}`} />
            </button>

            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs">
              <span className="relative flex h-2 w-2">
                <span
                  className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                    isHealthy ? 'bg-emerald-400' : 'bg-amber-400'
                  }`}
                />
                <span
                  className={`relative inline-flex rounded-full h-2 w-2 ${
                    isHealthy ? 'bg-emerald-500' : 'bg-amber-500'
                  }`}
                />
              </span>
              <span className="font-mono text-[11px] font-medium text-slate-300">
                {isHealthy ? 'Backend Ready' : 'Connecting'}
              </span>
              {health && (
                <span className="text-[10px] text-slate-500 hidden sm:inline border-l border-slate-700/60 pl-2 font-mono">
                  {health.num_users} users • {health.num_items} items
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Mobile Navigation Scrollbar */}
        <div className="flex md:hidden overflow-x-auto py-2 border-t border-slate-800/50 gap-1 no-scrollbar">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = activeTab === item.id
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap ${
                  isActive
                    ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                <span>{item.label}</span>
              </button>
            )
          })}
        </div>
      </div>
    </header>
  )
}

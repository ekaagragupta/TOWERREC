import React, { useEffect, useState } from 'react'
import {
  Layers,
  Database,
  Cpu,
  Zap,
  GitBranch,
  ShieldCheck,
  Server,
  Terminal,
  Activity,
  Radio,
  Clock,
  RefreshCw,
} from 'lucide-react'
import { api } from '../api/client'

export const ArchitectureView: React.FC = () => {
  const [feedbackStats, setFeedbackStats] = useState<{ total_events: number; recent_events: any[] }>({
    total_events: 0,
    recent_events: [],
  })
  const [refreshing, setRefreshing] = useState(false)

  const loadFeedbackStats = () => {
    setRefreshing(true)
    api
      .getFeedbackStats()
      .then(setFeedbackStats)
      .catch(console.error)
      .finally(() => setRefreshing(false))
  }

  useEffect(() => {
    loadFeedbackStats()
    const interval = setInterval(loadFeedbackStats, 5000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/60 backdrop-blur-md">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-semibold">
            End-to-End System Design
          </span>
          <span className="text-[11px] font-mono text-slate-500">
            CPU-Optimized Production Serving
          </span>
        </div>
        <h1 className="text-2xl font-bold tracking-tight text-white">
          Two-Stage Recommender Architecture
        </h1>
        <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">
          Production funnel decoupling lightweight, high-recall vector candidate retrieval (ANN via FAISS) from high-precision contextual scoring (GBDT LambdaRank with 10 ranking features).
        </p>
      </div>

      {/* Visual System Pipeline Stages */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Stage 1: Data & Features */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
                <Database className="h-4 w-4" />
              </div>
              <div>
                <span className="text-[10px] font-mono uppercase text-slate-500">Layer 1</span>
                <h4 className="text-xs font-bold text-white">Data & Temporal Split</h4>
              </div>
            </div>
            <ul className="space-y-2 text-[11px] text-slate-400 leading-tight">
              <li className="flex items-start gap-1.5">
                <span className="text-blue-400">•</span>
                <span>Strict global timestamp order (t_train &lt; t_val &lt; t_test)</span>
              </li>
              <li className="flex items-start gap-1.5">
                <span className="text-blue-400">•</span>
                <span>Recency-weighted interaction sequence builder ($\lambda=10^{-4}$)</span>
              </li>
              <li className="flex items-start gap-1.5">
                <span className="text-blue-400">•</span>
                <span>Zero future feature leakage verified by unit tests</span>
              </li>
            </ul>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 text-[10px] font-mono text-slate-500">
            Outputs: Features &amp; Sequences
          </div>
        </div>

        {/* Stage 2: Two-Tower Retrieval */}
        <div className="p-4 rounded-xl border border-cyan-500/30 bg-cyan-950/20 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                <Layers className="h-4 w-4" />
              </div>
              <div>
                <span className="text-[10px] font-mono uppercase text-cyan-400">Stage 1</span>
                <h4 className="text-xs font-bold text-white">Two-Tower Latent ANN</h4>
              </div>
            </div>
            <ul className="space-y-2 text-[11px] text-slate-300 leading-tight">
              <li className="flex items-start gap-1.5">
                <span className="text-cyan-400">•</span>
                <span>Dual PyTorch User Tower &amp; Item Tower (64-dim L2-normalized)</span>
              </li>
              <li className="flex items-start gap-1.5">
                <span className="text-cyan-400">•</span>
                <span>In-batch Sampled Softmax with $\log(q_j)$ popularity correction</span>
              </li>
              <li className="flex items-start gap-1.5">
                <span className="text-cyan-400">•</span>
                <span>FAISS index (FlatIP / HNSW) searches top 50 candidates in 0.01ms</span>
              </li>
            </ul>
          </div>
          <div className="mt-4 pt-3 border-t border-cyan-900/50 text-[10px] font-mono text-cyan-400">
            Funnel: 1,500 &rarr; 50 candidates
          </div>
        </div>

        {/* Stage 3: LightGBM Ranking */}
        <div className="p-4 rounded-xl border border-indigo-500/30 bg-indigo-950/20 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                <Cpu className="h-4 w-4" />
              </div>
              <div>
                <span className="text-[10px] font-mono uppercase text-indigo-400">Stage 2</span>
                <h4 className="text-xs font-bold text-white">LightGBM LambdaRank</h4>
              </div>
            </div>
            <ul className="space-y-2 text-[11px] text-slate-300 leading-tight">
              <li className="flex items-start gap-1.5">
                <span className="text-indigo-400">•</span>
                <span>Contextual GBDT scoring grouped by query timestamp</span>
              </li>
              <li className="flex items-start gap-1.5">
                <span className="text-indigo-400">•</span>
                <span>10 ranking features (TT score, user-genre affinity, popularity)</span>
              </li>
              <li className="flex items-start gap-1.5">
                <span className="text-indigo-400">•</span>
                <span>Early stopping on validation NDCG@10 (0.62ms ranking latency)</span>
              </li>
            </ul>
          </div>
          <div className="mt-4 pt-3 border-t border-indigo-900/50 text-[10px] font-mono text-indigo-400">
            Funnel: 50 candidates &rarr; Top-K
          </div>
        </div>

        {/* Stage 4: Production Serving & OPE */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <Server className="h-4 w-4" />
              </div>
              <div>
                <span className="text-[10px] font-mono uppercase text-emerald-400">Stage 4</span>
                <h4 className="text-xs font-bold text-white">FastAPI &amp; Feedback</h4>
              </div>
            </div>
            <ul className="space-y-2 text-[11px] text-slate-400 leading-tight">
              <li className="flex items-start gap-1.5">
                <span className="text-emerald-400">•</span>
                <span>Async FastAPI serving with cold-start fallback</span>
              </li>
              <li className="flex items-start gap-1.5">
                <span className="text-emerald-400">•</span>
                <span>Counterfactual logging with exploration propensities $p_0(a|x)$</span>
              </li>
              <li className="flex items-start gap-1.5">
                <span className="text-emerald-400">•</span>
                <span>p95 serving latency &lt; 1ms (1,247 QPS throughput on CPU)</span>
              </li>
            </ul>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 text-[10px] font-mono text-emerald-400">
            Telemetry: &lt; 1ms latency
          </div>
        </div>
      </div>

      {/* Live Counterfactual Feedback Stream Section */}
      <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/60 space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Radio className="h-4 w-4 text-rose-400 animate-pulse" />
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
              Live Counterfactual Feedback Event Log Stream
            </h3>
            <span className="text-xs font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded">
              Total Logged: {feedbackStats.total_events}
            </span>
          </div>

          <button
            onClick={loadFeedbackStats}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white px-2.5 py-1 rounded-lg bg-slate-800 transition"
          >
            <RefreshCw className={`h-3 w-3 ${refreshing ? 'animate-spin text-indigo-400' : ''}`} />
            <span>Refresh Stream</span>
          </button>
        </div>

        <p className="text-xs text-slate-400">
          Every interaction in the frontend triggers <code className="text-indigo-400 font-mono">POST /feedback</code>, recording the action, user, item, and policy propensity for Counterfactual Off-Policy Evaluation (OPE).
        </p>

        {feedbackStats.recent_events.length === 0 ? (
          <div className="text-center py-10 text-slate-500 text-xs font-mono">
            No feedback interactions recorded in this session yet. Like or rate movies in the Recommendation Studio to see live logged propensities.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-slate-500 text-[11px]">
                  <th className="py-2">Timestamp</th>
                  <th className="py-2 px-3">User ID</th>
                  <th className="py-2 px-3">Movie ID</th>
                  <th className="py-2 px-3">Action Type</th>
                  <th className="py-2 px-3">Logging Propensity p0(a|x)</th>
                  <th className="py-2 px-3">Observed Reward</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {feedbackStats.recent_events.map((evt, idx) => (
                  <tr key={idx} className="text-slate-300 hover:bg-slate-800/30">
                    <td className="py-2.5 text-slate-500 text-[10px]">
                      {evt.timestamp ? new Date(evt.timestamp * 1000).toLocaleTimeString() : 'now'}
                    </td>
                    <td className="py-2.5 px-3 text-indigo-300 font-bold">#{evt.user_id}</td>
                    <td className="py-2.5 px-3 text-slate-200">#{evt.item_id}</td>
                    <td className="py-2.5 px-3">
                      <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 text-[10px] font-bold uppercase">
                        {evt.action}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-cyan-400">{evt.propensity.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">
                      {evt.reward !== undefined && evt.reward !== null ? evt.reward.toFixed(2) : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

import React from 'react'
import { Clock, Zap, ArrowRight, Layers, CheckCircle2 } from 'lucide-react'
import { StageLatency } from '../types/api'

interface LatencyWaterfallProps {
  latencies: StageLatency
  isColdStart: boolean
  modelUsed: string
}

export const LatencyWaterfall: React.FC<LatencyWaterfallProps> = ({
  latencies,
  isColdStart,
  modelUsed,
}) => {
  const { feature_fetch_ms, retrieval_ms, ranking_ms, total_ms } = latencies

  // Calculate percentages
  const safeTotal = Math.max(total_ms, 0.001)
  const featPct = Math.min(100, (feature_fetch_ms / safeTotal) * 100)
  const retPct = Math.min(100, (retrieval_ms / safeTotal) * 100)
  const rankPct = Math.min(100, (ranking_ms / safeTotal) * 100)

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 shadow-sm backdrop-blur-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-indigo-400" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Pipeline Latency Waterfall Telemetry
          </h3>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            Target &lt; 100ms
          </span>
        </div>

        <div className="flex items-center gap-3 text-xs font-mono">
          <span className="text-slate-400">Total Response:</span>
          <span className="font-bold text-emerald-400 text-sm">
            {total_ms.toFixed(2)} ms
          </span>
          <span className="text-[11px] text-slate-500">
            (~{(1000 / Math.max(total_ms, 0.1)).toFixed(0)} QPS)
          </span>
        </div>
      </div>

      {/* Visual Multi-Segment Bar */}
      <div className="mt-4">
        <div className="h-2.5 w-full bg-slate-800 rounded-full overflow-hidden flex">
          <div
            style={{ width: `${featPct}%` }}
            className="bg-amber-400 transition-all duration-300"
            title={`Feature Fetch: ${feature_fetch_ms.toFixed(2)}ms`}
          />
          <div
            style={{ width: `${retPct}%` }}
            className="bg-cyan-400 transition-all duration-300"
            title={`ANN Retrieval: ${retrieval_ms.toFixed(2)}ms`}
          />
          <div
            style={{ width: `${rankPct}%` }}
            className="bg-indigo-500 transition-all duration-300"
            title={`LightGBM Ranking: ${ranking_ms.toFixed(2)}ms`}
          />
        </div>
      </div>

      {/* Stage Breakdown Grid */}
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Stage 1: Features */}
        <div className="flex items-start gap-2.5 p-2.5 rounded-lg bg-slate-800/40 border border-slate-800/60">
          <div className="mt-0.5 h-2 w-2 rounded-full bg-amber-400 shrink-0" />
          <div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Stage 0: User Embedding</span>
              <span className="font-mono text-slate-200 font-semibold">{feature_fetch_ms.toFixed(2)} ms</span>
            </div>
            <p className="text-[10px] text-slate-400 mt-0.5">
              Sequence encoding & history lookup
            </p>
          </div>
        </div>

        {/* Stage 2: ANN Retrieval */}
        <div className="flex items-start gap-2.5 p-2.5 rounded-lg bg-slate-800/40 border border-slate-800/60">
          <div className="mt-0.5 h-2 w-2 rounded-full bg-cyan-400 shrink-0" />
          <div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Stage 1: FAISS Retrieval</span>
              <span className="font-mono text-slate-200 font-semibold">{retrieval_ms.toFixed(2)} ms</span>
            </div>
            <p className="text-[10px] text-slate-400 mt-0.5">
              64-dim FlatIP/HNSW inner-product search
            </p>
          </div>
        </div>

        {/* Stage 3: LightGBM Re-Ranker */}
        <div className="flex items-start gap-2.5 p-2.5 rounded-lg bg-slate-800/40 border border-slate-800/60">
          <div className="mt-0.5 h-2 w-2 rounded-full bg-indigo-500 shrink-0" />
          <div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Stage 2: LightGBM Ranking</span>
              <span className="font-mono text-slate-200 font-semibold">{ranking_ms.toFixed(2)} ms</span>
            </div>
            <p className="text-[10px] text-slate-400 mt-0.5">
              10-feature GBDT LambdaRank scoring
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

import React, { useEffect } from 'react'
import { Radio, X, CheckCircle2 } from 'lucide-react'

interface FeedbackToastProps {
  message: string | null
  onClose: () => void
}

export const FeedbackToast: React.FC<FeedbackToastProps> = ({ message, onClose }) => {
  useEffect(() => {
    if (!message) return
    const timer = setTimeout(onClose, 4000)
    return () => clearTimeout(timer)
  }, [message, onClose])

  if (!message) return null

  return (
    <div className="fixed bottom-5 right-5 z-50 flex items-center gap-3 p-3.5 rounded-xl border border-indigo-500/40 bg-slate-900/95 shadow-2xl backdrop-blur-md text-xs text-white max-w-md animate-slideUp">
      <div className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400 shrink-0">
        <Radio className="h-4 w-4 animate-pulse" />
      </div>

      <div className="flex-1 pr-2">
        <span className="font-semibold text-indigo-300 block mb-0.5">
          Counterfactual OPE Logged
        </span>
        <p className="text-slate-300 text-[11px] leading-snug">{message}</p>
      </div>

      <button
        onClick={onClose}
        className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  )
}

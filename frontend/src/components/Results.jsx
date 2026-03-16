import React from 'react'
import {
  ShieldCheck,
  ShieldAlert,
  RefreshCw,
  Eye,
  Tag,
  ChevronRight,
} from 'lucide-react'

function confidenceColor(conf) {
  if (conf >= 0.8) return { bar: 'bg-red-500', text: 'text-red-400', badge: 'bg-red-500/10 text-red-400 border-red-500/20' }
  if (conf >= 0.6) return { bar: 'bg-orange-400', text: 'text-orange-400', badge: 'bg-orange-500/10 text-orange-400 border-orange-500/20' }
  return { bar: 'bg-yellow-400', text: 'text-yellow-400', badge: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20' }
}

function ImagePanel({ title, src, isBase64, icon: Icon }) {
  const imgSrc = isBase64 ? `data:image/png;base64,${src}` : src

  return (
    <div className="flex-1 min-w-0 flex flex-col rounded-2xl border border-white/10 bg-[#1a1a24] overflow-hidden">
      {/* Panel header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-white/5">
        <Icon className="w-4 h-4 text-indigo-400" />
        <span className="text-sm font-semibold text-slate-300">{title}</span>
      </div>
      {/* Image */}
      <div className="flex-1 flex items-center justify-center bg-[#0f0f13] p-2" style={{ minHeight: '220px' }}>
        <img
          src={imgSrc}
          alt={title}
          className="max-w-full max-h-72 object-contain rounded-lg"
        />
      </div>
    </div>
  )
}

function DetectionCard({ det, index }) {
  const pct = Math.round(det.confidence * 100)
  const colors = confidenceColor(det.confidence)

  return (
    <div className="flex flex-col gap-2 p-4 rounded-xl border border-white/8 bg-white/[0.03] hover:bg-white/[0.05] transition-colors duration-200">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <span className="flex-shrink-0 flex items-center justify-center w-6 h-6 rounded-full bg-indigo-500/15 text-indigo-400 text-xs font-bold">
            {index + 1}
          </span>
          <span className="font-semibold text-slate-200 capitalize truncate">{det.class_name.replace(/_/g, ' ')}</span>
        </div>
        <span className={`flex-shrink-0 text-xs font-bold px-2 py-0.5 rounded-full border ${colors.badge}`}>
          {pct}%
        </span>
      </div>

      {/* Confidence bar */}
      <div className="w-full h-1.5 rounded-full bg-white/5 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${colors.bar}`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* BBox coords */}
      <p className="text-xs text-slate-600 font-mono">
        bbox [{det.bbox.join(', ')}]
      </p>
    </div>
  )
}

export default function Results({ results, originalImage, onReset }) {
  const { annotated_image, detections, is_safe, alert_message } = results
  const originalSrc = originalImage ? URL.createObjectURL(originalImage) : null

  return (
    <div className="w-full space-y-6">
      {/* Alert banner */}
      {is_safe ? (
        <div className="flex items-start gap-4 px-5 py-4 rounded-2xl bg-emerald-500/10 border border-emerald-500/25">
          <div className="flex-shrink-0 flex items-center justify-center w-10 h-10 rounded-xl bg-emerald-500/20">
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
          </div>
          <div>
            <p className="font-semibold text-emerald-400 text-base">Content is Safe</p>
            <p className="text-sm text-emerald-500/80 mt-0.5">{alert_message}</p>
          </div>
        </div>
      ) : (
        <div className="flex items-start gap-4 px-5 py-4 rounded-2xl bg-red-500/10 border border-red-500/25">
          <div className="flex-shrink-0 flex items-center justify-center w-10 h-10 rounded-xl bg-red-500/20">
            <ShieldAlert className="w-5 h-5 text-red-400" />
          </div>
          <div>
            <p className="font-semibold text-red-400 text-base">Explicit Content Detected</p>
            <p className="text-sm text-red-400/70 mt-0.5">{alert_message}</p>
          </div>
        </div>
      )}

      {/* Image comparison */}
      <div className="flex flex-col sm:flex-row gap-4">
        {originalSrc && (
          <ImagePanel
            title="Original Image"
            src={originalSrc}
            isBase64={false}
            icon={Eye}
          />
        )}
        <ImagePanel
          title="Detected Objects"
          src={annotated_image}
          isBase64={true}
          icon={Tag}
        />
      </div>

      {/* Detections list */}
      {detections && detections.length > 0 && (
        <div className="rounded-2xl border border-white/10 bg-[#1a1a24] overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-white/5">
            <div className="flex items-center gap-2">
              <Tag className="w-4 h-4 text-indigo-400" />
              <span className="font-semibold text-slate-200 text-sm">Detection Results</span>
            </div>
            <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300">
              {detections.length} object{detections.length !== 1 ? 's' : ''} found
            </span>
          </div>

          {/* Cards */}
          <div className="p-4 grid gap-3 sm:grid-cols-2">
            {detections.map((det, i) => (
              <DetectionCard key={i} det={det} index={i} />
            ))}
          </div>

          {/* Confidence legend */}
          <div className="flex items-center gap-4 px-5 py-3 border-t border-white/5 text-xs text-slate-500">
            <span>Confidence:</span>
            <span className="flex items-center gap-1.5"><span className="inline-block w-2.5 h-2.5 rounded-full bg-red-500" /> &ge;80%</span>
            <span className="flex items-center gap-1.5"><span className="inline-block w-2.5 h-2.5 rounded-full bg-orange-400" /> 60–79%</span>
            <span className="flex items-center gap-1.5"><span className="inline-block w-2.5 h-2.5 rounded-full bg-yellow-400" /> &lt;60%</span>
          </div>
        </div>
      )}

      {detections && detections.length === 0 && (
        <div className="flex items-center gap-3 px-5 py-4 rounded-2xl border border-white/5 bg-white/[0.02] text-slate-500 text-sm">
          <ChevronRight className="w-4 h-4" />
          No detections above the confidence threshold (0.5).
        </div>
      )}

      {/* Reset button */}
      <button
        onClick={onReset}
        className="w-full flex items-center justify-center gap-2 py-3.5 px-6 rounded-2xl border border-white/10 bg-white/[0.03] text-slate-300 hover:text-white hover:border-indigo-500/40 hover:bg-indigo-500/5 font-semibold text-sm transition-all duration-200"
      >
        <RefreshCw className="w-4 h-4" />
        Analyze Another Image
      </button>
    </div>
  )
}

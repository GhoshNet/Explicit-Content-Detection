import React, { useState, useCallback } from 'react'
import axios from 'axios'
import { Cpu, Zap, Layers } from 'lucide-react'
import Header from './components/Header.jsx'
import DropZone from './components/DropZone.jsx'
import Results from './components/Results.jsx'

// ─── Stat pill ──────────────────────────────────────────────────────────────
function StatPill({ icon: Icon, label }) {
  return (
    <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/[0.04] border border-white/8 text-slate-400 text-sm">
      <Icon className="w-3.5 h-3.5 text-indigo-400 flex-shrink-0" />
      <span>{label}</span>
    </div>
  )
}

// ─── Error toast ─────────────────────────────────────────────────────────────
function ErrorBanner({ message, onClose }) {
  if (!message) return null
  return (
    <div className="flex items-start justify-between gap-3 px-5 py-4 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
      <span>{message}</span>
      <button onClick={onClose} className="flex-shrink-0 text-red-400/60 hover:text-red-300 transition-colors text-base leading-none">
        ✕
      </button>
    </div>
  )
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')

  const handleFileSelect = useCallback((selectedFile) => {
    setFile(selectedFile)
    setResults(null)
    setError('')
  }, [])

  const handleRemove = useCallback(() => {
    setFile(null)
    setResults(null)
    setError('')
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!file) return
    setLoading(true)
    setError('')
    setResults(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post('/api/detect', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000,
      })

      setResults(response.data)
    } catch (err) {
      if (err.response) {
        setError(`Server error ${err.response.status}: ${err.response.data?.detail || 'Unknown error'}`)
      } else if (err.request) {
        setError('Cannot reach the backend server. Make sure it is running on http://localhost:8000.')
      } else {
        setError(`Request failed: ${err.message}`)
      }
    } finally {
      setLoading(false)
    }
  }, [file])

  const handleReset = useCallback(() => {
    setFile(null)
    setResults(null)
    setError('')
  }, [])

  const showHero = !file && !results

  return (
    <div className="min-h-screen flex flex-col bg-[#0f0f13]">
      <Header />

      <main className="flex-1 w-full max-w-3xl mx-auto px-4 pb-16">
        {/* ── Hero ─────────────────────────────────────────────────────── */}
        {showHero && (
          <section className="text-center pt-16 pb-10">
            {/* Glow orb */}
            <div
              aria-hidden
              className="pointer-events-none absolute left-1/2 -translate-x-1/2 top-24 w-[480px] h-[200px] rounded-full opacity-20 blur-3xl"
              style={{ background: 'radial-gradient(ellipse, #6366f1 0%, transparent 70%)' }}
            />

            <div className="relative inline-block mb-4">
              <span className="text-xs font-semibold tracking-widest uppercase text-indigo-400 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20">
                AI Content Safety
              </span>
            </div>

            <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight leading-tight mb-4">
              <span className="text-white">Detect Explicit</span>
              <br />
              <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                Content Instantly
              </span>
            </h1>

            <p className="text-slate-400 text-base sm:text-lg max-w-lg mx-auto leading-relaxed mb-10">
              Upload any image and our{' '}
              <span className="text-indigo-400 font-medium">YOLOv9 AI model</span> will analyze it
              in seconds, highlighting explicit objects with bounding boxes.
            </p>

            {/* Stats */}
            <div className="flex flex-wrap items-center justify-center gap-3">
              <StatPill icon={Cpu}    label="22M+ Parameters" />
              <StatPill icon={Zap}    label="Real-time Detection" />
              <StatPill icon={Layers} label="YOLOv9 Powered" />
            </div>
          </section>
        )}

        {/* ── Upload card ──────────────────────────────────────────────── */}
        {!results && (
          <div className={`rounded-3xl border border-white/8 bg-[#1a1a24] p-6 sm:p-8 shadow-2xl shadow-black/40 ${showHero ? '' : 'mt-10'}`}>
            {!showHero && (
              <div className="mb-6">
                <h2 className="text-xl font-bold text-slate-100">Upload an Image</h2>
                <p className="text-slate-500 text-sm mt-1">Select or drag-and-drop a JPG, PNG, or WEBP file</p>
              </div>
            )}
            {showHero && (
              <div className="mb-6">
                <h2 className="text-lg font-bold text-slate-200">Upload an Image to Get Started</h2>
                <p className="text-slate-500 text-sm mt-1">JPG, PNG, or WEBP · Max 10 MB</p>
              </div>
            )}

            <ErrorBanner message={error} onClose={() => setError('')} />

            <DropZone
              file={file}
              onFileSelect={handleFileSelect}
              onRemove={handleRemove}
              onAnalyze={handleAnalyze}
              loading={loading}
            />
          </div>
        )}

        {/* ── Results ──────────────────────────────────────────────────── */}
        {results && (
          <div className="mt-10">
            <Results results={results} originalImage={file} onReset={handleReset} />
          </div>
        )}
      </main>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="w-full border-t border-white/5 bg-[#13131d] py-6 px-4 text-center">
        <div className="flex flex-col sm:flex-row items-center justify-center gap-2 text-slate-600 text-xs">
          <span>Powered by</span>
          <span className="font-semibold text-indigo-500">YOLOv9</span>
          <span className="hidden sm:inline">·</span>
          <span>Object detection via Ultralytics &amp; OpenCV</span>
          <span className="hidden sm:inline">·</span>
          <span>FastAPI + React</span>
        </div>
      </footer>
    </div>
  )
}

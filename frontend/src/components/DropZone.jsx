import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { UploadCloud, ImageIcon, X, FileWarning } from 'lucide-react'

const MAX_SIZE = 10 * 1024 * 1024 // 10 MB
const ACCEPTED_TYPES = { 'image/jpeg': [], 'image/png': [], 'image/webp': [] }

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function DropZone({ onFileSelect, file, onRemove, onAnalyze, loading }) {
  const [error, setError] = useState('')

  const onDrop = useCallback(
    (accepted, rejected) => {
      setError('')
      if (rejected && rejected.length > 0) {
        const firstErr = rejected[0].errors[0]
        if (firstErr.code === 'file-too-large') {
          setError('File is too large. Maximum size is 10 MB.')
        } else if (firstErr.code === 'file-invalid-type') {
          setError('Invalid file type. Please upload a JPG, PNG, or WEBP image.')
        } else {
          setError(firstErr.message)
        }
        return
      }
      if (accepted && accepted.length > 0) {
        onFileSelect(accepted[0])
      }
    },
    [onFileSelect],
  )

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_SIZE,
    multiple: false,
    disabled: loading,
  })

  const preview = file ? URL.createObjectURL(file) : null

  const borderColor = isDragReject
    ? 'border-red-500/60 bg-red-500/5'
    : isDragActive
    ? 'border-indigo-400/80 bg-indigo-500/10'
    : file
    ? 'border-indigo-500/40 bg-indigo-500/5'
    : 'border-white/10 hover:border-indigo-500/40 hover:bg-white/[0.02]'

  return (
    <div className="w-full space-y-5">
      {/* Drop zone */}
      {!file ? (
        <div
          {...getRootProps()}
          className={`
            relative cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-300
            ${borderColor} ${loading ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center justify-center gap-4 py-16 px-6 text-center">
            <div
              className={`
              relative flex items-center justify-center w-16 h-16 rounded-2xl transition-all duration-300
              ${isDragActive ? 'bg-indigo-500/20 scale-110' : 'bg-white/5'}
            `}
            >
              <UploadCloud
                className={`w-8 h-8 transition-colors duration-300 ${isDragActive ? 'text-indigo-400' : 'text-slate-400'}`}
              />
            </div>

            {isDragActive ? (
              <div>
                <p className="text-lg font-semibold text-indigo-400">Drop your image here</p>
                <p className="text-sm text-slate-500 mt-1">Release to upload</p>
              </div>
            ) : (
              <div>
                <p className="text-base font-semibold text-slate-300">
                  Drag &amp; drop an image here
                </p>
                <p className="text-sm text-slate-500 mt-1">
                  or{' '}
                  <span className="text-indigo-400 font-medium underline underline-offset-2 cursor-pointer">
                    browse files
                  </span>
                </p>
                <p className="text-xs text-slate-600 mt-3">
                  JPG, PNG, WEBP &nbsp;·&nbsp; Max 10 MB
                </p>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Preview card */
        <div className="rounded-2xl border border-white/10 bg-[#1a1a24] overflow-hidden">
          {/* Image preview */}
          <div className="relative w-full" style={{ maxHeight: '360px' }}>
            <img
              src={preview}
              alt="Selected preview"
              className="w-full object-contain"
              style={{ maxHeight: '360px', background: '#0f0f13' }}
            />
            {/* Remove button */}
            {!loading && (
              <button
                onClick={onRemove}
                className="absolute top-3 right-3 flex items-center justify-center w-8 h-8 rounded-full bg-black/60 border border-white/20 text-slate-300 hover:text-red-400 hover:border-red-400/40 hover:bg-red-500/10 transition-all duration-200"
                title="Remove image"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* File info */}
          <div className="flex items-center gap-3 px-4 py-3 border-t border-white/5">
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-indigo-500/10 flex-shrink-0">
              <ImageIcon className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-slate-200 truncate">{file.name}</p>
              <p className="text-xs text-slate-500">{formatBytes(file.size)}</p>
            </div>
            {!loading && (
              <div {...getRootProps()} className="cursor-pointer">
                <input {...getInputProps()} />
                <button className="text-xs text-indigo-400 hover:text-indigo-300 font-medium transition-colors whitespace-nowrap">
                  Change
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
          <FileWarning className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Analyze button */}
      {file && (
        <button
          onClick={onAnalyze}
          disabled={loading}
          className={`
            w-full relative flex items-center justify-center gap-3 py-4 px-6 rounded-2xl font-semibold text-base
            transition-all duration-300 overflow-hidden
            ${
              loading
                ? 'bg-indigo-600/50 cursor-not-allowed text-white/60'
                : 'bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/50 hover:scale-[1.01] active:scale-[0.99]'
            }
          `}
        >
          {loading ? (
            <>
              {/* Spinner */}
              <svg
                className="animate-spin w-5 h-5 text-white/70"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              <span>Analyzing image...</span>
            </>
          ) : (
            <>
              <span>Analyze Image</span>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
              </svg>
            </>
          )}
        </button>
      )}
    </div>
  )
}

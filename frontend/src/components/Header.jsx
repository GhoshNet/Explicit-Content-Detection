import React from 'react'
import { Shield } from 'lucide-react'

export default function Header() {
  return (
    <header className="relative w-full">
      {/* Main bar */}
      <div className="bg-[#13131d] px-6 py-4 flex items-center justify-between">
        {/* Logo + name */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-lg shadow-indigo-500/30">
            <Shield className="w-5 h-5 text-white" strokeWidth={2.5} />
          </div>
          <div>
            <span className="text-lg font-bold tracking-tight bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
              ContentGuard
            </span>
            <p className="text-xs text-slate-500 leading-none mt-0.5">AI-Powered Content Safety</p>
          </div>
        </div>

        {/* Badge */}
        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs font-medium text-indigo-300">YOLOv9 Active</span>
        </div>
      </div>

      {/* Gradient bottom border */}
      <div className="h-px w-full gradient-border-anim opacity-60" />
    </header>
  )
}

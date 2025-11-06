'use client'

import { useState, useEffect, useRef } from 'react'

interface LogEntry {
  timestamp: string
  level: 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG'
  component: string
  message: string
}

export function LogViewer() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [filter, setFilter] = useState<'ALL' | 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG'>('ALL')
  const [autoScroll, setAutoScroll] = useState(true)
  const [isPaused, setIsPaused] = useState(false)
  const logsEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Simulate log streaming (in real implementation, this would be WebSocket or SSE)
    if (!isPaused) {
      const interval = setInterval(() => {
        const mockLogs: LogEntry[] = [
          {
            timestamp: new Date().toISOString(),
            level: 'INFO',
            component: 'Orchestrator',
            message: 'Query received: What is the derivative of x²?'
          },
          {
            timestamp: new Date().toISOString(),
            level: 'INFO',
            component: 'Router',
            message: 'Selected MATH worker (confidence: 0.95)'
          },
          {
            timestamp: new Date().toISOString(),
            level: 'DEBUG',
            component: 'LATS',
            message: 'Starting tree search with depth=5, sims=10'
          },
          {
            timestamp: new Date().toISOString(),
            level: 'INFO',
            component: 'MathWorker',
            message: 'Executed LATS search in 2.3s'
          },
          {
            timestamp: new Date().toISOString(),
            level: 'INFO',
            component: 'Verification',
            message: 'SymPy verification PASSED'
          }
        ]

        setLogs(prev => [...prev, mockLogs[Math.floor(Math.random() * mockLogs.length)]].slice(-100))
      }, 3000)

      return () => clearInterval(interval)
    }
  }, [isPaused])

  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  const filteredLogs = filter === 'ALL' 
    ? logs 
    : logs.filter(log => log.level === filter)

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'INFO': return 'text-blue-600 dark:text-blue-400'
      case 'WARNING': return 'text-yellow-600 dark:text-yellow-400'
      case 'ERROR': return 'text-red-600 dark:text-red-400'
      case 'DEBUG': return 'text-slate-500 dark:text-slate-400'
      default: return 'text-slate-600 dark:text-slate-300'
    }
  }

  const getLevelBg = (level: string) => {
    switch (level) {
      case 'INFO': return 'bg-blue-100 dark:bg-blue-900/30'
      case 'WARNING': return 'bg-yellow-100 dark:bg-yellow-900/30'
      case 'ERROR': return 'bg-red-100 dark:bg-red-900/30'
      case 'DEBUG': return 'bg-slate-100 dark:bg-slate-800'
      default: return 'bg-slate-100 dark:bg-slate-800'
    }
  }

  const clearLogs = () => {
    setLogs([])
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-linear-to-r from-emerald-600 to-teal-600 rounded-2xl p-8 text-white">
        <h2 className="text-3xl font-bold mb-2">Live System Logs</h2>
        <p className="text-emerald-100">
          Real-time logging from all system components: Router, LATS, Workers, Verification, and Cache.
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          {/* Filter Buttons */}
          <div className="flex flex-wrap gap-2">
            {['ALL', 'INFO', 'WARNING', 'ERROR', 'DEBUG'].map((level) => (
              <button
                key={level}
                onClick={() => setFilter(level as typeof filter)}
                className={`px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                  filter === level
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
                }`}
              >
                {level}
                {level !== 'ALL' && (
                  <span className="ml-2 text-xs opacity-75">
                    ({logs.filter(l => l.level === level).length})
                  </span>
                )}
              </button>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => setAutoScroll(!autoScroll)}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                autoScroll
                  ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                  : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
              }`}
            >
              {autoScroll ? '🔽 Auto-scroll ON' : '⏸️ Auto-scroll OFF'}
            </button>

            <button
              onClick={() => setIsPaused(!isPaused)}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                isPaused
                  ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300'
                  : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
              }`}
            >
              {isPaused ? '▶️ Resume' : '⏸️ Pause'}
            </button>

            <button
              onClick={clearLogs}
              className="px-4 py-2 rounded-lg font-medium text-sm bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors"
            >
              🗑️ Clear
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="mt-4 flex gap-6 text-sm text-slate-600 dark:text-slate-400">
          <span>Total Logs: <strong className="text-slate-900 dark:text-white">{logs.length}</strong></span>
          <span>Filtered: <strong className="text-slate-900 dark:text-white">{filteredLogs.length}</strong></span>
          <span>Status: <strong className={isPaused ? 'text-yellow-600' : 'text-green-600'}>{isPaused ? 'Paused' : 'Live'}</strong></span>
        </div>
      </div>

      {/* Log Display */}
      <div className="bg-slate-900 rounded-2xl shadow-lg p-6 h-[600px] overflow-y-auto font-mono text-sm">
        {filteredLogs.length === 0 ? (
          <div className="text-center py-12 text-slate-500">
            No logs to display. Try running a query or changing the filter.
          </div>
        ) : (
          <div className="space-y-2">
            {filteredLogs.map((log, i) => (
              <div
                key={i}
                className={`${getLevelBg(log.level)} rounded-lg p-3 border-l-4 ${
                  log.level === 'ERROR' ? 'border-red-500' :
                  log.level === 'WARNING' ? 'border-yellow-500' :
                  log.level === 'INFO' ? 'border-blue-500' :
                  'border-slate-500'
                }`}
              >
                <div className="flex items-start gap-3">
                  <span className="text-slate-500 dark:text-slate-400 text-xs whitespace-nowrap">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  <span className={`${getLevelColor(log.level)} font-bold text-xs px-2 py-0.5 rounded uppercase whitespace-nowrap`}>
                    {log.level}
                  </span>
                  <span className="text-purple-600 dark:text-purple-400 text-xs font-semibold whitespace-nowrap">
                    [{log.component}]
                  </span>
                  <span className="text-slate-900 dark:text-slate-100 flex-1">
                    {log.message}
                  </span>
                </div>
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        )}
      </div>

      {/* Info */}
      <div className="bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 rounded-2xl p-6">
        <h3 className="text-lg font-bold text-emerald-900 dark:text-emerald-100 mb-3">📝 Log Components</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm text-emerald-800 dark:text-emerald-200">
          <div>
            <strong>Orchestrator:</strong> Main pipeline coordination
          </div>
          <div>
            <strong>Router:</strong> Neural worker selection decisions
          </div>
          <div>
            <strong>LATS:</strong> Tree search progress and pruning
          </div>
          <div>
            <strong>Workers:</strong> Expert execution (Math, Code, Logic, etc.)
          </div>
          <div>
            <strong>Verification:</strong> SymPy, AST, and semantic checks
          </div>
          <div>
            <strong>Cache:</strong> Similarity lookups and LLM validation
          </div>
          <div>
            <strong>Reflection:</strong> Self-correction analysis
          </div>
          <div>
            <strong>Metrics:</strong> Performance tracking
          </div>
        </div>
      </div>
    </div>
  )
}

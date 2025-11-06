'use client'

import { useState, useRef } from 'react'

interface LogEntry {
  timestamp: string
  level: 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG'
  component: string
  message: string
}

export function LogViewer() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [filter, setFilter] = useState<'ALL' | 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG'>('ALL')
  const logsEndRef = useRef<HTMLDivElement>(null)

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
        </div>
      </div>

      {/* Log Display */}
      <div className="bg-slate-900 rounded-2xl shadow-lg p-6 h-[600px] overflow-y-auto font-mono text-sm">
        {filteredLogs.length === 0 ? (
          <div className="text-center py-12 text-slate-400">
            <div className="text-6xl mb-4">📝</div>
            <p className="text-lg mb-2">No logs available</p>
            <p className="text-sm">Logs will appear here when you run queries through the system.</p>
            <p className="text-xs mt-4 text-slate-500">
              The backend logs all activity to console - this viewer is for future WebSocket/SSE integration.
            </p>
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
        <h3 className="text-lg font-bold text-emerald-900 dark:text-emerald-100 mb-3">📝 Live Logging (Coming Soon)</h3>
        <div className="text-sm text-emerald-800 dark:text-emerald-200 space-y-3">
          <p>
            <strong>Current Status:</strong> All system logs are currently written to the backend console. 
            View them by running <code className="bg-emerald-900/20 px-2 py-0.5 rounded">python backend/app.py</code>
          </p>
          <p>
            <strong>Planned Features:</strong> Real-time log streaming via WebSockets will show:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mt-3">
            <div>• Orchestrator pipeline coordination</div>
            <div>• Router worker selection</div>
            <div>• LATS tree search progress</div>
            <div>• Worker execution details</div>
            <div>• Verification results</div>
            <div>• Cache lookups & validation</div>
            <div>• Reflection self-correction</div>
            <div>• Metrics & performance data</div>
          </div>
          <p className="pt-3 border-t border-emerald-200 dark:border-emerald-700">
            <strong>For now:</strong> Check the backend terminal for detailed logging output while running queries.
          </p>
        </div>
      </div>
    </div>
  )
}

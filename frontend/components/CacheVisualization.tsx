'use client'

import { useState, useEffect } from 'react'

interface CacheStats {
  total_cached: number
  total_validations: number
  validations_accepted: number
  validations_rejected: number
  acceptance_rate: number
  rejection_rate: number
  cache_files: any[]
}

export function CacheVisualization() {
  const [stats, setStats] = useState<CacheStats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchCacheStats()
    const interval = setInterval(fetchCacheStats, 5000)
    return () => clearInterval(interval)
  }, [])

  const fetchCacheStats = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/stats/cache')
      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error('Failed to fetch cache stats:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="text-center py-12">Loading cache stats...</div>
  }

  if (!stats) {
    return <div className="text-center py-12 text-slate-600">No cache data available</div>
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-600 to-emerald-600 rounded-2xl p-8 text-white">
        <h2 className="text-3xl font-bold mb-2">Cache Validation System</h2>
        <p className="text-green-100">
          Two-stage validation: Fast cosine similarity (0.85 threshold) + LLM semantic check. 
          Prevents false positives while maintaining speed.
        </p>
        <div className="mt-4">
          <a 
            href="#" 
            onClick={(e) => { e.preventDefault(); (window as any).setActiveTab?.('trees'); }}
            className="inline-flex items-center gap-2 text-sm bg-white/20 hover:bg-white/30 px-4 py-2 rounded-lg transition-colors"
          >
            🌳 View Reasoning Trees & Cache Contents
          </a>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Cached Trees</div>
          <div className="text-3xl font-bold text-slate-900 dark:text-white">{stats.total_cached}</div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">LLM Validations</div>
          <div className="text-3xl font-bold text-blue-600">{stats.total_validations}</div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Acceptance Rate</div>
          <div className="text-3xl font-bold text-green-600">{(stats.acceptance_rate * 100).toFixed(0)}%</div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Rejection Rate</div>
          <div className="text-3xl font-bold text-orange-600">{(stats.rejection_rate * 100).toFixed(0)}%</div>
        </div>
      </div>

      {/* Validation Breakdown */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">Validation Breakdown</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Accepted */}
          <div className="border-2 border-green-200 dark:border-green-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-green-700 dark:text-green-300">✓ Accepted</h4>
              <span className="text-2xl font-bold text-green-600">{stats.validations_accepted}</span>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Cache hits validated by LLM as semantically equivalent. These queries reused cached LATS trees 
              instead of re-searching, saving computation time.
            </p>
            <div className="mt-4 bg-green-100 dark:bg-green-900/30 rounded-lg p-3">
              <div className="text-xs text-green-800 dark:text-green-200 font-mono">
                Speedup: ~10-50x faster than full LATS search
              </div>
            </div>
          </div>

          {/* Rejected */}
          <div className="border-2 border-orange-200 dark:border-orange-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-orange-700 dark:text-orange-300">✗ Rejected</h4>
              <span className="text-2xl font-bold text-orange-600">{stats.validations_rejected}</span>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Similarity matches rejected by LLM due to semantic differences. These queries proceeded 
              to full LATS search despite high cosine similarity.
            </p>
            <div className="mt-4 bg-orange-100 dark:bg-orange-900/30 rounded-lg p-3">
              <div className="text-xs text-orange-800 dark:text-orange-200 font-mono">
                False positive prevention: Saves incorrect answers
              </div>
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mt-6">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-green-700 dark:text-green-300 font-medium">Accepted</span>
            <span className="text-orange-700 dark:text-orange-300 font-medium">Rejected</span>
          </div>
          <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-4 flex overflow-hidden">
            <div
              className="bg-green-500 flex items-center justify-center text-xs text-white font-medium"
              style={{ width: `${stats.acceptance_rate * 100}%` }}
            >
              {stats.acceptance_rate > 0.1 && `${(stats.acceptance_rate * 100).toFixed(0)}%`}
            </div>
            <div
              className="bg-orange-500 flex items-center justify-center text-xs text-white font-medium"
              style={{ width: `${stats.rejection_rate * 100}%` }}
            >
              {stats.rejection_rate > 0.1 && `${(stats.rejection_rate * 100).toFixed(0)}%`}
            </div>
          </div>
        </div>
      </div>

      {/* Cache Files */}
      {stats.cache_files.length > 0 && (
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-4">Cached LATS Trees</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {stats.cache_files.map((file, i) => (
              <div key={i} className="border border-slate-200 dark:border-slate-700 rounded-lg p-4 hover:border-green-400 dark:hover:border-green-600 transition-colors">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
                    {file.worker.toUpperCase()}
                  </span>
                  <span className="text-xs text-slate-400 dark:text-slate-500">
                    {file.nodes} nodes
                  </span>
                </div>
                <p className="text-sm text-slate-700 dark:text-slate-300 line-clamp-2 mb-2">
                  {file.query}
                </p>
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  ID: {file.cache_id.slice(0, 16)}...
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* How It Works */}
      <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-2xl p-6">
        <h3 className="text-lg font-bold text-green-900 dark:text-green-100 mb-3">🎯 How Smart Caching Works</h3>
        <div className="space-y-2 text-sm text-green-800 dark:text-green-200">
          <p>
            <strong>1. Query Embedding:</strong> New query converted to 384-dim vector using SentenceTransformer 
            (all-MiniLM-L6-v2)
          </p>
          <p>
            <strong>2. Fast Similarity Search:</strong> Cosine similarity computed against all cached embeddings. 
            Matches &ge; 0.85 proceed to validation
          </p>
          <p>
            <strong>3. LLM Semantic Validation:</strong> LLM analyzes both queries to determine semantic equivalence. 
            Considers context, intent, and required reasoning approach
          </p>
          <p>
            <strong>4. Decision:</strong> If validated, reuse cached LATS tree (10-50x speedup). 
            If rejected, perform full search and cache result
          </p>
          <p>
            <strong>5. Learning:</strong> All validation decisions logged to .kaelum/cache_validation/ 
            for future analysis and threshold tuning
          </p>
        </div>
      </div>
    </div>
  )
}

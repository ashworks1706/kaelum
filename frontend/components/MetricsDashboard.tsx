'use client'

import { useState, useEffect } from 'react'

interface Metrics {
  total_queries: number
  total_successes: number
  total_failures: number
  avg_execution_time: number
  avg_nodes_explored: number
  avg_iterations: number
  cache_hit_rate: number
  worker_metrics: Record<string, {
    queries: number
    success_rate: number
    avg_reward: number
    avg_time: number
  }>
  verification_metrics: {
    total_verified: number
    passed: number
    failed: number
    pass_rate: number
  }
  reflection_metrics: {
    total_reflections: number
    avg_iterations: number
    improvement_rate: number
  }
}

export function MetricsDashboard() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchMetrics()
    const interval = setInterval(fetchMetrics, 5000)
    return () => clearInterval(interval)
  }, [])

  const fetchMetrics = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/metrics')
      const data = await response.json()
      setMetrics(data)
    } catch (error) {
      console.error('Failed to fetch metrics:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="text-center py-12">Loading metrics...</div>
  }

  if (!metrics) {
    return <div className="text-center py-12 text-slate-600">No metrics available yet</div>
  }

  const successRate = metrics.total_queries > 0 
    ? (metrics.total_successes / metrics.total_queries) * 100
    : 0

  const workerColors: Record<string, string> = {
    math: 'border-blue-500',
    code: 'border-green-500',
    logic: 'border-purple-500',
    factual: 'border-orange-500',
    creative: 'border-pink-500',
    analysis: 'border-indigo-500',
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-linear-to-r from-indigo-600 to-purple-600 rounded-2xl p-8 text-white">
        <h2 className="text-3xl font-bold mb-2">System Metrics</h2>
        <p className="text-indigo-100">
          Comprehensive analytics across all system components. Real-time updates every 5 seconds.
        </p>
      </div>

      {/* Overall Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Total Queries</div>
          <div className="text-3xl font-bold text-slate-900 dark:text-white">{metrics.total_queries}</div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Success Rate</div>
          <div className="text-3xl font-bold text-green-600">{successRate.toFixed(0)}%</div>
          <div className="text-xs text-slate-500 mt-1">
            {metrics.total_successes} / {metrics.total_queries}
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Avg Time</div>
          <div className="text-3xl font-bold text-blue-600">{metrics.avg_execution_time.toFixed(1)}s</div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Cache Hit Rate</div>
          <div className="text-3xl font-bold text-purple-600">{(metrics.cache_hit_rate * 100).toFixed(0)}%</div>
        </div>
      </div>

      {/* LATS Performance */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">LATS Search Performance</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="border border-slate-200 dark:border-slate-700 rounded-lg p-6">
            <div className="text-sm text-slate-600 dark:text-slate-400 mb-2">Avg Nodes Explored</div>
            <div className="text-4xl font-bold text-blue-600 mb-4">{metrics.avg_nodes_explored.toFixed(1)}</div>
            <div className="text-xs text-slate-500">
              Monte Carlo tree search explores multiple reasoning paths to find optimal solutions
            </div>
          </div>
          
          <div className="border border-slate-200 dark:border-slate-700 rounded-lg p-6">
            <div className="text-sm text-slate-600 dark:text-slate-400 mb-2">Avg Iterations</div>
            <div className="text-4xl font-bold text-purple-600 mb-4">{metrics.avg_iterations.toFixed(1)}</div>
            <div className="text-xs text-slate-500">
              Self-correction cycles through reflection when verification detects issues
            </div>
          </div>
        </div>
      </div>

      {/* Worker Performance */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">Expert Worker Performance</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(metrics.worker_metrics)
            .sort(([, a], [, b]) => b.queries - a.queries)
            .map(([worker, stats]) => (
              <div 
                key={worker} 
                className={`border-l-4 ${workerColors[worker] || 'border-slate-400'} bg-slate-50 dark:bg-slate-900/50 rounded-lg p-5`}
              >
                <h4 className="text-lg font-bold text-slate-900 dark:text-white capitalize mb-3">{worker}</h4>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600 dark:text-slate-400">Queries</span>
                    <span className="font-semibold text-slate-900 dark:text-white">{stats.queries}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600 dark:text-slate-400">Success Rate</span>
                    <span className={`font-semibold ${
                      stats.success_rate > 0.8 ? 'text-green-600' : 
                      stats.success_rate > 0.6 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {(stats.success_rate * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600 dark:text-slate-400">Avg Reward</span>
                    <span className="font-semibold text-blue-600">{stats.avg_reward.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600 dark:text-slate-400">Avg Time</span>
                    <span className="font-semibold text-purple-600">{stats.avg_time.toFixed(1)}s</span>
                  </div>
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Verification & Reflection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Verification */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">Verification Engine</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Total Verified</span>
              <span className="text-2xl font-bold text-slate-900 dark:text-white">
                {metrics.verification_metrics.total_verified}
              </span>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-green-700 dark:text-green-300">✓ Passed</span>
                <span className="font-semibold text-green-600">{metrics.verification_metrics.passed}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-red-700 dark:text-red-300">✗ Failed</span>
                <span className="font-semibold text-red-600">{metrics.verification_metrics.failed}</span>
              </div>
            </div>

            <div className="pt-4 border-t border-slate-200 dark:border-slate-700">
              <div className="text-sm text-slate-600 dark:text-slate-400 mb-2">Pass Rate</div>
              <div className="text-3xl font-bold text-green-600">
                {(metrics.verification_metrics.pass_rate * 100).toFixed(0)}%
              </div>
            </div>

            <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
              <div className="text-xs text-blue-800 dark:text-blue-200">
                Multi-layer verification: SymPy (math), AST (code), semantic coherence, logical consistency
              </div>
            </div>
          </div>
        </div>

        {/* Reflection */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">Reflection System</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Total Reflections</span>
              <span className="text-2xl font-bold text-slate-900 dark:text-white">
                {metrics.reflection_metrics.total_reflections}
              </span>
            </div>

            <div className="pt-4 border-t border-slate-200 dark:border-slate-700">
              <div className="text-sm text-slate-600 dark:text-slate-400 mb-2">Avg Iterations</div>
              <div className="text-3xl font-bold text-purple-600">
                {metrics.reflection_metrics.avg_iterations.toFixed(1)}
              </div>
            </div>

            <div className="pt-4 border-t border-slate-200 dark:border-slate-700">
              <div className="text-sm text-slate-600 dark:text-slate-400 mb-2">Improvement Rate</div>
              <div className="text-3xl font-bold text-orange-600">
                {(metrics.reflection_metrics.improvement_rate * 100).toFixed(0)}%
              </div>
            </div>

            <div className="mt-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
              <div className="text-xs text-purple-800 dark:text-purple-200">
                Self-correction: When verification fails, reflection analyzes issues and regenerates improved solutions
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Success/Failure Breakdown */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">Query Outcomes</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between text-lg">
            <span className="text-green-700 dark:text-green-300 font-medium">✓ Successes</span>
            <span className="text-2xl font-bold text-green-600">{metrics.total_successes}</span>
          </div>
          <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-3">
            <div
              className="bg-green-500 h-3 rounded-full transition-all duration-500"
              style={{ width: `${successRate}%` }}
            />
          </div>

          <div className="flex items-center justify-between text-lg pt-4">
            <span className="text-red-700 dark:text-red-300 font-medium">✗ Failures</span>
            <span className="text-2xl font-bold text-red-600">{metrics.total_failures}</span>
          </div>
          <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-3">
            <div
              className="bg-red-500 h-3 rounded-full transition-all duration-500"
              style={{ width: `${100 - successRate}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

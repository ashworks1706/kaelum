'use client'

import { useState, useEffect } from 'react'

interface RouterStats {
  total_queries: number
  model_trained: boolean
  training_steps: number
  training_buffer_size: number
  next_training_at: string | number
  success_rate: number
  workers_distribution: Record<string, number>
  recent_queries: any[]
  online_learning: {
    enabled: boolean
    learning_rate: number
    buffer_size: number
    exploration_rate: number
    training_steps: number
  }
}

export function RouterVisualization() {
  const [stats, setStats] = useState<RouterStats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchRouterStats()
    const interval = setInterval(fetchRouterStats, 5000)
    return () => clearInterval(interval)
  }, [])

  const fetchRouterStats = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/stats/router')
      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error('Failed to fetch router stats:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="text-center py-12">Loading router stats...</div>
  }

  if (!stats) {
    return <div className="text-center py-12 text-slate-600">No router data available</div>
  }

  const workerColors: Record<string, string> = {
    math: 'bg-blue-500',
    code: 'bg-green-500',
    logic: 'bg-purple-500',
    factual: 'bg-orange-500',
    creative: 'bg-pink-500',
    analysis: 'bg-indigo-500',
  }

  const totalWorkerQueries = Object.values(stats.workers_distribution).reduce((a, b) => a + b, 0)

  return (
    <div className="space-y-6">
      {}
      <div className="bg-linear-to-r from-purple-600 to-indigo-600 rounded-2xl p-8 text-white">
        <h2 className="text-3xl font-bold mb-2">Neural Router Analytics</h2>
        <p className="text-purple-100">
          Deep learning model that learns which expert worker to route each query to.
          Trains automatically after every 32 queries.
        </p>
      </div>

      {}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Total Queries</div>
          <div className="text-3xl font-bold text-slate-900 dark:text-white">{stats.total_queries}</div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Model Status</div>
          <div className="text-3xl font-bold">
            {stats.model_trained ? (
              <span className="text-green-600">✓ Trained</span>
            ) : (
              <span className="text-orange-600">⟳ Learning</span>
            )}
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Training Steps</div>
          <div className="text-3xl font-bold text-purple-600">{stats.training_steps || 0}</div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Success Rate</div>
          <div className="text-3xl font-bold text-blue-600">{(stats.success_rate * 100).toFixed(0)}%</div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Next Training</div>
          <div className="text-2xl font-bold text-indigo-600">
            {typeof stats.next_training_at === 'number'
              ? `${stats.next_training_at} queries`
              : stats.next_training_at}
          </div>
        </div>
      </div>

      {}
      {stats.online_learning && (
        <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 border border-green-200 dark:border-green-800 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-green-900 dark:text-green-100">
              ⚡ Online Learning Active
            </h3>
            <span className="px-3 py-1 bg-green-500 text-white rounded-full text-sm font-medium animate-pulse">
              LIVE
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <div className="text-xs text-slate-600 dark:text-slate-400">Learning Rate</div>
              <div className="text-lg font-bold text-slate-900 dark:text-white">
                {stats.online_learning.learning_rate}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-600 dark:text-slate-400">Buffer Size</div>
              <div className="text-lg font-bold text-slate-900 dark:text-white">
                {stats.online_learning.buffer_size}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-600 dark:text-slate-400">Exploration Rate</div>
              <div className="text-lg font-bold text-slate-900 dark:text-white">
                {(stats.online_learning.exploration_rate * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-600 dark:text-slate-400">Current Buffer</div>
              <div className="text-lg font-bold text-slate-900 dark:text-white">
                {stats.training_buffer_size}/{stats.online_learning.buffer_size}
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2 mt-2">
                <div
                  className="bg-green-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${(stats.training_buffer_size / stats.online_learning.buffer_size) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">Worker Distribution</h3>

        {}
        <div className="space-y-4 mb-8">
          {Object.entries(stats.workers_distribution)
            .sort(([, a], [, b]) => b - a)
            .map(([worker, count]) => {
              const percentage = (count / totalWorkerQueries) * 100
              return (
                <div key={worker}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="font-medium text-slate-700 dark:text-slate-300 capitalize">{worker}</span>
                    <span className="text-slate-600 dark:text-slate-400">{count} queries ({percentage.toFixed(0)}%)</span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-3">
                    <div
                      className={`${workerColors[worker] || 'bg-slate-400'} h-3 rounded-full transition-all duration-500`}
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              )
            })}
        </div>

        {}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {Object.keys(workerColors).map((worker) => (
            <div key={worker} className="flex items-center space-x-2">
              <div className={`w-4 h-4 ${workerColors[worker]} rounded`} />
              <span className="text-sm text-slate-700 dark:text-slate-300 capitalize">{worker}</span>
            </div>
          ))}
        </div>
      </div>

      {}
      {stats.recent_queries.length > 0 && (
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-4">Recent Routing Decisions</h3>
          <div className="space-y-3">
            {stats.recent_queries.map((query, i) => (
              <div key={i} className="border border-slate-200 dark:border-slate-700 rounded-lg p-4">
                <div className="flex items-start justify-between mb-2">
                  <span className={`px-3 py-1 ${workerColors[query.worker] || 'bg-slate-400'} text-white text-sm font-medium rounded-full capitalize`}>
                    {query.worker}
                  </span>
                  <span className={`px-2 py-1 text-xs rounded ${
                    query.success
                      ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                      : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                  }`}>
                    {query.success ? 'Success' : 'Failed'}
                  </span>
                </div>
                <p className="text-sm text-slate-700 dark:text-slate-300 line-clamp-2">{query.query}</p>
                <div className="mt-2 flex items-center space-x-4 text-xs text-slate-500 dark:text-slate-400">
                  <span>Confidence: {(query.confidence * 100).toFixed(0)}%</span>
                  <span>Latency: {query.latency_ms.toFixed(0)}ms</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-2xl p-6">
        <h3 className="text-lg font-bold text-blue-900 dark:text-blue-100 mb-3">🧠 Online Learning Process</h3>
        <div className="space-y-2 text-sm text-blue-800 dark:text-blue-200">
          <p>
            <strong>1. Feature Extraction:</strong> Query → 384-dim embedding + 14 structural features
            (length, math symbols, code keywords, etc.) = 398-dim input vector
          </p>
          <p>
            <strong>2. Neural Network:</strong> PolicyNetwork (398 → 256 → 128) with ReLU + Dropout.
            Outputs: worker probabilities + optimal depth + simulation count + cache decision
          </p>
          <p>
            <strong>3. Exploration:</strong> {stats.online_learning ?
              `${(stats.online_learning.exploration_rate * 100).toFixed(0)}% chance` : '10% chance'} of random
            worker selection for diversity and discovering new patterns
          </p>
          <p>
            <strong>4. Online Training:</strong> After every query, records (worker, success, reward).
            Trains model using Adam optimizer (lr={stats.online_learning?.learning_rate || 0.001}) when buffer reaches {stats.online_learning?.buffer_size || 32} samples
          </p>
          <p>
            <strong>5. Continuous Improvement:</strong> Model automatically saves after each training step.
            Exploration rate gradually decreases (×0.95 every 10 steps, min 5%) as model becomes more confident
          </p>
          <p className="pt-2 border-t border-blue-200 dark:border-blue-700 font-semibold">
            📊 Current Status: {stats.training_steps || 0} training steps completed, learning rate adapts dynamically
          </p>
        </div>
      </div>
    </div>
  )
}

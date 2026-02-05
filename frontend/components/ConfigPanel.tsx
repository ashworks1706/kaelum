'use client'

import { useState, useEffect } from 'react'

interface Config {
  base_url: string
  model: string
  api_key: string
  temperature: number
  max_tokens: number
  embedding_model: string
  use_symbolic_verification: boolean
  use_factual_verification: boolean
  max_reflection_iterations: number
  enable_routing: boolean
  parallel: boolean
  max_workers: number
  router_learning_rate: number
  router_buffer_size: number
  router_exploration_rate: number
}

export function ConfigPanel() {
  const [config, setConfig] = useState<Config | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)

  useEffect(() => {
    fetchConfig()
  }, [])

  const fetchConfig = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/config')
      const data = await response.json()
      setConfig(data)
    } catch (error) {
      console.error('Failed to fetch config:', error)
      setMessage({ type: 'error', text: 'Failed to load configuration' })
    } finally {
      setLoading(false)
    }
  }

  const saveConfig = async () => {
    if (!config) return

    setSaving(true)
    setMessage(null)

    try {
      const response = await fetch('http://localhost:5000/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })

      if (response.ok) {
        setMessage({ type: 'success', text: 'Configuration saved successfully! System reinitialized.' })
      } else {
        throw new Error('Failed to save configuration')
      }
    } catch (error: any) {
      setMessage({ type: 'error', text: error.message || 'Failed to save configuration' })
    } finally {
      setSaving(false)
    }
  }

  const updateConfig = (key: string, value: any) => {
    if (config) {
      setConfig({ ...config, [key]: value })
    }
  }

  if (loading) {
    return <div className="text-center py-12">Loading configuration...</div>
  }

  if (!config) {
    return <div className="text-center py-12 text-red-600">Failed to load configuration</div>
  }

  return (
    <div className="space-y-6">
      {}
      <div className="bg-linear-to-r from-indigo-600 to-purple-600 rounded-2xl p-8 text-white">
        <h2 className="text-3xl font-bold mb-2">System Configuration</h2>
        <p className="text-indigo-100">
          Configure LLM backend, routing, verification, and search parameters.
          Changes require system reinitialization.
        </p>
      </div>

      {}
      {message && (
        <div className={`rounded-xl p-4 ${
          message.type === 'success'
            ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-800 dark:text-green-200'
            : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200'
        }`}>
          {message.text}
        </div>
      )}

      {}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">🤖 LLM Backend</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Base URL
            </label>
            <input
              type="text"
              value={config.base_url}
              onChange={(e) => updateConfig('base_url', e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
              placeholder="http://localhost:8000/v1"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              vLLM/Ollama/OpenAI compatible endpoint
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Model Name
            </label>
            <input
              type="text"
              value={config.model}
              onChange={(e) => updateConfig('model', e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
              placeholder="Qwen/Qwen2.5-7B-Instruct"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Model identifier on the backend
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Temperature
            </label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={config.temperature}
              onChange={(e) => updateConfig('temperature', parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400">
              <span>Deterministic (0.0)</span>
              <span className="font-semibold">{config.temperature.toFixed(1)}</span>
              <span>Creative (2.0)</span>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Max Tokens
            </label>
            <input
              type="number"
              min="128"
              max="8192"
              value={config.max_tokens}
              onChange={(e) => updateConfig('max_tokens', parseInt(e.target.value))}
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Maximum output length per generation
            </p>
          </div>
        </div>
      </div>

      {}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">🧠 Neural Router & Embeddings</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Embedding Model
            </label>
            <select
              value={config.embedding_model}
              onChange={(e) => updateConfig('embedding_model', e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
            >
              <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2 (Fast, 384-dim)</option>
              <option value="all-mpnet-base-v2">all-mpnet-base-v2 (Accurate, 768-dim)</option>
              <option value="all-MiniLM-L12-v2">all-MiniLM-L12-v2 (Balanced, 384-dim)</option>
            </select>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Used for cache similarity and router features
            </p>
          </div>

          <div>
            <label className="flex items-center space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={config.enable_routing}
                onChange={(e) => updateConfig('enable_routing', e.target.checked)}
                className="w-5 h-5 text-blue-600 border-slate-300 rounded focus:ring-blue-500"
              />
              <div>
                <div className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  Enable Neural Router
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  AI learns optimal worker selection
                </div>
              </div>
            </label>
          </div>
        </div>

        {}
        {config.enable_routing && (
          <div className="mt-6 pt-6 border-t border-slate-200 dark:border-slate-700">
            <h4 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">⚡ Online Learning</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Learning Rate
                </label>
                <input
                  type="number"
                  min="0.0001"
                  max="0.01"
                  step="0.0001"
                  value={config.router_learning_rate}
                  onChange={(e) => updateConfig('router_learning_rate', parseFloat(e.target.value))}
                  className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
                />
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  How fast the router learns (0.001 = default)
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Buffer Size
                </label>
                <input
                  type="number"
                  min="8"
                  max="128"
                  step="8"
                  value={config.router_buffer_size}
                  onChange={(e) => updateConfig('router_buffer_size', parseInt(e.target.value))}
                  className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
                />
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  Queries before model update (32 = default)
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Exploration Rate
                </label>
                <input
                  type="number"
                  min="0.0"
                  max="0.5"
                  step="0.05"
                  value={config.router_exploration_rate}
                  onChange={(e) => updateConfig('router_exploration_rate', parseFloat(e.target.value))}
                  className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
                />
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  Random exploration for diversity (0.1 = 10%)
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">✅ Verification & Reflection</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <label className="flex items-center space-x-3 cursor-pointer">
            <input
              type="checkbox"
              checked={config.use_symbolic_verification}
              onChange={(e) => updateConfig('use_symbolic_verification', e.target.checked)}
              className="w-5 h-5 text-blue-600 border-slate-300 rounded focus:ring-blue-500"
            />
            <div>
              <div className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Symbolic Verification (SymPy)
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400">
                Math equation checking
              </div>
            </div>
          </label>

          <label className="flex items-center space-x-3 cursor-pointer">
            <input
              type="checkbox"
              checked={config.use_factual_verification}
              onChange={(e) => updateConfig('use_factual_verification', e.target.checked)}
              className="w-5 h-5 text-blue-600 border-slate-300 rounded focus:ring-blue-500"
            />
            <div>
              <div className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Factual Verification
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400">
                Semantic coherence checks
              </div>
            </div>
          </label>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Max Reflection Iterations
            </label>
            <input
              type="number"
              min="0"
              max="5"
              value={config.max_reflection_iterations}
              onChange={(e) => updateConfig('max_reflection_iterations', parseInt(e.target.value))}
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Self-correction retries (0 = disabled)
            </p>
          </div>
        </div>
      </div>

      {}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">🌳 LATS Search Settings</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <label className="flex items-center space-x-3 cursor-pointer">
            <input
              type="checkbox"
              checked={config.parallel}
              onChange={(e) => updateConfig('parallel', e.target.checked)}
              className="w-5 h-5 text-blue-600 border-slate-300 rounded focus:ring-blue-500"
            />
            <div>
              <div className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Parallel LATS Search
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400">
                Faster search using multiple workers
              </div>
            </div>
          </label>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Max Workers (if parallel)
            </label>
            <input
              type="number"
              min="1"
              max="16"
              value={config.max_workers}
              onChange={(e) => updateConfig('max_workers', parseInt(e.target.value))}
              disabled={!config.parallel}
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Number of parallel simulation workers
            </p>
          </div>
        </div>
      </div>

      {}
      <div className="flex justify-end space-x-4">
        <button
          onClick={fetchConfig}
          disabled={saving}
          className="px-6 py-3 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg font-medium hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Reset
        </button>
        <button
          onClick={saveConfig}
          disabled={saving}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {saving ? 'Saving...' : 'Save Configuration'}
        </button>
      </div>

      {}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-2xl p-6">
        <h3 className="text-lg font-bold text-blue-900 dark:text-blue-100 mb-3">ℹ️ Configuration Notes</h3>
        <ul className="space-y-2 text-sm text-blue-800 dark:text-blue-200">
          <li>
            <strong>Model Changes:</strong> Ensure vLLM/Ollama is running with the specified model before saving.
          </li>
          <li>
            <strong>Router:</strong> Disabling routing will use default logic worker for all queries.
          </li>
          <li>
            <strong>Verification:</strong> Disabling verification increases speed but reduces accuracy guarantees.
          </li>
          <li>
            <strong>Reflection:</strong> Set to 0 to disable self-correction for faster (but less accurate) results.
          </li>
          <li>
            <strong>Parallel Search:</strong> Only enable if you have sufficient CPU/GPU resources.
          </li>
        </ul>
      </div>
    </div>
  )
}

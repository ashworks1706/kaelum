'use client'

import { useState } from 'react'

export function FineTuningPanel() {
  const [selectedStrategy, setSelectedStrategy] = useState('mixed')
  const [batchSize, setBatchSize] = useState(20)
  const [exportPath, setExportPath] = useState('/tmp/kaelum_training.jsonl')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)

  const strategies = [
    {
      id: 'uncertainty',
      name: 'Uncertainty Sampling',
      description: 'Selects queries where model had lowest confidence',
      icon: '🎲',
      useCase: 'Improve weak areas'
    },
    {
      id: 'diversity',
      name: 'Diversity Sampling',
      description: 'Selects semantically diverse queries',
      icon: '🌈',
      useCase: 'Broad coverage'
    },
    {
      id: 'error',
      name: 'Error-Based Sampling',
      description: 'Prioritizes queries that failed verification',
      icon: '❌',
      useCase: 'Learn from mistakes'
    },
    {
      id: 'complexity',
      name: 'Complexity Sampling',
      description: 'Selects queries with highest reasoning complexity',
      icon: '🧩',
      useCase: 'Hard examples'
    },
    {
      id: 'mixed',
      name: 'Mixed Strategy',
      description: 'Balanced combination of all strategies',
      icon: '🎯',
      useCase: 'Recommended'
    }
  ]

  const exportTrainingData = async () => {
    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch('http://localhost:5000/api/export/training-data')
      const data = await response.json()

      if (data.status === 'exported') {
        setMessage({
          type: 'success',
          text: `Successfully exported ${data.count} training examples to ${data.path}`
        })
      } else {
        throw new Error('Export failed')
      }
    } catch (error) {
      setMessage({
        type: 'error',
        text: 'Failed to export training data. Ensure queries have been executed.'
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-linear-to-r from-purple-600 to-pink-600 rounded-2xl p-8 text-white">
        <h2 className="text-3xl font-bold mb-2">Fine-tuning Data Collection</h2>
        <p className="text-purple-100">
          Intelligently select queries for model fine-tuning. Export high-quality reasoning traces 
          to train specialized worker models.
        </p>
      </div>

      {/* Message Display */}
      {message && (
        <div className={`rounded-xl p-4 ${
          message.type === 'success'
            ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-800 dark:text-green-200'
            : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200'
        }`}>
          {message.text}
        </div>
      )}

      {/* Strategy Selection */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">🎯 Selection Strategy</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {strategies.map((strategy) => (
            <button
              key={strategy.id}
              onClick={() => setSelectedStrategy(strategy.id)}
              className={`text-left p-5 rounded-xl border-2 transition-all ${
                selectedStrategy === strategy.id
                  ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                  : 'border-slate-200 dark:border-slate-700 hover:border-purple-300 dark:hover:border-purple-700'
              }`}
            >
              <div className="text-3xl mb-2">{strategy.icon}</div>
              <h4 className="font-bold text-slate-900 dark:text-white mb-1">{strategy.name}</h4>
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">{strategy.description}</p>
              <span className="inline-block px-2 py-1 text-xs bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-full">
                {strategy.useCase}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Configuration */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">⚙️ Export Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Batch Size
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Number of queries to select (1-100)
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Export Path
            </label>
            <input
              type="text"
              value={exportPath}
              onChange={(e) => setExportPath(e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              File path for exported training data
            </p>
          </div>
        </div>

        <button
          onClick={exportTrainingData}
          disabled={loading}
          className="mt-6 px-6 py-3 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Exporting...' : '📦 Export Training Data'}
        </button>
      </div>

      {/* Fine-tuning Workflow */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">🔄 Fine-tuning Workflow</h3>
        <div className="space-y-4">
          <div className="flex items-start gap-4 p-4 bg-slate-50 dark:bg-slate-900/50 rounded-lg">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">
              1
            </div>
            <div>
              <h4 className="font-semibold text-slate-900 dark:text-white mb-1">Run Queries</h4>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Execute diverse queries through the system. The active learning engine automatically collects high-quality traces.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-4 p-4 bg-slate-50 dark:bg-slate-900/50 rounded-lg">
            <div className="flex-shrink-0 w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center font-bold">
              2
            </div>
            <div>
              <h4 className="font-semibold text-slate-900 dark:text-white mb-1">Select Strategy & Export</h4>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Choose selection strategy above and export training data. File will contain instruction-tuning format with queries and reasoning traces.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-4 p-4 bg-slate-50 dark:bg-slate-900/50 rounded-lg">
            <div className="flex-shrink-0 w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">
              3
            </div>
            <div>
              <h4 className="font-semibold text-slate-900 dark:text-white mb-1">Fine-tune Model</h4>
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                Use the exported data to fine-tune your model. Run the fine-tuning script:
              </p>
              <code className="block p-3 bg-slate-900 text-green-400 rounded text-xs font-mono overflow-x-auto">
                python finetune_setup.py --model Qwen/Qwen2.5-3B-Instruct \<br/>
                {'  '}--domain math --epochs 3 --batch-size 4 \<br/>
                {'  '}--output-dir ./finetuned_models/qwen-math
              </code>
            </div>
          </div>

          <div className="flex items-start gap-4 p-4 bg-slate-50 dark:bg-slate-900/50 rounded-lg">
            <div className="flex-shrink-0 w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">
              4
            </div>
            <div>
              <h4 className="font-semibold text-slate-900 dark:text-white mb-1">Deploy & Test</h4>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Deploy the fine-tuned model with vLLM and update configuration to use the new model endpoint.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Available Scripts */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">📜 Available Scripts</h3>
        <div className="space-y-4">
          <div>
            <h4 className="font-semibold text-slate-900 dark:text-white mb-2">finetune_setup.py</h4>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
              Main fine-tuning script using HuggingFace Transformers. Supports domain-specific training.
            </p>
            <code className="block p-3 bg-slate-900 text-green-400 rounded text-xs font-mono overflow-x-auto">
              python finetune_setup.py --help
            </code>
          </div>

          <div>
            <h4 className="font-semibold text-slate-900 dark:text-white mb-2">export_cache_validation_data.py</h4>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
              Export cache validation logs for training semantic validation models.
            </p>
            <code className="block p-3 bg-slate-900 text-green-400 rounded text-xs font-mono overflow-x-auto">
              python export_cache_validation_data.py --output validation_training.jsonl
            </code>
          </div>
        </div>
      </div>

      {/* Info */}
      <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-2xl p-6">
        <h3 className="text-lg font-bold text-purple-900 dark:text-purple-100 mb-3">💡 Fine-tuning Tips</h3>
        <ul className="space-y-2 text-sm text-purple-800 dark:text-purple-200">
          <li>
            <strong>Start Small:</strong> Begin with 3B-7B models (Qwen, Phi-3, Llama-3.2) for faster iteration.
          </li>
          <li>
            <strong>Domain Specialization:</strong> Use <code>--domain math</code> to train domain-specific experts.
          </li>
          <li>
            <strong>Quality Over Quantity:</strong> The active learning engine already filters for high-reward traces.
          </li>
          <li>
            <strong>Continuous Learning:</strong> Export and fine-tune periodically as you collect more data.
          </li>
          <li>
            <strong>LoRA/QLoRA:</strong> Consider parameter-efficient fine-tuning for larger models (14B+).
          </li>
        </ul>
      </div>
    </div>
  )
}

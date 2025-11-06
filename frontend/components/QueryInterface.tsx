'use client'

import { useState, useEffect, useRef } from 'react'

interface StreamEvent {
  type: 'status' | 'router' | 'reasoning_step' | 'answer' | 'verification' | 'done' | 'error' | 'log'
  message?: string
  worker?: string
  confidence?: number
  index?: number
  content?: string
  passed?: boolean
  execution_time?: number
  cache_hit?: boolean
  iterations?: number
  metadata?: Record<string, unknown>
  // Log-specific fields
  level?: string
  logger?: string
  timestamp?: number
}

interface LogEntry {
  timestamp: string
  level: 'info' | 'error' | 'success'
  message: string
}

interface QueryResult {
  query: string
  answer: string
  reasoning_steps: string[]
  worker: string
  confidence: number
  verification_passed: boolean
  execution_time: number
  cache_hit: boolean
  iterations: number
  metadata: Record<string, unknown>
}

export function QueryInterface() {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState<QueryResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [streamingStatus, setStreamingStatus] = useState<string>('')
  const [streamingSteps, setStreamingSteps] = useState<string[]>([])
  const [logs, setLogs] = useState<LogEntry[]>([])
  const logsEndRef = useRef<HTMLDivElement>(null)

  const addLog = (level: 'info' | 'error' | 'success', message: string) => {
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prev => [...prev, { timestamp, level, message }])
  }

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const exampleQueries = [
    "What is the derivative of x² + 3x?",
    "Write a Python function to find prime numbers",
    "Explain the logical fallacy in: All dogs bark. My cat barks. Therefore my cat is a dog.",
    "What are the benefits of renewable energy?",
    "Write a creative story about a robot learning emotions",
  ]

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)
    setStreamingStatus('')
    setStreamingSteps([])
    setLogs([]) // Clear previous logs

    addLog('info', `Starting query: "${query.substring(0, 50)}${query.length > 50 ? '...' : ''}"`)

    try {
      addLog('info', 'Connecting to backend at http://localhost:5000/api/query')
      
      const response = await fetch('http://localhost:5000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, stream: true })
      })

      addLog('info', `Response status: ${response.status} ${response.statusText}`)

      if (!response.ok) {
        const errorText = await response.text()
        addLog('error', `API request failed: ${errorText}`)
        throw new Error(`API request failed: ${response.status} ${response.statusText}`)
      }

      // Handle Server-Sent Events
      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) throw new Error('No response body')

      addLog('success', 'SSE stream started, receiving events...')

      const partialResult: QueryResult = {
        query,
        answer: '',
        reasoning_steps: [],
        worker: 'unknown',
        confidence: 0,
        verification_passed: false,
        cache_hit: false,
        iterations: 1,
        execution_time: 0,
        metadata: {}
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          addLog('success', 'Stream completed')
          break
        }

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const eventData: StreamEvent = JSON.parse(line.slice(6))

              switch (eventData.type) {
                case 'log':
                  // Handle real-time logs from backend
                  if (eventData.message) {
                    const logLevel = eventData.level?.toLowerCase()
                    const level = logLevel === 'info' ? 'info' : logLevel === 'error' ? 'error' : 'success'
                    const loggerName = eventData.logger ? `[${eventData.logger}]` : ''
                    addLog(level, `${loggerName} ${eventData.message}`)
                  }
                  break

                case 'status':
                  setStreamingStatus(eventData.message || '')
                  addLog('info', `Status: ${eventData.message}`)
                  break

                case 'router':
                  partialResult.worker = eventData.worker || 'unknown'
                  partialResult.confidence = eventData.confidence || 0
                  setStreamingStatus(`Routing to ${eventData.worker} worker...`)
                  addLog('success', `Routed to ${eventData.worker} worker (confidence: ${((eventData.confidence || 0) * 100).toFixed(0)}%)`)
                  break

                case 'reasoning_step':
                  if (eventData.content) {
                    setStreamingSteps(prev => [...prev, eventData.content!])
                    partialResult.reasoning_steps.push(eventData.content)
                    addLog('info', `Reasoning step ${(eventData.index || 0) + 1}: ${eventData.content.substring(0, 60)}...`)
                  }
                  break

                case 'answer':
                  partialResult.answer = eventData.content || ''
                  setStreamingStatus('Generating answer...')
                  addLog('success', `Answer generated (${eventData.content?.length || 0} chars)`)
                  break

                case 'verification':
                  partialResult.verification_passed = eventData.passed || false
                  setStreamingStatus('Verifying answer...')
                  addLog(eventData.passed ? 'success' : 'error', `Verification ${eventData.passed ? 'PASSED' : 'FAILED'}`)
                  break

                case 'done':
                  partialResult.execution_time = eventData.execution_time || 0
                  partialResult.cache_hit = eventData.cache_hit || false
                  partialResult.iterations = eventData.iterations || 1
                  partialResult.metadata = eventData.metadata || {}
                  setResult(partialResult)
                  setStreamingStatus('Complete!')
                  addLog('success', `Query completed in ${(eventData.execution_time || 0).toFixed(2)}s`)
                  break

                case 'error':
                  addLog('error', `Error: ${eventData.message}`)
                  throw new Error(eventData.message || 'Unknown error')
              }
            } catch (parseError) {
              addLog('error', `Failed to parse SSE event: ${parseError}`)
            }
          }
        }
      }
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to process query'
      setError(errorMsg)
      addLog('error', errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl p-8 text-white">
        <h2 className="text-3xl font-bold mb-2">Try the AI Reasoning System</h2>
        <p className="text-blue-100">
          Ask anything! The system automatically routes to the right expert (Math, Code, Logic, etc.), 
          explores multiple solution paths with LATS, and verifies answers.
        </p>
      </div>

      {/* Query Input */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Your Question
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What would you like to know?"
              className="w-full px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white resize-none"
              rows={3}
            />
          </div>

          <div className="flex items-center space-x-4">
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  {streamingStatus || 'Processing...'}
                </span>
              ) : (
                'Ask Kaelum'
              )}
            </button>

            {result && (
              <button
                type="button"
                onClick={() => {
                  setQuery('')
                  setResult(null)
                  setError(null)
                }}
                className="px-6 py-3 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg font-medium hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
              >
                Clear
              </button>
            )}
          </div>
        </form>

        {/* Example Queries */}
        <div className="mt-6">
          <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">Example queries:</p>
          <div className="flex flex-wrap gap-2">
            {exampleQueries.map((example, i) => (
              <button
                key={i}
                onClick={() => setQuery(example)}
                className="px-3 py-1.5 text-sm bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-full hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors"
              >
                {example.substring(0, 40)}...
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4">
          <p className="text-red-800 dark:text-red-200 font-medium">Error: {error}</p>
        </div>
      )}

      {/* Streaming Progress */}
      {loading && streamingSteps.length > 0 && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-6">
          <h3 className="text-lg font-bold text-blue-900 dark:text-blue-100 mb-4">
            🔄 {streamingStatus}
          </h3>
          <div className="space-y-2">
            {streamingSteps.map((step, i) => (
              <div key={i} className="flex items-start animate-fadeIn">
                <div className="flex-shrink-0 w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-semibold mr-3">
                  {i + 1}
                </div>
                <p className="text-blue-800 dark:text-blue-200 text-sm pt-0.5">{step}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="space-y-4">
          {/* Metadata Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow">
              <div className="text-sm text-slate-600 dark:text-slate-400">Worker</div>
              <div className="text-lg font-bold text-blue-600 dark:text-blue-400 capitalize">{result.worker}</div>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow">
              <div className="text-sm text-slate-600 dark:text-slate-400">Confidence</div>
              <div className="text-lg font-bold text-green-600 dark:text-green-400">{(result.confidence * 100).toFixed(0)}%</div>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow">
              <div className="text-sm text-slate-600 dark:text-slate-400">Verification</div>
              <div className="text-lg font-bold">
                {result.verification_passed ? (
                  <span className="text-green-600 dark:text-green-400">✓ Passed</span>
                ) : (
                  <span className="text-red-600 dark:text-red-400">✗ Failed</span>
                )}
              </div>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow">
              <div className="text-sm text-slate-600 dark:text-slate-400">Time</div>
              <div className="text-lg font-bold text-indigo-600 dark:text-indigo-400">{result.execution_time.toFixed(2)}s</div>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow">
              <div className="text-sm text-slate-600 dark:text-slate-400">Cache</div>
              <div className="text-lg font-bold">
                {result.cache_hit ? (
                  <span className="text-purple-600 dark:text-purple-400">⚡ Hit</span>
                ) : (
                  <span className="text-slate-400">Miss</span>
                )}
              </div>
            </div>
          </div>

          {/* Answer */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
            <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-4">Answer</h3>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 whitespace-pre-wrap">{result.answer}</p>
            </div>
          </div>

          {/* Reasoning Steps */}
          {result.reasoning_steps && result.reasoning_steps.length > 0 && (
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-4">
                Reasoning Steps ({result.reasoning_steps.length})
              </h3>
              <div className="space-y-3">
                {result.reasoning_steps.map((step: string, i: number) => (
                  <div key={i} className="flex">
                    <div className="flex-shrink-0 w-8 h-8 bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300 rounded-full flex items-center justify-center font-semibold mr-3">
                      {i + 1}
                    </div>
                    <p className="text-slate-700 dark:text-slate-300 pt-1">{step}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Iterations Info */}
          {result.iterations > 1 && (
            <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl p-4">
              <p className="text-amber-800 dark:text-amber-200">
                <span className="font-semibold">Self-Correction:</span> This answer required {result.iterations} iteration(s) 
                to pass verification. The system automatically improved its reasoning through reflection.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Live Logs Panel */}
      {logs.length > 0 && (
        <div className="mt-8 bg-white dark:bg-gray-800 rounded-xl shadow-md overflow-hidden">
          <div className="bg-gray-100 dark:bg-gray-700 px-6 py-3 border-b border-gray-200 dark:border-gray-600">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              System Logs ({logs.length})
            </h3>
          </div>
          <div className="p-4 max-h-96 overflow-y-auto bg-gray-50 dark:bg-gray-900">
            {logs.map((log, index) => (
              <div
                key={index}
                className={`flex items-start py-2 px-3 rounded mb-2 text-sm font-mono ${
                  log.level === 'error'
                    ? 'bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-300'
                    : log.level === 'success'
                    ? 'bg-green-50 dark:bg-green-900/20 text-green-800 dark:text-green-300'
                    : 'bg-blue-50 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300'
                }`}
              >
                <span className="opacity-60 mr-3 shrink-0">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </span>
                <span className="break-all">{log.message}</span>
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}
    </div>
  )
}

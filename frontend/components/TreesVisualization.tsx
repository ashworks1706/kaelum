'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { useState, useEffect, useRef, useCallback } from 'react'

interface TreeNode {
  id: string
  query: string
  children: TreeNode[]
  visits: number
  total_reward: number
  avg_reward: number
  is_pruned: boolean
  is_best_path: boolean
  depth: number
  worker_type?: string
}

interface ReasoningTree {
  tree_id: string
  query: string
  worker: string
  timestamp: string
  root: TreeNode
  best_path: string[]
  total_nodes: number
  pruned_nodes: number
  max_depth: number
  avg_reward: number
  cache_status: 'high' | 'low' | 'none'
  execution_time: number
  verification_passed: boolean
}

interface CacheStats {
  total_trees: number
  high_quality: number
  low_quality: number
  hit_rate: number
  avg_similarity: number
}

export function TreesVisualization() {
  const API_BASE = (process.env.NEXT_PUBLIC_API_BASE as string) || (typeof window !== 'undefined'
    ? (window.location.hostname === 'localhost' ? `${window.location.protocol}//localhost:5000` : window.location.origin)
    : 'http://localhost:5000')

  const [trees, setTrees] = useState<ReasoningTree[]>([])
  const [selectedTree, setSelectedTree] = useState<ReasoningTree | null>(null)
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null)
  const [filterWorker, setFilterWorker] = useState<string>('all')
  const [filterQuality, setFilterQuality] = useState<string>('all')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Fetch trees and cache stats
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true)
        
        // Fetch reasoning trees
        const treesResponse = await fetch(`${API_BASE}/api/trees`)
        if (treesResponse.ok) {
          const treesData = await treesResponse.json()
          setTrees(treesData.trees || [])
        }

        // Fetch cache statistics
        const cacheResponse = await fetch(`${API_BASE}/api/stats/cache`)
        if (cacheResponse.ok) {
          const cacheData = await cacheResponse.json()
          setCacheStats(cacheData)
        }
      } catch (error) {
        console.error('Failed to fetch trees data:', error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()

    // Auto-refresh every 5 seconds if enabled
    if (autoRefresh) {
      const interval = setInterval(fetchData, 5000)
      return () => clearInterval(interval)
    }
  }, [API_BASE, autoRefresh])

  // Filter trees
  const filteredTrees = trees.filter(tree => {
    if (filterWorker !== 'all' && tree.worker !== filterWorker) return false
    if (filterQuality !== 'all') {
      if (filterQuality === 'high' && tree.cache_status !== 'high') return false
      if (filterQuality === 'low' && tree.cache_status !== 'low') return false
      if (filterQuality === 'cached' && tree.cache_status === 'none') return false
    }
    return true
  })

  // Worker colors
  const getWorkerColor = (worker: string) => {
    const colors: Record<string, string> = {
      'math': '#ef4444', // red
      'logic': '#a855f7', // purple
      'code': '#3b82f6', // blue
      'factual': '#22c55e', // green
      'creative': '#ec4899', // pink
      'analysis': '#f97316', // orange
    }
    return colors[worker.toLowerCase()] || '#6366f1'
  }

  // Get node color based on state
  const getNodeColor = (node: TreeNode, isSelected: boolean) => {
    if (node.is_pruned) return '#7f1d1d' // dark red for pruned
    if (node.is_best_path) return '#fbbf24' // gold for best path
    if (isSelected) return '#06b6d4' // cyan for selected
    
    // Color based on reward (gradient from red to green)
    const reward = node.avg_reward
    if (reward < 0.3) return '#991b1b' // dark red
    if (reward < 0.5) return '#ea580c' // orange
    if (reward < 0.7) return '#eab308' // yellow
    if (reward < 0.85) return '#84cc16' // lime
    return '#16a34a' // green
  }

  // Draw tree on canvas
  const drawTree = useCallback((tree: ReasoningTree) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Calculate layout
    const nodeRadius = 20
    const levelHeight = 80
    const horizontalSpacing = 60

    const drawNode = (node: TreeNode, x: number, y: number, parentX?: number, parentY?: number) => {
      // Draw connection to parent
      if (parentX !== undefined && parentY !== undefined) {
        ctx.beginPath()
        ctx.moveTo(parentX, parentY)
        ctx.lineTo(x, y)
        ctx.strokeStyle = node.is_best_path ? '#fbbf24' : node.is_pruned ? '#7f1d1d' : '#475569'
        ctx.lineWidth = node.is_best_path ? 3 : 2
        ctx.stroke()
      }

      // Draw node circle
      ctx.beginPath()
      ctx.arc(x, y, nodeRadius, 0, 2 * Math.PI)
      ctx.fillStyle = getNodeColor(node, false)
      ctx.fill()
      ctx.strokeStyle = node.is_best_path ? '#fbbf24' : '#1e293b'
      ctx.lineWidth = node.is_best_path ? 3 : 2
      ctx.stroke()

      // Draw visit count
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 12px sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(node.visits.toString(), x, y)

      // Draw reward below node
      ctx.fillStyle = '#94a3b8'
      ctx.font = '10px sans-serif'
      ctx.fillText(node.avg_reward.toFixed(2), x, y + nodeRadius + 12)

      // Draw pruned indicator
      if (node.is_pruned) {
        ctx.fillStyle = '#ef4444'
        ctx.font = 'bold 16px sans-serif'
        ctx.fillText('✗', x + nodeRadius + 5, y - nodeRadius - 5)
      }

      // Draw best path indicator
      if (node.is_best_path) {
        ctx.fillStyle = '#fbbf24'
        ctx.font = 'bold 16px sans-serif'
        ctx.fillText('★', x - nodeRadius - 5, y - nodeRadius - 5)
      }

      // Recursively draw children
      if (node.children && node.children.length > 0) {
        const totalWidth = (node.children.length - 1) * horizontalSpacing
        const startX = x - totalWidth / 2

        node.children.forEach((child, i) => {
          const childX = startX + i * horizontalSpacing
          const childY = y + levelHeight
          drawNode(child, childX, childY, x, y)
        })
      }
    }

    // Start drawing from root
    if (tree.root) {
      const startX = canvas.width / 2
      const startY = 50
      drawNode(tree.root, startX, startY)
    }
  }, [])

  // Animate tree drawing when selected
  useEffect(() => {
    if (selectedTree) {
      drawTree(selectedTree)
    }
  }, [selectedTree, drawTree])

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold text-white mb-2 flex items-center gap-3">
          <span className="text-5xl">🌳</span>
          Reasoning Trees
        </h1>
        <p className="text-slate-400">
          Visualize LATS search trees, rewards, and cached results
        </p>
      </motion.div>

      {/* Cache Statistics */}
      {cacheStats && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8"
        >
          <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4">
            <div className="text-slate-400 text-sm mb-1">Total Trees</div>
            <div className="text-3xl font-bold text-white">{cacheStats.total_trees}</div>
          </div>
          <div className="bg-emerald-900/30 backdrop-blur border border-emerald-700 rounded-xl p-4">
            <div className="text-emerald-400 text-sm mb-1">High Quality</div>
            <div className="text-3xl font-bold text-emerald-400">{cacheStats.high_quality}</div>
          </div>
          <div className="bg-amber-900/30 backdrop-blur border border-amber-700 rounded-xl p-4">
            <div className="text-amber-400 text-sm mb-1">Low Quality</div>
            <div className="text-3xl font-bold text-amber-400">{cacheStats.low_quality}</div>
          </div>
          <div className="bg-blue-900/30 backdrop-blur border border-blue-700 rounded-xl p-4">
            <div className="text-blue-400 text-sm mb-1">Hit Rate</div>
            <div className="text-3xl font-bold text-blue-400">{(cacheStats.hit_rate * 100).toFixed(1)}%</div>
          </div>
          <div className="bg-purple-900/30 backdrop-blur border border-purple-700 rounded-xl p-4">
            <div className="text-purple-400 text-sm mb-1">Avg Similarity</div>
            <div className="text-3xl font-bold text-purple-400">{(cacheStats.avg_similarity * 100).toFixed(1)}%</div>
          </div>
        </motion.div>
      )}

      {/* Controls */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4 mb-6 flex flex-wrap gap-4 items-center"
      >
        {/* Worker Filter */}
        <div className="flex items-center gap-2">
          <label className="text-slate-400 text-sm">Worker:</label>
          <select
            value={filterWorker}
            onChange={(e) => setFilterWorker(e.target.value)}
            className="bg-slate-700 text-white px-3 py-2 rounded-lg text-sm border border-slate-600 focus:border-indigo-500 focus:outline-none"
          >
            <option value="all">All Workers</option>
            <option value="math">➗ Math</option>
            <option value="logic">🧠 Logic</option>
            <option value="code">💻 Code</option>
            <option value="factual">📚 Factual</option>
            <option value="creative">🎨 Creative</option>
            <option value="analysis">🔬 Analysis</option>
          </select>
        </div>

        {/* Quality Filter */}
        <div className="flex items-center gap-2">
          <label className="text-slate-400 text-sm">Quality:</label>
          <select
            value={filterQuality}
            onChange={(e) => setFilterQuality(e.target.value)}
            className="bg-slate-700 text-white px-3 py-2 rounded-lg text-sm border border-slate-600 focus:border-indigo-500 focus:outline-none"
          >
            <option value="all">All Quality</option>
            <option value="high">✅ High Quality</option>
            <option value="low">⚠️ Low Quality</option>
            <option value="cached">💾 Cached Only</option>
          </select>
        </div>

        {/* Auto-refresh Toggle */}
        <div className="flex items-center gap-2 ml-auto">
          <label className="text-slate-400 text-sm">Auto-refresh:</label>
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              autoRefresh
                ? 'bg-green-600 text-white'
                : 'bg-slate-700 text-slate-400'
            }`}
          >
            {autoRefresh ? '🔄 On' : '⏸️ Off'}
          </button>
        </div>
      </motion.div>

      {/* Trees Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Tree List */}
        <div className="space-y-4">
          <h2 className="text-2xl font-bold text-white mb-4">
            Tree History ({filteredTrees.length})
          </h2>
          
          {isLoading ? (
            <div className="text-center py-12">
              <div className="inline-block animate-spin text-5xl mb-4">🌳</div>
              <p className="text-slate-400">Loading trees...</p>
            </div>
          ) : filteredTrees.length === 0 ? (
            <div className="text-center py-12 bg-slate-800/30 rounded-xl border border-slate-700">
              <p className="text-slate-400 text-lg">No trees found</p>
              <p className="text-slate-500 text-sm mt-2">Run some queries to generate reasoning trees</p>
            </div>
          ) : (
            <AnimatePresence>
              {filteredTrees.map((tree, idx) => (
                <motion.div
                  key={tree.tree_id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ delay: idx * 0.05 }}
                  onClick={() => setSelectedTree(tree)}
                  className={`bg-slate-800/50 backdrop-blur border rounded-xl p-4 cursor-pointer transition-all hover:scale-[1.02] ${
                    selectedTree?.tree_id === tree.tree_id
                      ? 'border-indigo-500 shadow-lg shadow-indigo-500/20'
                      : 'border-slate-700 hover:border-slate-600'
                  }`}
                  style={{
                    borderLeftWidth: '4px',
                    borderLeftColor: getWorkerColor(tree.worker)
                  }}
                >
                  {/* Header */}
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span
                          className="px-2 py-1 rounded text-xs font-bold"
                          style={{
                            backgroundColor: `${getWorkerColor(tree.worker)}20`,
                            color: getWorkerColor(tree.worker)
                          }}
                        >
                          {tree.worker.toUpperCase()}
                        </span>
                        {tree.cache_status !== 'none' && (
                          <span className={`px-2 py-1 rounded text-xs font-bold ${
                            tree.cache_status === 'high'
                              ? 'bg-emerald-900/30 text-emerald-400'
                              : 'bg-amber-900/30 text-amber-400'
                          }`}>
                            💾 {tree.cache_status.toUpperCase()}
                          </span>
                        )}
                        {tree.verification_passed && (
                          <span className="text-emerald-400 text-sm">✅</span>
                        )}
                      </div>
                      <p className="text-white text-sm font-medium line-clamp-2">
                        {tree.query}
                      </p>
                    </div>
                    <div className="text-xs text-slate-500 ml-2">
                      {new Date(tree.timestamp).toLocaleTimeString()}
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    <div>
                      <div className="text-slate-500">Nodes</div>
                      <div className="text-white font-bold">{tree.total_nodes}</div>
                    </div>
                    <div>
                      <div className="text-slate-500">Pruned</div>
                      <div className="text-red-400 font-bold">{tree.pruned_nodes}</div>
                    </div>
                    <div>
                      <div className="text-slate-500">Depth</div>
                      <div className="text-blue-400 font-bold">{tree.max_depth}</div>
                    </div>
                    <div>
                      <div className="text-slate-500">Reward</div>
                      <div className="text-green-400 font-bold">{tree.avg_reward.toFixed(2)}</div>
                    </div>
                  </div>

                  {/* Execution Time */}
                  <div className="mt-2 text-xs text-slate-500">
                    ⏱️ {tree.execution_time.toFixed(2)}s
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          )}
        </div>

        {/* Tree Visualization Canvas */}
        <div className="sticky top-4">
          <h2 className="text-2xl font-bold text-white mb-4">
            Tree Visualization
          </h2>
          
          {selectedTree ? (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-slate-900/80 backdrop-blur border border-slate-700 rounded-xl overflow-hidden"
            >
              {/* Tree Info */}
              <div className="p-4 border-b border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <span
                    className="px-3 py-1 rounded-lg text-sm font-bold"
                    style={{
                      backgroundColor: `${getWorkerColor(selectedTree.worker)}20`,
                      color: getWorkerColor(selectedTree.worker)
                    }}
                  >
                    {selectedTree.worker.toUpperCase()} WORKER
                  </span>
                  {selectedTree.verification_passed && (
                    <span className="text-emerald-400">✅ Verified</span>
                  )}
                </div>
                <p className="text-white text-sm">{selectedTree.query}</p>
              </div>

              {/* Legend */}
              <div className="p-4 bg-slate-800/50 border-b border-slate-700 text-xs">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full bg-green-600"></div>
                    <span className="text-slate-300">High Reward (&gt;0.85)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full bg-yellow-600"></div>
                    <span className="text-slate-300">Medium (0.5-0.85)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full bg-red-900"></div>
                    <span className="text-slate-300">Low Reward (&lt;0.5)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-yellow-400 text-lg">★</span>
                    <span className="text-slate-300">Best Path</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-red-400 text-lg">✗</span>
                    <span className="text-slate-300">Pruned Branch</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full border-2 border-yellow-400"></div>
                    <span className="text-slate-300">Number = Visits</span>
                  </div>
                </div>
              </div>

              {/* Canvas */}
              <div className="relative bg-slate-950/50" style={{ height: '600px' }}>
                <canvas
                  ref={canvasRef}
                  className="w-full h-full"
                  style={{ imageRendering: 'crisp-edges' }}
                />
              </div>

              {/* Tree Stats */}
              <div className="p-4 bg-slate-800/50 border-t border-slate-700">
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="text-slate-400 mb-1">Total Nodes</div>
                    <div className="text-white font-bold text-lg">{selectedTree.total_nodes}</div>
                  </div>
                  <div>
                    <div className="text-slate-400 mb-1">Pruned Nodes</div>
                    <div className="text-red-400 font-bold text-lg">{selectedTree.pruned_nodes}</div>
                  </div>
                  <div>
                    <div className="text-slate-400 mb-1">Max Depth</div>
                    <div className="text-blue-400 font-bold text-lg">{selectedTree.max_depth}</div>
                  </div>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="bg-slate-800/30 border border-slate-700 rounded-xl p-12 text-center">
              <div className="text-6xl mb-4">🌳</div>
              <p className="text-slate-400 text-lg">Select a tree to visualize</p>
              <p className="text-slate-500 text-sm mt-2">
                Click on any tree from the list to see its reasoning structure
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

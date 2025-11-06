'use client'

import { useState, useEffect } from 'react'
import { QueryInterface } from '../components/QueryInterface'
import { MetricsDashboard } from '../components/MetricsDashboard'
import { RouterVisualization } from '../components/RouterVisualization'
import { ConfigPanel } from '../components/ConfigPanel'
import { FineTuningPanel } from '../components/FineTuningPanel'
import { TreesVisualization } from '../components/TreesVisualization'

export default function Home() {
  const [activeTab, setActiveTab] = useState('query')
  const [apiHealth, setApiHealth] = useState<boolean | null>(null)

  // Make setActiveTab available globally for cross-component navigation
  useEffect(() => {
    (window as any).setActiveTab = setActiveTab
  }, [])

  useEffect(() => {
    // Check API health on mount
    fetch('http://localhost:5000/api/health')
      .then(res => res.json())
      .then(() => setApiHealth(true))
      .catch(() => setApiHealth(false))
    
    // Recheck every 10 seconds
    const interval = setInterval(() => {
      fetch('http://localhost:5000/api/health')
        .then(res => res.json())
        .then(() => setApiHealth(true))
        .catch(() => setApiHealth(false))
    }, 10000)
    
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">K</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  Kaelum AI 
                </h1>
              </div>
            </div>
            
            {/* API Status */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${apiHealth ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
              <span className="text-sm text-slate-600 dark:text-slate-400">
                {apiHealth ? 'API Connected' : 'API Offline'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-4 overflow-x-auto">
            {[
              { id: 'query', label: 'Query Interface', icon: '💬' },
              { id: 'metrics', label: 'Metrics & Cache', icon: '📊' },
              { id: 'trees', label: 'Trees & Cache', icon: '🌳' },
              { id: 'router', label: 'Neural Router', icon: '🧠' },
              { id: 'config', label: 'Configuration', icon: '⚙️' },
              { id: 'finetune', label: 'Fine-tuning', icon: '🎯' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300 dark:text-slate-400 dark:hover:text-slate-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'query' && <QueryInterface />}
        {activeTab === 'metrics' && <MetricsDashboard />}
        {activeTab === 'trees' && <TreesVisualization />}
        {activeTab === 'router' && <RouterVisualization />}
        {activeTab === 'config' && <ConfigPanel />}
        {activeTab === 'finetune' && <FineTuningPanel />}
      </main>

      {/* Footer */}
      <footer className="mt-16 border-t border-slate-200 dark:border-slate-700 bg-white/50 dark:bg-slate-900/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-slate-600 dark:text-slate-400">
            Kaelum v2.0 - Advanced AI Reasoning System | Educational Research Project
          </p>
        </div>
      </footer>
    </div>
  )
}

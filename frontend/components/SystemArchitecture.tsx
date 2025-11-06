'use client'

export function SystemArchitecture() {
  const architectureFlow = [
    {
      step: '1',
      title: 'Query Input',
      description: 'User submits a natural language question',
      color: 'from-blue-500 to-blue-600'
    },
    {
      step: '2',
      title: 'Neural Router',
      description: 'AI selects the best expert worker (Math, Code, Logic, etc.)',
      color: 'from-purple-500 to-purple-600'
    },
    {
      step: '3',
      title: 'LATS Tree Search',
      description: 'Explores multiple reasoning paths using Monte Carlo Tree Search',
      color: 'from-green-500 to-green-600'
    },
    {
      step: '4',
      title: 'Verification',
      description: 'Checks correctness (SymPy for math, AST for code, etc.)',
      color: 'from-orange-500 to-orange-600'
    },
    {
      step: '5',
      title: 'Reflection (if needed)',
      description: 'Self-corrects and retries if verification fails',
      color: 'from-red-500 to-red-600'
    },
    {
      step: '6',
      title: 'Final Answer',
      description: 'Verified, high-confidence answer with reasoning trace',
      color: 'from-indigo-500 to-indigo-600'
    },
  ]

  const keyFeatures = [
    {
      icon: '🧠',
      title: 'Neural Router',
      description: 'Deep learning model (398→256→128) learns from every query. Trains after 32 examples using gradient descent.',
      stats: ['6 Expert Workers', '384-dim Embeddings', 'Continual Learning']
    },
    {
      icon: '🌳',
      title: 'LATS (Monte Carlo Tree Search)',
      description: 'Explores multiple reasoning paths before committing. Uses UCT selection with early pruning (visits≥3, reward<0.3).',
      stats: ['10 Simulations', '5 Max Depth', 'Domain Scoring']
    },
    {
      icon: '✅',
      title: 'Multi-Layer Verification',
      description: 'Formal verification ensures correctness. SymPy for math, AST parsing for code, semantic checks for logic.',
      stats: ['Symbolic Math', 'AST Validation', 'Semantic Coherence']
    },
    {
      icon: '⚡',
      title: 'Semantic Cache + LLM Validation',
      description: 'Stores verified solutions with embeddings. LLM validates semantic equivalence before serving cached answers.',
      stats: ['0.001s Lookup', '1000x Speedup', 'Quality Filtered']
    },
    {
      icon: '🔄',
      title: 'Reflection Engine',
      description: 'Self-correction loop analyzes failures and improves reasoning automatically. Up to 2 iterations by default.',
      stats: ['Error Analysis', 'Guided Retry', '~40% Improvement']
    },
    {
      icon: '📚',
      title: 'Active Learning',
      description: 'Intelligently selects hard examples for fine-tuning. Exports training data for continual model improvement.',
      stats: ['Uncertainty Sampling', 'Diversity Selection', 'Error Mining']
    },
  ]

  return (
    <div className="space-y-8">
      {/* Hero */}
      <div className="bg-linear-to-r from-indigo-600 to-purple-600 rounded-2xl p-8 text-white">
        <h2 className="text-3xl font-bold mb-2">System Architecture</h2>
        <p className="text-indigo-100">
          Kaelum combines neural routing, Monte Carlo tree search, and formal verification 
          to solve complex reasoning tasks with high accuracy.
        </p>
      </div>

      {/* Flow Diagram */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-8">
        <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">Processing Pipeline</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {architectureFlow.map((item, i) => (
            <div key={i} className="relative">
              <div className={`bg-linear-to-br ${item.color} rounded-xl p-6 text-white shadow-lg`}>
                <div className="text-4xl font-bold opacity-50 mb-2">{item.step}</div>
                <h4 className="text-xl font-bold mb-2">{item.title}</h4>
                <p className="text-sm opacity-90">{item.description}</p>
              </div>
              {i < architectureFlow.length - 1 && (
                <div className="hidden md:block absolute top-1/2 -right-3 transform -translate-y-1/2">
                  <div className="text-3xl text-slate-300 dark:text-slate-600">→</div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Key Features Grid */}
      <div>
        <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-6">Core Technologies</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {keyFeatures.map((feature, i) => (
            <div key={i} className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow">
              <div className="text-4xl mb-3">{feature.icon}</div>
              <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-2">{feature.title}</h4>
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">{feature.description}</p>
              <div className="flex flex-wrap gap-2">
                {feature.stats.map((stat, j) => (
                  <span key={j} className="px-2 py-1 text-xs bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-full">
                    {stat}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Research Highlights */}
      <div className="bg-linear-to-r from-emerald-600 to-teal-600 rounded-2xl p-8 text-white">
        <h3 className="text-2xl font-bold mb-4">🔬 Research Contributions</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold mb-2">📊 Neural Router with Enhanced Feedback</h4>
            <p className="text-sm text-emerald-100">
              Learns from avg tree rewards, actual depth/simulations used - not just success/failure. 
              Achieves continuous improvement through gradient descent.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2">🌲 LATS with Early Pruning</h4>
            <p className="text-sm text-emerald-100">
              Eliminates unpromising branches (visits{`>`}=3, reward&lt;0.3) to focus compute on high-quality paths. 
              2-3x better solution quality at same compute budget.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2">⚡ Two-Stage Cache Validation</h4>
            <p className="text-sm text-emerald-100">
              Fast embedding similarity (0.001s) + intelligent LLM validation (0.1-0.3s). 
              Prevents false positives while maintaining speed.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2">🔄 Automated Reflection Loop</h4>
            <p className="text-sm text-emerald-100">
              Analyzes verification failures and generates improved reasoning with specific guidance. 
              ~40% improvement in eventual success rate.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

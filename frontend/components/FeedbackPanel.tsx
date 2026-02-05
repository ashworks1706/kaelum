"use client"

import { useState } from 'react'
import { apiUrl } from '@/lib/api-config'
import { motion, AnimatePresence } from 'framer-motion'

interface FeedbackPanelProps {
  query: string
  answer: string
  worker: string
  confidence: number
  verificationPassed: boolean
  executionTime: number
  reasoningSteps: string[]
  onFeedbackSubmit?: () => void
}

interface FeedbackData {
  overall_liked: boolean
  overall_rating: number
  worker_selected: string
  worker_correct: boolean
  suggested_worker?: string
  answer_correct: boolean
  answer_helpful: boolean
  answer_complete: boolean
  answer_rating: number
  steps_helpful: boolean[]
  steps_rating: number[]
  comment?: string
  confidence_shown: number
  verification_passed: boolean
  execution_time: number
  query: string
}

const WORKERS = ['math', 'code', 'logic', 'factual', 'creative', 'analysis']

const StarRating = ({ rating, onRate, size = 'md' }: { rating: number; onRate: (r: number) => void; size?: 'sm' | 'md' }) => {
  const sizeClass = size === 'sm' ? 'w-4 h-4' : 'w-6 h-6'

  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          onClick={() => onRate(star)}
          className={`${sizeClass} transition-colors`}
        >
          <svg
            className={`w-full h-full ${star <= rating ? 'fill-yellow-400 stroke-yellow-500' : 'fill-none stroke-slate-300'}`}
            viewBox="0 0 24 24"
            strokeWidth="2"
          >
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
          </svg>
        </button>
      ))}
    </div>
  )
}

export function FeedbackPanel({
  query,
  worker,
  confidence,
  verificationPassed,
  executionTime,
  reasoningSteps,
  onFeedbackSubmit
}: FeedbackPanelProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)

  const [overallLiked, setOverallLiked] = useState<boolean | null>(null)
  const [overallRating, setOverallRating] = useState(3)

  const [workerCorrect, setWorkerCorrect] = useState<boolean | null>(null)
  const [suggestedWorker, setSuggestedWorker] = useState<string>('')

  const [answerCorrect, setAnswerCorrect] = useState<boolean | null>(null)
  const [answerHelpful, setAnswerHelpful] = useState<boolean | null>(null)
  const [answerComplete, setAnswerComplete] = useState<boolean | null>(null)
  const [answerRating, setAnswerRating] = useState(3)

  const [stepsHelpful, setStepsHelpful] = useState<boolean[]>(reasoningSteps.map(() => true))
  const [stepsRating, setStepsRating] = useState<number[]>(reasoningSteps.map(() => 3))

  const [comment, setComment] = useState('')

  const API_BASE = (process.env.NEXT_PUBLIC_API_BASE as string) || (typeof window !== 'undefined'
    ? (window.location.hostname === 'localhost' ? `${window.location.protocol}//localhost:5000` : window.location.origin)
    : 'http://localhost:5000')

  const handleSubmit = async () => {
    if (overallLiked === null || workerCorrect === null || answerCorrect === null) {
      alert('Please provide feedback on overall satisfaction, worker selection, and answer correctness')
      return
    }

    setIsSubmitting(true)

    const feedbackData: FeedbackData = {
      query,
      overall_liked: overallLiked,
      overall_rating: overallRating,
      worker_selected: worker,
      worker_correct: workerCorrect,
      suggested_worker: workerCorrect ? undefined : suggestedWorker || undefined,
      answer_correct: answerCorrect,
      answer_helpful: answerHelpful ?? true,
      answer_complete: answerComplete ?? true,
      answer_rating: answerRating,
      steps_helpful: stepsHelpful,
      steps_rating: stepsRating,
      comment: comment.trim() || undefined,
      confidence_shown: confidence,
      verification_passed: verificationPassed,
      execution_time: executionTime
    }

    try {
      const response = await fetch(`${API_BASE}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedbackData)
      })

      if (!response.ok) {
        throw new Error('Failed to submit feedback')
      }

      const result = await response.json()
      console.log('Feedback submitted:', result)

      setSubmitted(true)
      setTimeout(() => {
        setIsOpen(false)
        if (onFeedbackSubmit) onFeedbackSubmit()
      }, 2000)
    } catch (error) {
      console.error('Error submitting feedback:', error)
      alert('Failed to submit feedback. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const toggleStepHelpful = (index: number) => {
    const newHelpful = [...stepsHelpful]
    newHelpful[index] = !newHelpful[index]
    setStepsHelpful(newHelpful)
  }

  const setStepRating = (index: number, rating: number) => {
    const newRatings = [...stepsRating]
    newRatings[index] = rating
    setStepsRating(newRatings)
  }

  return (
    <>
      {}
      <motion.button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 bg-gradient-to-r from-purple-600 to-pink-600 text-white px-6 py-3 rounded-full shadow-lg hover:shadow-xl transition-all flex items-center gap-2 z-40"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <span className="text-xl">💬</span>
        <span className="font-semibold">Give Feedback</span>
      </motion.button>

      {}
      <AnimatePresence>
        {isOpen && (
          <>
            {}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
              onClick={() => !isSubmitting && setIsOpen(false)}
            />

            {}
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="fixed inset-4 md:inset-auto md:left-1/2 md:top-1/2 md:-translate-x-1/2 md:-translate-y-1/2 md:w-full md:max-w-4xl md:max-h-[90vh] bg-white dark:bg-slate-800 rounded-2xl shadow-2xl z-50 overflow-hidden flex flex-col"
            >
              {}
              <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold mb-1">Help Us Improve! 🚀</h2>
                    <p className="text-purple-100 text-sm">
                      Your feedback trains the AI to get better at routing, reasoning, and answering
                    </p>
                  </div>
                  <button
                    onClick={() => !isSubmitting && setIsOpen(false)}
                    className="text-white/80 hover:text-white text-2xl leading-none"
                  >
                    ✕
                  </button>
                </div>
              </div>

              {}
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {submitted ? (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="flex flex-col items-center justify-center py-12 text-center"
                  >
                    <div className="text-6xl mb-4">🎉</div>
                    <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
                      Thank You!
                    </h3>
                    <p className="text-slate-600 dark:text-slate-400">
                      Your feedback has been submitted and will help improve the AI
                    </p>
                  </motion.div>
                ) : (
                  <>
                    {}
                    <div className="bg-slate-50 dark:bg-slate-700/50 rounded-xl p-4">
                      <h3 className="font-bold text-slate-900 dark:text-white mb-3">
                        1️⃣ Overall Experience
                      </h3>
                      <div className="space-y-3">
                        <div>
                          <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                            Did you like this response?
                          </p>
                          <div className="flex gap-3">
                            <button
                              onClick={() => setOverallLiked(true)}
                              className={`flex-1 py-3 rounded-lg font-semibold transition-all ${
                                overallLiked === true
                                  ? 'bg-green-500 text-white shadow-lg scale-105'
                                  : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-green-50 dark:hover:bg-slate-600'
                              }`}
                            >
                              👍 Yes
                            </button>
                            <button
                              onClick={() => setOverallLiked(false)}
                              className={`flex-1 py-3 rounded-lg font-semibold transition-all ${
                                overallLiked === false
                                  ? 'bg-red-500 text-white shadow-lg scale-105'
                                  : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-red-50 dark:hover:bg-slate-600'
                              }`}
                            >
                              👎 No
                            </button>
                          </div>
                        </div>
                        <div>
                          <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                            Rate overall quality:
                          </p>
                          <StarRating rating={overallRating} onRate={setOverallRating} />
                        </div>
                      </div>
                    </div>

                    {}
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4">
                      <h3 className="font-bold text-slate-900 dark:text-white mb-3">
                        2️⃣ Worker Selection
                      </h3>
                      <div className="space-y-3">
                        <div className="bg-white dark:bg-slate-700 rounded-lg p-3">
                          <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">
                            Selected worker:
                          </p>
                          <p className="font-bold text-lg text-blue-600 dark:text-blue-400 capitalize">
                            {worker}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                            Was this the right worker for your question?
                          </p>
                          <div className="flex gap-3">
                            <button
                              onClick={() => setWorkerCorrect(true)}
                              className={`flex-1 py-3 rounded-lg font-semibold transition-all ${
                                workerCorrect === true
                                  ? 'bg-green-500 text-white shadow-lg scale-105'
                                  : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-green-50 dark:hover:bg-slate-600'
                              }`}
                            >
                              ✓ Correct
                            </button>
                            <button
                              onClick={() => setWorkerCorrect(false)}
                              className={`flex-1 py-3 rounded-lg font-semibold transition-all ${
                                workerCorrect === false
                                  ? 'bg-red-500 text-white shadow-lg scale-105'
                                  : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-red-50 dark:hover:bg-slate-600'
                              }`}
                            >
                              ✗ Wrong
                            </button>
                          </div>
                        </div>
                        {workerCorrect === false && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                          >
                            <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                              Which worker should have been used?
                            </p>
                            <div className="grid grid-cols-3 gap-2">
                              {WORKERS.filter(w => w !== worker).map(w => (
                                <button
                                  key={w}
                                  onClick={() => setSuggestedWorker(w)}
                                  className={`py-2 px-3 rounded-lg text-sm font-semibold capitalize transition-all ${
                                    suggestedWorker === w
                                      ? 'bg-blue-500 text-white shadow-lg'
                                      : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-blue-50 dark:hover:bg-slate-600'
                                  }`}
                                >
                                  {w}
                                </button>
                              ))}
                            </div>
                          </motion.div>
                        )}
                      </div>
                    </div>

                    {}
                    <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-4">
                      <h3 className="font-bold text-slate-900 dark:text-white mb-3">
                        3️⃣ Answer Quality
                      </h3>
                      <div className="space-y-3">
                        <div>
                          <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                            Was the answer correct?
                          </p>
                          <div className="flex gap-3">
                            <button
                              onClick={() => setAnswerCorrect(true)}
                              className={`flex-1 py-3 rounded-lg font-semibold transition-all ${
                                answerCorrect === true
                                  ? 'bg-green-500 text-white shadow-lg scale-105'
                                  : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-green-50 dark:hover:bg-slate-600'
                              }`}
                            >
                              ✓ Correct
                            </button>
                            <button
                              onClick={() => setAnswerCorrect(false)}
                              className={`flex-1 py-3 rounded-lg font-semibold transition-all ${
                                answerCorrect === false
                                  ? 'bg-red-500 text-white shadow-lg scale-105'
                                  : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-red-50 dark:hover:bg-slate-600'
                              }`}
                            >
                              ✗ Incorrect
                            </button>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                              Was it helpful?
                            </p>
                            <div className="flex gap-2">
                              <button
                                onClick={() => setAnswerHelpful(true)}
                                className={`flex-1 py-2 rounded-lg font-semibold text-sm transition-all ${
                                  answerHelpful === true
                                    ? 'bg-green-500 text-white'
                                    : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300'
                                }`}
                              >
                                Yes
                              </button>
                              <button
                                onClick={() => setAnswerHelpful(false)}
                                className={`flex-1 py-2 rounded-lg font-semibold text-sm transition-all ${
                                  answerHelpful === false
                                    ? 'bg-red-500 text-white'
                                    : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300'
                                }`}
                              >
                                No
                              </button>
                            </div>
                          </div>

                          <div>
                            <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                              Was it complete?
                            </p>
                            <div className="flex gap-2">
                              <button
                                onClick={() => setAnswerComplete(true)}
                                className={`flex-1 py-2 rounded-lg font-semibold text-sm transition-all ${
                                  answerComplete === true
                                    ? 'bg-green-500 text-white'
                                    : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300'
                                }`}
                              >
                                Yes
                              </button>
                              <button
                                onClick={() => setAnswerComplete(false)}
                                className={`flex-1 py-2 rounded-lg font-semibold text-sm transition-all ${
                                  answerComplete === false
                                    ? 'bg-red-500 text-white'
                                    : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300'
                                }`}
                              >
                                No
                              </button>
                            </div>
                          </div>
                        </div>

                        <div>
                          <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                            Rate answer quality:
                          </p>
                          <StarRating rating={answerRating} onRate={setAnswerRating} />
                        </div>
                      </div>
                    </div>

                    {}
                    {reasoningSteps.length > 0 && (
                      <div className="bg-amber-50 dark:bg-amber-900/20 rounded-xl p-4">
                        <h3 className="font-bold text-slate-900 dark:text-white mb-3">
                          4️⃣ Reasoning Steps ({reasoningSteps.length})
                        </h3>
                        <div className="space-y-2 max-h-60 overflow-y-auto">
                          {reasoningSteps.map((step, i) => (
                            <div key={i} className="bg-white dark:bg-slate-700 rounded-lg p-3">
                              <div className="flex items-start gap-3 mb-2">
                                <div className="flex-shrink-0 w-6 h-6 bg-amber-500 text-white rounded-full flex items-center justify-center text-sm font-bold">
                                  {i + 1}
                                </div>
                                <p className="text-sm text-slate-700 dark:text-slate-300 flex-1">
                                  {step.substring(0, 100)}...
                                </p>
                              </div>
                              <div className="flex items-center justify-between pl-9">
                                <button
                                  onClick={() => toggleStepHelpful(i)}
                                  className={`text-sm font-semibold transition-colors ${
                                    stepsHelpful[i]
                                      ? 'text-green-600 dark:text-green-400'
                                      : 'text-red-600 dark:text-red-400'
                                  }`}
                                >
                                  {stepsHelpful[i] ? '✓ Helpful' : '✗ Not helpful'}
                                </button>
                                <StarRating
                                  rating={stepsRating[i]}
                                  onRate={(r) => setStepRating(i, r)}
                                  size="sm"
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {}
                    <div className="bg-slate-50 dark:bg-slate-700/50 rounded-xl p-4">
                      <h3 className="font-bold text-slate-900 dark:text-white mb-3">
                        5️⃣ Additional Comments (Optional)
                      </h3>
                      <textarea
                        value={comment}
                        onChange={(e) => setComment(e.target.value)}
                        placeholder="Any other feedback? What could be improved?"
                        className="w-full px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent dark:bg-slate-700 dark:text-white resize-none"
                        rows={3}
                      />
                    </div>
                  </>
                )}
              </div>

              {}
              {!submitted && (
                <div className="border-t border-slate-200 dark:border-slate-700 p-6 bg-slate-50 dark:bg-slate-900/50">
                  <div className="flex gap-3">
                    <button
                      onClick={() => setIsOpen(false)}
                      disabled={isSubmitting}
                      className="flex-1 py-3 px-4 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg font-semibold hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-50 transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleSubmit}
                      disabled={isSubmitting || overallLiked === null || workerCorrect === null || answerCorrect === null}
                      className="flex-1 py-3 px-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      {isSubmitting ? (
                        <span className="flex items-center justify-center gap-2">
                          <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                          Submitting...
                        </span>
                      ) : (
                        'Submit Feedback'
                      )}
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  )
}

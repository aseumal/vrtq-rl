/**
 * hooks/useApi.js
 * Custom React hooks for VRTQ-RL API calls.
 */
import { useState, useCallback } from 'react'

const API_BASE = '/api'

export function usePrioritize() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const run = useCallback(async ({ modules, churn, budget }) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/prioritize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modules, churn, budget, use_llm: false }),
      })
      if (!res.ok) throw new Error(`API error: ${res.status}`)
      const json = await res.json()
      setData(json)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return { data, loading, error, run }
}

export function useCompare() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const run = useCallback(async (budget = 50) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/compare?budget=${budget}`)
      if (!res.ok) throw new Error(`API error: ${res.status}`)
      setData(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return { data, loading, error, run }
}

export function useLearningCurve() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)

  const run = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/learning-curve`)
      if (res.ok) setData(await res.json())
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }, [])

  return { data, loading, run }
}

export function useStatus() {
  const [data, setData] = useState(null)

  const fetch_ = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/status`)
      if (res.ok) setData(await res.json())
    } catch (e) {
      console.error(e)
    }
  }, [])

  return { data, refresh: fetch_ }
}

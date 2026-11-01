import { useState, useEffect, useRef } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import { checkDemoMode } from './api'

export default function App() {
  const [demoMode, setDemoMode] = useState(false)
  const [apiError, setApiError] = useState('')
  const successfulInference = useRef(false)
  useEffect(() => {
    let active = true
    checkDemoMode().then((mode) => { if (active) setDemoMode(mode) }).catch((error) => {
      if (active && !successfulInference.current) setApiError(error.message || 'API unavailable')
    })
    return () => { active = false }
  }, [])
  function inferenceAvailable() {
    successfulInference.current = true
    setApiError('')
  }
  return (
    <BrowserRouter>
      {demoMode && <div role="status" className="bg-amber-600 text-slate-950 text-center py-2">Sample mode is enabled. Model inference is unavailable in this mode.</div>}
      {apiError && <div role="alert" className="bg-red-900 text-white text-center py-2">Model API unavailable: {apiError}. No prediction has been generated.</div>}
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard onInferenceSuccess={inferenceAvailable} />} />
          <Route path="*" element={<p className="p-6">This release supports local image uploads. Return to the home page.</p>} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

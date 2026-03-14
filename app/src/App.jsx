import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Compare from './pages/Compare'
import Batch from './pages/Batch'
import { checkDemoMode } from './api'

export default function App() {
  const [demoMode, setDemoMode] = useState(false)
  const [checkingApi, setCheckingApi] = useState(true)

  useEffect(() => {
    checkDemoMode().then((isDemo) => {
      setDemoMode(isDemo)
      setCheckingApi(false)
    })
  }, [])

  return (
    <BrowserRouter>
      {demoMode && !checkingApi && (
        <div className="bg-amber-600 text-slate-950 text-center py-2 text-sm font-medium">
          <span className="material-symbols-outlined text-sm align-middle mr-1">info</span>
          Demo Mode - Using sample data. Configure GOOGLE_MAPS_API_KEY for live satellite imagery.
        </div>
      )}
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="compare" element={<Compare />} />
          <Route path="batch" element={<Batch />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

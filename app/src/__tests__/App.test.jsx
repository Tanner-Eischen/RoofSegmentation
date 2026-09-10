import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'

// Mock the api module before importing App
vi.mock('../api', () => ({
  checkDemoMode: () => Promise.resolve(true),
  isDemoMode: () => true,
}))

describe('App', () => {
  it('renders without crashing', async () => {
    // Import App after mocks are set up
    const { default: App } = await import('../App')

    // App already contains BrowserRouter, so render directly
    const { container } = render(<App />)
    expect(container).toBeTruthy()
    expect(await screen.findByText(/Demo Mode - Using sample data/i)).toBeInTheDocument()
  })
})

import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock the api module before importing App
vi.mock('../api', () => ({
  checkDemoMode: () => Promise.resolve(false),
  isDemoMode: () => false,
}))

describe('App', () => {
  it('renders without crashing', async () => {
    // Import App after mocks are set up
    const { default: App } = await import('../App')

    // App already contains BrowserRouter, so render directly
    const { container } = render(<App />)
    expect(container).toBeTruthy()
  })
})

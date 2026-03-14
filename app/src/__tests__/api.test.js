import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock fetch globally
const mockFetch = vi.fn()
global.fetch = mockFetch

// Mock canvas for demo mode
HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
  fillRect: vi.fn(),
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  closePath: vi.fn(),
  fill: vi.fn(),
}))
HTMLCanvasElement.prototype.toDataURL = vi.fn(() => 'data:image/png;base64,mockBase64Data')

// Reset module cache between tests
beforeEach(() => {
  vi.resetModules()
  mockFetch.mockReset()
})

describe('API client', () => {
  it('geocode returns coordinates in demo mode', async () => {
    // Force demo mode by making fetch fail
    mockFetch.mockRejectedValueOnce(new Error('Network error'))

    const { geocode, checkDemoMode } = await import('../api')

    // First call to checkDemoMode will fail and set demo mode
    await checkDemoMode()

    const result = await geocode('123 Test St')

    // In demo mode, should return mock coordinates
    expect(result.lat).toBeDefined()
    expect(result.lng).toBeDefined()
    expect(result.formatted_address).toBeDefined()
  })

  it('runInferenceFromBlob returns mask and confidence in demo mode', async () => {
    // Force demo mode
    mockFetch.mockRejectedValueOnce(new Error('Network error'))

    const { runInferenceFromBlob, checkDemoMode } = await import('../api')

    // Initialize demo mode
    await checkDemoMode()

    const blob = new Blob(['test'], { type: 'image/png' })
    const result = await runInferenceFromBlob(blob)

    expect(result.mask_base64).toBeDefined()
    expect(result.confidence).toBeDefined()
    expect(result.roof_area_px).toBeDefined()
    expect(typeof result.confidence).toBe('number')
    expect(result.confidence).toBeGreaterThanOrEqual(0)
    expect(result.confidence).toBeLessThanOrEqual(100)
  })

  it('getDemoSamples returns sample data', async () => {
    const { getDemoSamples } = await import('../api')

    const samples = getDemoSamples()

    expect(Array.isArray(samples)).toBe(true)
    expect(samples.length).toBeGreaterThan(0)
    expect(samples[0]).toHaveProperty('address')
    expect(samples[0]).toHaveProperty('confidence')
    expect(samples[0]).toHaveProperty('roof_area_px')
  })
})

import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, it, expect, vi } from 'vitest'

// Mock the api module
vi.mock('../api', () => ({
  geocode: vi.fn().mockResolvedValue({ lat: 39.78, lng: -89.65, formatted_address: 'Test Address' }),
  getSatelliteUrl: vi.fn().mockResolvedValue('http://example.com/satellite.png'),
  fetchSatelliteImageBlob: vi.fn().mockResolvedValue(new Blob()),
  runInferenceFromBlob: vi.fn().mockResolvedValue({
    mask_base64: 'test',
    polygons: [],
    roof_area_px: 45000,
    confidence: 87.5
  }),
}))

describe('Dashboard', () => {
  it('shows address input and analyze button', async () => {
    const { default: Dashboard } = await import('../pages/Dashboard')
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    )

    expect(screen.getByPlaceholderText(/enter address for roof analysis/i)).toBeTruthy()
    expect(screen.getByRole('button', { name: /analyze address/i })).toBeTruthy()
  })

  it('displays error message on failed analysis', async () => {
    const api = await import('../api')
    api.geocode.mockRejectedValueOnce(new Error('Geocode failed'))

    const { default: Dashboard } = await import('../pages/Dashboard')
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    )

    const input = screen.getByPlaceholderText(/enter address for roof analysis/i)
    const button = screen.getByRole('button', { name: /analyze address/i })

    fireEvent.change(input, { target: { value: '123 Test St' } })
    fireEvent.click(button)

    expect(await screen.findByText('Geocode failed')).toBeInTheDocument()
  })
})

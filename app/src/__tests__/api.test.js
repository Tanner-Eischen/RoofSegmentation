import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
const mockFetch = vi.fn();
beforeEach(() => { vi.resetModules(); vi.stubEnv('VITE_DEMO_MODE', 'false'); vi.stubGlobal('fetch', mockFetch); mockFetch.mockReset(); });
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); });
describe('API failure boundaries', () => {
  it('propagates outages without enabling demo or synthesizing predictions', async () => {
    mockFetch.mockRejectedValue(new Error('Network unavailable'));
    const api = await import('../api');
    await expect(api.checkDemoMode()).rejects.toThrow('Network unavailable');
    expect(api.isDemoMode()).toBe(false);
    await expect(api.runInference(new File(['x'], 'x.png'))).rejects.toThrow('Network unavailable');
  });
  it('requires explicit demo and rejects inference without fetching', async () => {
    vi.stubEnv('VITE_DEMO_MODE', 'true');
    const api = await import('../api');
    expect(await api.checkDemoMode()).toBe(true);
    await expect(api.runInference(new File(['x'], 'x.png'))).rejects.toThrow('Sample mode does not run inference');
    expect(mockFetch).not.toHaveBeenCalled();
  });
  it('propagates readiness and inference errors', async () => {
    mockFetch.mockResolvedValue({ ok:false, status:503, json:async () => ({ detail:'Checkpoint unavailable' }) });
    const api = await import('../api');
    await expect(api.checkDemoMode()).rejects.toThrow('Checkpoint unavailable');
    await expect(api.runInference(new File(['x'], 'x.png'))).rejects.toThrow('Checkpoint unavailable');
  });
  it('sends real upload bytes through multipart', async () => {
    mockFetch.mockResolvedValue({ ok:true, json:async () => ({ mode:'real', roof_area_px:10 }) });
    const api = await import('../api');
    const file = new File(['actual bytes'], 'roof.png', { type:'image/png' });
    expect((await api.runInference(file)).mode).toBe('real');
    expect(mockFetch.mock.calls[0][0]).toBe('/api/inference');
    expect(mockFetch.mock.calls[0][1].body.get('file').name).toBe('roof.png');
  });
});

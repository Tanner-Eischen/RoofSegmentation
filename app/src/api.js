const API_BASE = import.meta.env.VITE_API_BASE || '';
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true';

// Demo is an explicit build setting. API outages never enable synthetic inference.
export async function checkDemoMode() {
  if (DEMO_MODE) return true;
  await apiJson('/api/ready');
  return false;
}

export function isDemoMode() { return DEMO_MODE; }

async function apiFetch(endpoint, options = {}) {
  if (DEMO_MODE) throw new Error('Sample mode does not run inference. Configure a real API checkpoint.');
  const r = await fetch(`${API_BASE}${endpoint}`, { signal: AbortSignal.timeout(65000), ...options });
  if (!r.ok) {
    let message = `API request failed (${r.status})`;
    try { message = (await r.json()).detail || message; } catch { /* retain status */ }
    throw new Error(message);
  }
  return r;
}
async function apiJson(endpoint, options = {}) { return (await apiFetch(endpoint, options)).json(); }
export function getDemoSamples() { return []; }
export function geocode(address) { return apiJson(`/api/geocode?address=${encodeURIComponent(address)}`); }
export async function getSatelliteUrl(lat, lng, zoom = 20, width = 640, height = 640) {
  return (await apiJson(`/api/satellite?lat=${lat}&lng=${lng}&zoom=${zoom}&width=${width}&height=${height}`)).url;
}
export async function fetchSatelliteImageBlob(lat, lng, zoom = 20, width = 640, height = 640) {
  return (await apiFetch(`/api/satellite/image?lat=${lat}&lng=${lng}&zoom=${zoom}&width=${width}&height=${height}`)).blob();
}
export function runInference(imageFile) {
  const form = new FormData(); form.append('file', imageFile);
  return apiJson('/api/inference', { method: 'POST', body: form });
}
export function runInferenceFromBlob(blob) { return runInference(new File([blob], 'image.png', { type: blob.type || 'image/png' })); }
export function createBatch(addresses) {
  return apiJson('/api/batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ addresses }) });
}
export function uploadBatchCsv(file) {
  const form = new FormData(); form.append('file', file);
  return apiJson('/api/batch/upload', { method: 'POST', body: form });
}
export function getBatchStatus(jobId) { return apiJson(`/api/batch/${encodeURIComponent(jobId)}`); }

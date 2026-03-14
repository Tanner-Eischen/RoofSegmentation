const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

// Demo mode detection - check if API is available
let _demoMode = null;

export async function checkDemoMode() {
  if (_demoMode !== null) return _demoMode;
  try {
    const r = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) });
    _demoMode = !r.ok;
  } catch {
    _demoMode = true;
  }
  return _demoMode;
}

export function isDemoMode() {
  return _demoMode === true;
}

// Fetch wrapper with error handling
async function apiFetch(endpoint, options = {}) {
  const r = await fetch(`${API_BASE}${endpoint}`, options);
  if (!r.ok) throw new Error(await r.text());
  return r;
}

async function apiJson(endpoint, options = {}) {
  const r = await apiFetch(endpoint, options);
  return r.json();
}

// Demo data for when API is unavailable
const DEMO_DATA = [
  {
    id: "sample1",
    label: "Sample Residential Roof 1",
    address: "123 Oak Street, Springfield, IL",
    lat: 39.7817,
    lng: -89.6501,
    roof_area_px: 45600,
    confidence: 87.3,
    polygon_count: 2,
    roof_coverage_percent: 11.1
  },
  {
    id: "sample2",
    label: "Sample Residential Roof 2",
    address: "456 Maple Avenue, Austin, TX",
    lat: 30.2672,
    lng: -97.7431,
    roof_area_px: 62300,
    confidence: 91.5,
    polygon_count: 3,
    roof_coverage_percent: 15.2
  },
  {
    id: "sample3",
    label: "Sample Commercial Roof",
    address: "789 Industrial Blvd, Denver, CO",
    lat: 39.7392,
    lng: -104.9903,
    roof_area_px: 128400,
    confidence: 94.2,
    polygon_count: 1,
    roof_coverage_percent: 31.3
  }
];

export function getDemoSamples() {
  return DEMO_DATA;
}

export async function geocode(address) {
  if (_demoMode) {
    // Find matching demo sample or return mock data
    const sample = DEMO_DATA.find(s => s.address.toLowerCase().includes(address.toLowerCase()));
    if (sample) {
      return { lat: sample.lat, lng: sample.lng, formatted_address: sample.address };
    }
    // Return mock geocode for any address
    return { lat: 39.7817, lng: -89.6501, formatted_address: address };
  }
  return apiJson(`/api/geocode?address=${encodeURIComponent(address)}`);
}

export async function getSatelliteUrl(lat, lng, zoom = 20, width = 640, height = 640) {
  if (_demoMode) {
    // Return placeholder image URL (SVG works as img src)
    return `/samples/placeholder-satellite.svg`;
  }
  const data = await apiJson(`/api/satellite?lat=${lat}&lng=${lng}&zoom=${zoom}&width=${width}&height=${height}`);
  return data.url;
}

export async function fetchSatelliteImageBlob(lat, lng, zoom = 20, width = 640, height = 640) {
  if (_demoMode) {
    // Return placeholder image blob (fetch SVG and convert)
    const response = await fetch('/samples/placeholder-satellite.svg');
    return response.blob();
  }
  const r = await apiFetch(`/api/satellite/image?lat=${lat}&lng=${lng}&zoom=${zoom}&width=${width}&height=${height}`);
  return r.blob();
}

export async function runInference(imageFile) {
  if (_demoMode) {
    // Return mock inference result
    const sample = DEMO_DATA[Math.floor(Math.random() * DEMO_DATA.length)];
    // Create a simple placeholder mask (gray square)
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 640;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#333333';
    ctx.fillRect(0, 0, 640, 640);
    // Draw some random roof-shaped polygons
    ctx.fillStyle = '#666666';
    ctx.beginPath();
    ctx.moveTo(200, 150);
    ctx.lineTo(400, 150);
    ctx.lineTo(420, 300);
    ctx.lineTo(180, 300);
    ctx.closePath();
    ctx.fill();

    const maskBase64 = canvas.toDataURL('image/png').split(',')[1];
    return {
      mask_base64: maskBase64,
      polygons: [[[200, 150], [400, 150], [420, 300], [180, 300]]],
      roof_area_px: sample.roof_area_px,
      confidence: sample.confidence
    };
  }
  const form = new FormData();
  form.append('file', imageFile);
  return apiJson('/api/inference', { method: 'POST', body: form });
}

export async function runInferenceFromBlob(blob) {
  const file = new File([blob], 'image.png', { type: blob.type || 'image/png' });
  return runInference(file);
}

export async function createBatch(addresses) {
  if (_demoMode) {
    // Mock batch creation
    return { job_id: 'demo-job-' + Date.now(), status: 'queued' };
  }
  return apiJson('/api/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ addresses }),
  });
}

export async function uploadBatchCsv(file) {
  if (_demoMode) {
    // Mock batch upload
    return { job_id: 'demo-job-' + Date.now(), status: 'queued' };
  }
  const form = new FormData();
  form.append('file', file);
  return apiJson('/api/batch/upload', { method: 'POST', body: form });
}

export async function getBatchStatus(jobId) {
  if (_demoMode) {
    // Mock batch status
    return {
      job_id: jobId,
      status: 'complete',
      addresses: ['Demo Address 1', 'Demo Address 2'],
      results: DEMO_DATA.map(s => ({
        address: s.address,
        formatted_address: s.address,
        status: 'complete',
        roof_area_px: s.roof_area_px,
        confidence: s.confidence,
        polygons: []
      })),
      total: 3,
      done: 3
    };
  }
  return apiJson(`/api/batch/${jobId}`);
}

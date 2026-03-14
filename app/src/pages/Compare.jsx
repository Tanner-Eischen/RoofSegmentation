import { useState } from 'react';
import { geocode, getSatelliteUrl, fetchSatelliteImageBlob, runInferenceFromBlob } from '../api';

function LocationCard({ label, address, onAddressChange, onAnalyze, loading, satelliteUrl, maskBase64, confidence }) {
  const [localAddr, setLocalAddr] = useState(address || '');

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 overflow-hidden flex flex-col flex-1 min-w-0">
      <div className="px-4 py-3 border-b border-slate-800 flex justify-between items-center shrink-0">
        <span className="text-sm font-semibold flex items-center gap-2">
          <span className="material-symbols-outlined text-[var(--color-primary)]">location_on</span>
          {label}
        </span>
      </div>
      <div className="p-3 border-b border-slate-800">
        <input
          type="text"
          value={localAddr}
          onChange={(e) => setLocalAddr(e.target.value)}
          placeholder={`Enter address ${label}...`}
          className="w-full rounded-lg bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:ring-2 focus:ring-[var(--color-primary)]"
        />
        <button
          type="button"
          onClick={() => onAnalyze(localAddr)}
          disabled={loading || !localAddr.trim()}
          className="mt-2 w-full rounded-lg bg-[var(--color-primary)] text-slate-950 py-2 text-sm font-semibold disabled:opacity-50"
        >
          {loading ? 'Analyzing…' : 'Analyze'}
        </button>
      </div>
      <div className="flex-1 relative min-h-[220px] bg-slate-950">
        {satelliteUrl ? (
          <>
            <img src={satelliteUrl} alt={`${label} satellite`} className="absolute inset-0 w-full h-full object-cover" />
            {maskBase64 && (
              <img
                src={`data:image/png;base64,${maskBase64}`}
                alt="Mask"
                className="absolute inset-0 w-full h-full object-cover mix-blend-multiply opacity-50"
                style={{ imageRendering: 'pixelated' }}
              />
            )}
            {confidence != null && (
              <div className="absolute bottom-2 right-2 rounded bg-slate-900/90 px-2 py-1 text-[10px] font-semibold text-[var(--color-primary)]">
                {confidence}% confidence
              </div>
            )}
          </>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">Enter address and Analyze</div>
        )}
      </div>
    </div>
  );
}

export default function Compare() {
  const [loadingA, setLoadingA] = useState(false);
  const [loadingB, setLoadingB] = useState(false);
  const [error, setError] = useState(null);
  const [dataA, setDataA] = useState({ satelliteUrl: null, maskBase64: null, confidence: null });
  const [dataB, setDataB] = useState({ satelliteUrl: null, maskBase64: null, confidence: null });

  async function analyzeLocation(label, address) {
    if (!address.trim()) return;
    setError(null);
    const setLoading = label === 'Location A' ? setLoadingA : setLoadingB;
    const setData = label === 'Location A' ? setDataA : setDataB;
    setLoading(true);
    setData({ satelliteUrl: null, maskBase64: null, confidence: null });
    try {
      const geo = await geocode(address.trim());
      const url = await getSatelliteUrl(geo.lat, geo.lng);
      setData((d) => ({ ...d, satelliteUrl: url }));
      const blob = await fetchSatelliteImageBlob(geo.lat, geo.lng);
      const result = await runInferenceFromBlob(blob);
      setData((d) => ({ ...d, maskBase64: result.mask_base64, confidence: result.confidence }));
    } catch (err) {
      setError(err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-4 lg:p-6 flex flex-col h-full">
      <div className="mb-4">
        <h1 className="text-xl font-bold">Property Comparison</h1>
        <p className="text-slate-400 text-sm mt-1">Compare satellite and roof detection for two addresses.</p>
      </div>
      {error && (
        <div className="rounded-lg bg-red-900/30 border border-red-700 text-red-200 px-4 py-2 text-sm mb-4">
          {error}
        </div>
      )}
      <div className="flex gap-6 flex-1 min-h-0">
        <LocationCard
          label="Location A"
          onAnalyze={(addr) => analyzeLocation('Location A', addr)}
          loading={loadingA}
          satelliteUrl={dataA.satelliteUrl}
          maskBase64={dataA.maskBase64}
          confidence={dataA.confidence}
        />
        <LocationCard
          label="Location B"
          onAnalyze={(addr) => analyzeLocation('Location B', addr)}
          loading={loadingB}
          satelliteUrl={dataB.satelliteUrl}
          maskBase64={dataB.maskBase64}
          confidence={dataB.confidence}
        />
      </div>
    </div>
  );
}

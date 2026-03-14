import { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { geocode, getSatelliteUrl, fetchSatelliteImageBlob, runInferenceFromBlob } from '../api';

export default function Dashboard() {
  const location = useLocation();
  const batchResults = location.state?.batchResults || null;
  const [address, setAddress] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [satelliteUrl, setSatelliteUrl] = useState(null);
  const [propertyInfo, setPropertyInfo] = useState(null);
  const [maskBase64, setMaskBase64] = useState(null);
  const [polygons, setPolygons] = useState([]);
  const [roofAreaPx, setRoofAreaPx] = useState(null);

  async function handleAnalyze(e) {
    e.preventDefault();
    if (!address.trim()) return;
    setError(null);
    setLoading(true);
    setSatelliteUrl(null);
    setPropertyInfo(null);
    setMaskBase64(null);
    setPolygons([]);
    setRoofAreaPx(null);
    try {
      const geo = await geocode(address.trim());
      setPropertyInfo({ formatted_address: geo.formatted_address, lat: geo.lat, lng: geo.lng });
      const url = await getSatelliteUrl(geo.lat, geo.lng);
      setSatelliteUrl(url);
      const blob = await fetchSatelliteImageBlob(geo.lat, geo.lng);
      const result = await runInferenceFromBlob(blob);
      setMaskBase64(result.mask_base64);
      setPolygons(result.polygons || []);
      setRoofAreaPx(result.roof_area_px);
    } catch (err) {
      setError(err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-1 overflow-hidden p-4 lg:p-6">
      <div className="flex flex-1 gap-6 min-h-0">
        <div className="flex-1 flex flex-col gap-4 overflow-auto">
          {/* Batch Results View */}
          {batchResults && batchResults.length > 0 ? (
            <div className="flex flex-col gap-4">
              <h2 className="text-lg font-semibold">Batch Analysis Results</h2>
              <div className="rounded-xl border border-slate-800 overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-slate-900 border-b border-slate-800">
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">Address</th>
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">Roof Area (px²)</th>
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">Polygons</th>
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">Confidence</th>
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchResults.map((result, idx) => (
                      <tr key={idx} className="border-b border-slate-800 hover:bg-slate-900/50">
                        <td className="py-3 px-4 text-slate-100 max-w-xs truncate">{result.formatted_address || result.address}</td>
                        <td className="py-3 px-4">{result.roof_area_px ? Math.round(result.roof_area_px).toLocaleString() : '—'}</td>
                        <td className="py-3 px-4">{result.polygons?.length || 0}</td>
                        <td className="py-3 px-4">{result.confidence ? `${result.confidence}%` : '—'}</td>
                        <td className="py-3 px-4">
                          <span className={`inline-flex px-2 py-0.5 rounded text-xs font-medium ${
                            result.status === 'complete' ? 'bg-green-900/40 text-green-300' : 'bg-red-900/40 text-red-300'
                          }`}>
                            {result.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <>
          {/* Single Address Analysis */}
          <form onSubmit={handleAnalyze} className="flex gap-3 max-w-2xl">
            <div className="relative flex-1">
              <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-xl">search</span>
              <input
                type="text"
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                placeholder="Enter address for roof analysis..."
                className="w-full rounded-lg border-0 bg-slate-800 py-2.5 pl-10 pr-4 text-slate-100 ring-1 ring-slate-700 placeholder:text-slate-400 focus:ring-2 focus:ring-[var(--color-primary)] text-sm"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="shrink-0 rounded-lg bg-[var(--color-primary)] px-5 py-2.5 text-sm font-semibold text-slate-950 hover:opacity-90 disabled:opacity-50"
            >
              {loading ? 'Analyzing…' : 'Analyze Address'}
            </button>
          </form>
          {error && (
            <div className="rounded-lg bg-red-900/30 border border-red-700 text-red-200 px-4 py-2 text-sm">
              {error}
            </div>
          )}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-0">
            <div className="rounded-xl border border-slate-800 bg-slate-900 overflow-hidden flex flex-col h-[280px] lg:h-[350px]">
              <div className="px-4 py-2 border-b border-slate-800 flex justify-between items-center shrink-0">
                <span className="text-sm font-medium">Original Satellite View</span>
              </div>
              <div className="flex-1 relative bg-slate-950 min-h-0">
                {satelliteUrl ? (
                  <img src={satelliteUrl} alt="Satellite" className="absolute inset-0 w-full h-full object-cover" />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">Enter address and click Analyze</div>
                )}
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900 overflow-hidden flex flex-col h-[280px] lg:h-[350px]">
              <div className="px-4 py-2 border-b border-slate-800 flex justify-between items-center shrink-0">
                <span className="text-sm font-medium">Roof Polygons</span>
                {maskBase64 && <span className="text-[10px] uppercase tracking-wider text-[var(--color-primary)] font-bold">Processed</span>}
              </div>
              <div className="flex-1 relative bg-slate-950 min-h-0">
                {satelliteUrl && (
                  <>
                    <img src={satelliteUrl} alt="Satellite" className="absolute inset-0 w-full h-full object-cover" />
                    {maskBase64 && (
                      <img
                        src={`data:image/png;base64,${maskBase64}`}
                        alt="Mask"
                        className="absolute inset-0 w-full h-full object-cover mix-blend-multiply opacity-60"
                        style={{ imageRendering: 'pixelated' }}
                      />
                    )}
                  </>
                )}
                {!satelliteUrl && (
                  <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">—</div>
                )}
              </div>
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 overflow-hidden flex flex-col min-h-[200px]">
            <div className="px-4 py-2 border-b border-slate-800 shrink-0">
              <span className="text-sm font-medium">Roof Segmentation</span>
            </div>
            <div className="flex-1 relative bg-slate-950 min-h-[160px] p-2">
              <div className="absolute bottom-2 left-2 rounded-lg border border-slate-700 bg-slate-900/90 px-3 py-2 text-xs">
                <div className="font-medium text-slate-400 mb-1">Legend</div>
                <div className="flex items-center gap-2"><span className="w-3 h-3 rounded bg-slate-500" /> Roof (detected)</div>
              </div>
              {satelliteUrl && maskBase64 && (
                <div className="absolute inset-2 flex gap-2">
                  <img src={satelliteUrl} alt="Satellite" className="flex-1 object-contain max-h-full" />
                  <img src={`data:image/png;base64,${maskBase64}`} alt="Mask" className="flex-1 object-contain max-h-full opacity-80" style={{ imageRendering: 'pixelated' }} />
                </div>
              )}
            </div>
          </div>
            </>
          )}
        </div>
        <aside className="w-80 shrink-0 flex flex-col gap-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h3 className="text-sm font-medium text-slate-400 flex items-center gap-2 mb-2">
              <span className="material-symbols-outlined text-lg">location_on</span>
              Property Info
            </h3>
            {propertyInfo ? (
              <>
                <div className="text-slate-100">{propertyInfo.formatted_address}</div>
              </>
            ) : (
              <div className="text-slate-500 text-sm">—</div>
            )}
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h3 className="text-sm font-medium text-slate-400 mb-2">Detection Summary</h3>
            {roofAreaPx != null ? (
              <>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden flex">
                  <div className="bg-[var(--color-primary)]" style={{ width: `${Math.min(100, (roofAreaPx / (640 * 640)) * 100)}%` }} title="Roof coverage" />
                </div>
                <div className="text-xs text-slate-500 mt-1">{((roofAreaPx / (640 * 640)) * 100).toFixed(1)}% roof coverage</div>
              </>
            ) : (
              <div className="text-slate-500 text-sm">—</div>
            )}
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h3 className="text-sm font-medium text-slate-400 mb-2">Results Summary</h3>
            {roofAreaPx != null ? (
              <>
                <div className="text-2xl font-bold text-[var(--color-primary)]">{Math.round(roofAreaPx).toLocaleString()}</div>
                <div className="text-xs text-slate-500">roof area (px²)</div>
                <div className="mt-2 text-sm text-slate-400">{polygons.length} polygon(s) detected</div>
              </>
            ) : (
              <div className="text-slate-500 text-sm">—</div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

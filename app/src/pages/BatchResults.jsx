import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getBatchStatus } from '../api';

export default function BatchResults() {
  const { jobId } = useParams();
  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadJob() {
      try {
        const data = await getBatchStatus(jobId);
        setJob(data);
        if (data.status === 'processing') {
          // Refresh every 2s while processing
          setTimeout(loadJob, 2000);
        }
      } catch (err) {
        setError(err.message || 'Failed to load batch results');
      } finally {
        setLoading(false);
      }
    }
    loadJob();
  }, [jobId]);

  if (loading) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <div className="flex items-center gap-3 text-slate-400">
          <span className="material-symbols-outlined animate-spin">progress_activity</span>
          Loading batch results...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <div className="rounded-lg bg-red-900/30 border border-red-700 text-red-200 px-4 py-3">
          {error}
        </div>
        <Link to="/batch" className="mt-4 inline-flex items-center gap-1 text-[var(--color-primary)] hover:underline">
          <span className="material-symbols-outlined text-sm">arrow_back</span>
          Back to batches
        </Link>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <div className="text-slate-400">Batch not found</div>
        <Link to="/batch" className="mt-4 inline-flex items-center gap-1 text-[var(--color-primary)] hover:underline">
          <span className="material-symbols-outlined text-sm">arrow_back</span>
          Back to batches
        </Link>
      </div>
    );
  }

  const completedCount = job.results?.filter(r => r.status === 'complete').length ?? 0;
  const failedCount = job.results?.filter(r => r.status === 'failed').length ?? 0;
  const totalArea = job.results?.reduce((sum, r) => sum + (r.roof_area_px || 0), 0) ?? 0;

  return (
    <div className="p-4 lg:p-6 max-w-5xl mx-auto">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold flex items-center gap-2 text-[var(--color-primary)]">
            <span className="material-symbols-outlined">assessment</span>
            Batch Results
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Job ID: <code className="bg-slate-800 px-1 rounded">{jobId}</code>
          </p>
        </div>
        <Link
          to="/batch"
          className="inline-flex items-center gap-1 text-slate-400 hover:text-white text-sm"
        >
          <span className="material-symbols-outlined text-sm">arrow_back</span>
          Back to batches
        </Link>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
          <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Total</div>
          <div className="text-2xl font-bold">{job.total ?? job.results?.length ?? 0}</div>
          <div className="text-slate-500 text-xs">addresses</div>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
          <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Completed</div>
          <div className="text-2xl font-bold text-green-400">{completedCount}</div>
          <div className="text-slate-500 text-xs">successful</div>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
          <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Failed</div>
          <div className="text-2xl font-bold text-red-400">{failedCount}</div>
          <div className="text-slate-500 text-xs">errors</div>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
          <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Total Area</div>
          <div className="text-2xl font-bold text-[var(--color-primary)]">
            {Math.round(totalArea).toLocaleString()}
          </div>
          <div className="text-slate-500 text-xs">px²</div>
        </div>
      </div>

      {job.status === 'processing' && (
        <div className="mb-6 rounded-lg bg-amber-900/30 border border-amber-700 text-amber-200 px-4 py-3 flex items-center gap-2">
          <span className="material-symbols-outlined animate-spin">progress_activity</span>
          Processing {job.done ?? 0} / {job.total ?? 0} addresses...
        </div>
      )}

      {/* Results Table */}
      <div className="rounded-xl border border-slate-800 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-900 border-b border-slate-800">
              <th className="text-left py-3 px-4 text-slate-400 font-medium">#</th>
              <th className="text-left py-3 px-4 text-slate-400 font-medium">Address</th>
              <th className="text-left py-3 px-4 text-slate-400 font-medium">Roof Area</th>
              <th className="text-left py-3 px-4 text-slate-400 font-medium">Polygons</th>
              <th className="text-left py-3 px-4 text-slate-400 font-medium">Confidence</th>
              <th className="text-left py-3 px-4 text-slate-400 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {(!job.results || job.results.length === 0) && (
              <tr>
                <td colSpan={6} className="py-8 text-center text-slate-500">No results yet</td>
              </tr>
            )}
            {job.results?.map((result, idx) => (
              <tr key={idx} className="border-b border-slate-800 hover:bg-slate-900/50">
                <td className="py-3 px-4 text-slate-500">{idx + 1}</td>
                <td className="py-3 px-4 text-slate-100 max-w-xs truncate" title={result.formatted_address || result.address}>
                  {result.formatted_address || result.address}
                </td>
                <td className="py-3 px-4">
                  {result.roof_area_px ? Math.round(result.roof_area_px).toLocaleString() : '—'}
                </td>
                <td className="py-3 px-4">{result.polygons?.length || 0}</td>
                <td className="py-3 px-4">{result.confidence ? `${result.confidence}%` : '—'}</td>
                <td className="py-3 px-4">
                  <span className={`inline-flex px-2 py-0.5 rounded text-xs font-medium ${
                    result.status === 'complete'
                      ? 'bg-green-900/40 text-green-300'
                      : 'bg-red-900/40 text-red-300'
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
  );
}

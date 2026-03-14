import { useState, useCallback, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { uploadBatchCsv, getBatchStatus } from '../api';

export default function Batch() {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [dragOver, setDragOver] = useState(false);

  const refreshJob = useCallback((jobId) => {
    getBatchStatus(jobId).then((job) => {
      setJobs((prev) => prev.map((j) => (j.job_id === jobId ? job : j)));
    }).catch(() => {});
  }, []);

  const pendingJobId = jobs.find((j) => j.status === 'processing' || j.status === 'queued')?.job_id;
  useEffect(() => {
    if (!pendingJobId) return;
    const id = setInterval(async () => {
      try {
        const job = await getBatchStatus(pendingJobId);
        setJobs((prev) => prev.map((j) => (j.job_id === pendingJobId ? job : j)));
        if (job.status === 'complete' || job.status === 'failed') clearInterval(id);
      } catch {
        clearInterval(id);
      }
    }, 2000);
    return () => clearInterval(id);
  }, [pendingJobId]);

  async function handleFile(files) {
    const file = files?.[0];
    if (!file || !file.name.toLowerCase().endsWith('.csv')) {
      setError('Please select a CSV file');
      return;
    }
    setError(null);
    setUploading(true);
    try {
      const { job_id } = await uploadBatchCsv(file);
      const job = await getBatchStatus(job_id);
      setJobs((prev) => [{ ...job, name: file.name }, ...prev]);
    } catch (err) {
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="p-4 lg:p-6 max-w-5xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-bold flex items-center gap-2 text-[var(--color-primary)]">
          <span className="material-symbols-outlined">folder_zip</span>
          Batch Processing
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Upload a CSV with an <code className="bg-slate-800 px-1 rounded">address</code> or <code className="bg-slate-800 px-1 rounded">Address</code> column. Max 500 addresses per batch.
        </p>
      </div>

      <div
        className={`rounded-xl border-2 border-dashed py-12 px-6 text-center cursor-pointer transition-colors ${
          dragOver ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10' : 'border-slate-700 bg-slate-900/50 hover:bg-slate-900'
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files); }}
      >
        <span className="material-symbols-outlined text-5xl text-[var(--color-primary)]">cloud_upload</span>
        <h3 className="text-lg font-semibold mt-2">Drag and drop CSV</h3>
        <p className="text-slate-400 text-sm mt-1 mb-4">or</p>
        <label className="inline-flex items-center gap-2 rounded-lg bg-[var(--color-primary)] text-slate-950 px-4 py-2 text-sm font-semibold cursor-pointer">
          <span className="material-symbols-outlined text-lg">upload_file</span>
          Select file
          <input type="file" accept=".csv" className="hidden" onChange={(e) => handleFile(e.target.files)} disabled={uploading} />
        </label>
        <p className="text-slate-500 text-xs mt-2">Download template: use column header <code>address</code></p>
      </div>

      {error && (
        <div className="mt-4 rounded-lg bg-red-900/30 border border-red-700 text-red-200 px-4 py-2 text-sm">
          {error}
        </div>
      )}

      <div className="mt-8">
        <h2 className="text-lg font-semibold mb-3">Uploaded batches</h2>
        <div className="rounded-xl border border-slate-800 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-900 border-b border-slate-800">
                <th className="text-left py-3 px-4 text-slate-400 font-medium">Batch</th>
                <th className="text-left py-3 px-4 text-slate-400 font-medium">Addresses</th>
                <th className="text-left py-3 px-4 text-slate-400 font-medium">Status</th>
                <th className="text-left py-3 px-4 text-slate-400 font-medium">Action</th>
              </tr>
            </thead>
            <tbody>
              {jobs.length === 0 && (
                <tr>
                  <td colSpan={4} className="py-8 text-center text-slate-500">No batches yet</td>
                </tr>
              )}
              {jobs.map((job) => (
                <tr key={job.job_id} className="border-b border-slate-800 hover:bg-slate-900/50">
                  <td className="py-3 px-4">{job.name || `Job ${job.job_id.slice(0, 8)}`}</td>
                  <td className="py-3 px-4">{job.total ?? job.addresses?.length ?? 0}</td>
                  <td className="py-3 px-4">
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${
                        job.status === 'complete'
                          ? 'bg-green-900/40 text-green-300'
                          : job.status === 'failed'
                          ? 'bg-red-900/40 text-red-300'
                          : job.status === 'processing'
                          ? 'bg-amber-900/40 text-amber-300'
                          : 'bg-slate-700 text-slate-300'
                      }`}
                    >
                      {job.status === 'processing' && `${job.done ?? 0}/${job.total ?? 0} `}
                      {job.status}
                    </span>
                    {job.status === 'processing' && (
                      <button type="button" onClick={() => refreshJob(job.job_id)} className="ml-2 text-slate-400 hover:text-white text-xs">
                        Refresh
                      </button>
                    )}
                  </td>
                  <td className="py-3 px-4">
                    {job.status === 'complete' && job.results?.length > 0 && (
                      <Link
                        to="/"
                        state={{ batchResults: job.results, batchId: job.job_id }}
                        className="text-[var(--color-primary)] hover:underline inline-flex items-center gap-1"
                      >
                        View report
                        <span className="material-symbols-outlined text-sm">arrow_forward</span>
                      </Link>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

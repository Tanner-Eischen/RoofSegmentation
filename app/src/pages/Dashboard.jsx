import { useEffect, useRef, useState } from 'react';
import { runInference } from '../api';

export default function Dashboard({ onInferenceSuccess }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const previewRef = useRef('');
  useEffect(() => () => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
  }, []);
  function selectFile(event) {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    const selected = event.target.files?.[0] || null;
    previewRef.current = selected && selected.size <= 8000000 ? URL.createObjectURL(selected) : '';
    setPreview(previewRef.current);
    setFile(selected); setResult(null); setError('');
  }

  async function analyze(event) {
    event.preventDefault();
    if (!file) return;
    setError(''); setResult(null);
    if (file.size > 8000000) { setError('Choose an image smaller than 8 MB.'); return; }
    setLoading(true);
    try {
      const prediction = await runInference(file);
      setResult(prediction);
      onInferenceSuccess?.();
    }
    catch (err) { setError(err.message || 'Analysis failed. No prediction generated.'); }
    finally { setLoading(false); }
  }

  return (
    <section className="max-w-6xl mx-auto p-6 space-y-5">
      <h1 className="text-2xl font-bold">Roof segmentation</h1>
      <p>Upload aerial imagery to run the configured model. PNG, JPEG or WebP, up to 8 MB and 4,194,304 pixels.</p>
      <p className="text-slate-400">Pixel area is not physical roof area. Predictions outside the evaluated imagery domain need separate validation.</p>
      <form onSubmit={analyze} className="flex flex-wrap gap-4 items-center min-w-0">
        <label htmlFor="roof-image">Aerial image</label>
        <input id="roof-image" type="file" accept="image/png,image/jpeg,image/webp" disabled={loading}
          className="block w-full min-w-0 max-w-full sm:w-auto rounded focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-cyan-300"
          onChange={selectFile} />
        <button type="submit" disabled={!file || loading} className="rounded bg-teal-700 px-5 py-2 disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-cyan-300">
          {loading ? 'Analyzing...' : 'Analyze image'}
        </button>
      </form>
      {error && <p role="alert" className="text-red-300">{error}</p>}
      <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-5 min-w-0">
        {preview && <figure><img key={preview} src={preview} alt="Uploaded aerial imagery" className="max-w-full" onError={() => {
          if (previewRef.current === preview) {
            URL.revokeObjectURL(preview); previewRef.current = ''; setPreview('');
          }
        }} /><figcaption>Input image</figcaption></figure>}
        {result && <figure><img src={`data:image/png;base64,${result.mask_base64}`} alt="Predicted binary roof mask" className="max-w-full" /><figcaption>Predicted mask, {result.width} x {result.height} pixels</figcaption></figure>}
        {result?.probability_preview_base64 && <figure>
          <img src={`data:image/png;base64,${result.probability_preview_base64}`} alt="Predicted foreground probability" className="max-w-full" />
          <figcaption>Foreground probability: black 0, white 1. Linear 8-bit preview; no auto-scaling.</figcaption>
        </figure>}
      </div>
      {result && <div className="space-y-2">
        <p>{result.roof_area_px.toLocaleString()} roof pixels ({(100 * result.roof_area_px / (result.width * result.height)).toFixed(2)}% coverage)</p>
        <p>{result.polygons.length} external polygon contours. Holes and tiny components are omitted from simplified contours.</p>
        <p>Mean predicted roof probability: {result.mean_foreground_probability == null ? 'Not applicable (no positive pixels)' : `${(100 * result.mean_foreground_probability).toFixed(1)}%`}. This score is uncalibrated.</p>
        <p>Model: {result.model.architecture}, threshold {result.model.threshold}.</p>
        <p className="break-all">Run: {result.run_id}. Model release: {result.model.model_release_sha256}.</p>
        <p className="break-all">Input SHA-256: {result.input_sha256}. Image-space outputs; physical area unavailable.</p>
        <a download={result.artifacts?.mask.filename || "roof-mask.png"} href={`data:image/png;base64,${result.mask_base64}`} className="underline">Download predicted mask</a>
        {result.probability_base64 && <a download={result.artifacts.probability.filename} href={`data:application/octet-stream;base64,${result.probability_base64}`} className="block underline">Download float32 probability array</a>}
        {result.probability_preview_base64 && <a download={result.artifacts.probability_preview.filename} href={`data:image/png;base64,${result.probability_preview_base64}`} className="block underline">Download probability preview</a>}
      </div>}
    </section>
  );
}

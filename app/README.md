# Roof Detection Frontend

React + Vite frontend for the roof detection application.

## Development

```bash
npm install
npm run dev
```

Opens at http://localhost:5173. Requires the backend API running on port 8000.

## Pages

| Route | Description |
|-------|-------------|
| `/` | Dashboard - single address analysis with satellite view and roof detection |
| `/compare` | Side-by-side comparison of two properties |
| `/batch` | Upload CSV for bulk analysis, view job status |
| `/batch/:jobId/results` | View results from a completed batch job |

## Key Files

- `src/App.jsx` - Routes and demo mode banner
- `src/api.js` - API client functions
- `src/pages/Dashboard.jsx` - Main analysis view
- `src/pages/Batch.jsx` - CSV upload and job tracking
- `src/pages/Compare.jsx` - Property comparison
- `src/components/Layout.jsx` - Shared navigation and layout

## Build

```bash
npm run build
```

Outputs to `dist/`.

## Tech Stack

- React 19
- React Router 7
- Vite 6
- Tailwind CSS 4
- Vitest + Playwright for testing

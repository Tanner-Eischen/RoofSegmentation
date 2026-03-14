import { Link, Outlet } from 'react-router-dom';

export default function Layout() {
  return (
    <div className="flex h-screen flex-col bg-[var(--color-background-dark)] text-slate-100">
      <header className="flex shrink-0 items-center justify-between border-b border-slate-800 px-6 py-3">
        <div className="flex items-center gap-6">
          <Link to="/" className="flex items-center gap-3 text-[var(--color-primary)]">
            <span className="material-symbols-outlined text-2xl">layers</span>
            <span className="text-lg font-bold tracking-tight">Roof Detection</span>
          </Link>
          <nav className="hidden md:flex gap-6">
            <Link to="/" className="text-slate-400 hover:text-white text-sm font-medium">Dashboard</Link>
            <Link to="/compare" className="text-slate-400 hover:text-white text-sm font-medium">Compare</Link>
            <Link to="/batch" className="text-slate-400 hover:text-white text-sm font-medium">Batch</Link>
          </nav>
        </div>
        <div className="flex items-center gap-2">
          <button type="button" className="p-2 text-slate-400 hover:text-[var(--color-primary)] rounded-lg">
            <span className="material-symbols-outlined">notifications</span>
          </button>
          <button type="button" className="w-10 h-10 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
            <span className="material-symbols-outlined">account_circle</span>
          </button>
        </div>
      </header>
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}

import { Link, Outlet } from 'react-router-dom';

export default function Layout() {
  return (
    <div className="flex h-screen flex-col bg-[var(--color-background-dark)] text-slate-100">
      <header className="flex shrink-0 items-center justify-between border-b border-slate-800 px-6 py-3">
        <div className="flex items-center gap-6">
          <Link to="/" className="flex items-center gap-3 text-[var(--color-primary)]">
            <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="m3 8 9-6 9 6-9 6-9-6Zm0 5 9 6 9-6M3 18l9 6 9-6" transform="translate(0 -1) scale(1 .9)" />
            </svg>
            <span className="text-lg font-bold tracking-tight">Roof Detection</span>
          </Link>
          <nav className="hidden md:flex gap-6">
            <Link to="/" className="text-slate-400 hover:text-white text-sm font-medium">Dashboard</Link>
            <span className="text-slate-500 text-sm">Local image upload</span>
          </nav>
        </div>
      </header>
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}

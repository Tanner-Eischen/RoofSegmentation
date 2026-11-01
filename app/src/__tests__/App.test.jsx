import { render, screen, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
const check = vi.fn();
vi.mock('../api', () => ({ checkDemoMode:()=>check(), runInference:vi.fn() }));
import App from '../App';
afterEach(cleanup);
describe('App state', () => {
  it('shows an explicit sample mode', async () => {
    check.mockResolvedValue(true);
    render(<App />);
    expect(await screen.findByRole('status')).toHaveTextContent('Model inference is unavailable');
  });
  it('keeps API outages visible', async () => {
    check.mockRejectedValue(new Error('No model'));
    render(<App />);
    expect(await screen.findByRole('alert')).toHaveTextContent('No prediction has been generated');
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });
});

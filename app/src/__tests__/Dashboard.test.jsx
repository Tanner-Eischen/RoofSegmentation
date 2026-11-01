import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
const runInference = vi.fn();
vi.mock('../api', () => ({ runInference: (...args) => runInference(...args) }));
import Dashboard from '../pages/Dashboard';
beforeEach(() => { runInference.mockReset(); vi.stubGlobal('URL', { createObjectURL:vi.fn(()=>'blob:preview'), revokeObjectURL:vi.fn() }); });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
describe('Local upload workflow', () => {
  it('shows real dimensions and mask-based coverage', async () => {
    runInference.mockResolvedValue({ mask_base64:'AA==', width:100, height:200, roof_area_px:10000, polygons:[], mean_foreground_probability:null, model:{ architecture:'unet-resnet34', threshold:.5 } });
    render(<Dashboard />);
    fireEvent.change(screen.getByLabelText('Aerial image'), { target:{ files:[new File(['x'],'x.png',{type:'image/png'})] } });
    fireEvent.click(screen.getByRole('button',{name:'Analyze image'}));
    expect(await screen.findByText(/50.00% coverage/)).toBeInTheDocument();
    expect(screen.getByText(/Not applicable/)).toBeInTheDocument();
    expect(screen.getByRole('link',{name:'Download predicted mask'})).toBeInTheDocument();
  });
  it('shows outage and no result', async () => {
    runInference.mockRejectedValue(new Error('Checkpoint unavailable'));
    render(<Dashboard />);
    fireEvent.change(screen.getByLabelText('Aerial image'), { target:{ files:[new File(['x'],'x.png')] } });
    fireEvent.click(screen.getByRole('button',{name:'Analyze image'}));
    expect(await screen.findByRole('alert')).toHaveTextContent('Checkpoint unavailable');
    expect(screen.queryByAltText('Predicted binary roof mask')).not.toBeInTheDocument();
  });
});

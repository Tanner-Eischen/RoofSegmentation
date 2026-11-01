import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
const check = vi.fn();
const infer = vi.fn();
const NativeURL = globalThis.URL;
vi.mock('../api', () => ({ checkDemoMode:()=>check(), runInference:(...args)=>infer(...args) }));
import App from '../App';
beforeEach(() => {
  check.mockReset(); infer.mockReset();
  vi.stubGlobal('URL', class extends NativeURL {
    static createObjectURL = vi.fn(() => 'blob:preview');
    static revokeObjectURL = vi.fn();
  });
  infer.mockResolvedValue({ mask_base64:'AA==',width:2,height:2,roof_area_px:0,polygons:[],mean_foreground_probability:null,model:{architecture:'tiny',threshold:.5} });
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
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
  it('clears a startup outage after a successful actual request boundary', async () => {
    check.mockRejectedValue(new Error('No model'));
    render(<App />);
    expect(await screen.findByRole('alert')).toHaveTextContent('No model');
    fireEvent.change(screen.getByLabelText('Aerial image'),{target:{files:[new File(['x'],'x.png')]}});
    fireEvent.click(screen.getByRole('button',{name:'Analyze image'}));
    await screen.findByAltText('Predicted binary roof mask');
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument());
  });
  it('ignores a late failed startup check after inference has succeeded', async () => {
    let rejectCheck;
    check.mockImplementation(() => new Promise((_resolve,reject) => { rejectCheck=reject; }));
    render(<App />);
    fireEvent.change(screen.getByLabelText('Aerial image'),{target:{files:[new File(['x'],'x.png')]}});
    fireEvent.click(screen.getByRole('button',{name:'Analyze image'}));
    await screen.findByAltText('Predicted binary roof mask');
    await act(async () => rejectCheck(new Error('Stale startup failure')));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});

import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

// Authored but not run in the November 1 integration work. No success mocks.
const image = resolve('e2e/fixtures/generated-rgb.png');
const sha = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');

test.describe('Actual local-upload integration', () => {
  test('has a live, ready real checkpoint and usable upload controls', async ({ page, request }) => {
    const live = await request.get('/api/health/live');
    expect(live.ok()).toBeTruthy();
    const ready = await request.get('/api/health/ready');
    expect(ready.ok()).toBeTruthy();
    const metadata = await ready.json();
    expect(metadata.mode).toBe('real');
    expect(metadata.model.architecture).toBe('tiny');
    expect(metadata.model.checkpoint_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(metadata.model.model_release_sha256).toMatch(/^[a-f0-9]{64}$/);
    await page.goto('/');
    await expect(page.getByLabel('Aerial image')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Analyze image' })).toBeDisabled();
    await expect(page.getByRole('alert')).toHaveCount(0);
  });

  test('uploads actual bytes and downloads the exact returned model artifacts', async ({ page }) => {
    await page.goto('/');
    await page.getByLabel('Aerial image').setInputFiles(image);
    const responsePromise = page.waitForResponse(response =>
      new URL(response.url()).pathname === '/api/inference' && response.request().method() === 'POST');
    await page.getByRole('button', { name: 'Analyze image' }).click();
    const response = await responsePromise;
    expect(response.status()).toBe(200);
    const result = await response.json();
    expect(result.mode).toBe('real');
    expect(result.width).toBe(43);
    expect(result.height).toBe(29);
    expect(result.valid_pixel_count).toBe(43 * 29);
    expect(result.input_sha256).toBe(sha(await readFile(image)));
    expect(result.area).toBeNull();
    expect(result.confidence).toBeUndefined();
    expect(result.roof_area_px).toBeGreaterThanOrEqual(0);
    expect(result.roof_area_px).toBeLessThanOrEqual(43 * 29);
    await expect(page.getByAltText('Predicted binary roof mask')).toBeVisible();
    await expect(page.getByText(/Predicted mask, 43 x 29 pixels/)).toBeVisible();
    await expect(page.getByText(new RegExp(`Run: ${result.run_id}`))).toBeVisible();
    await expect(page.getByText(/This score is uncalibrated/)).toBeVisible();
    const maskEvent = page.waitForEvent('download');
    await page.getByRole('link', { name: 'Download predicted mask', exact: true }).click();
    const mask = await maskEvent;
    expect(mask.suggestedFilename()).toBe(result.artifacts.mask.filename);
    const maskPath = await mask.path();
    expect(maskPath).not.toBeNull();
    expect(sha(await readFile(maskPath!))).toBe(result.artifacts.mask.sha256);
    const probabilityEvent = page.waitForEvent('download');
    await page.getByRole('link', { name: 'Download float32 probability array' }).click();
    const probability = await probabilityEvent;
    expect(probability.suggestedFilename()).toBe(result.artifacts.probability.filename);
    const probabilityPath = await probability.path();
    expect(probabilityPath).not.toBeNull();
    expect(sha(await readFile(probabilityPath!))).toBe(result.artifacts.probability.sha256);
  });

  test('invalid uploads show a real failure and cannot leave a plausible prediction', async ({ page }) => {
    await page.goto('/');
    await page.getByLabel('Aerial image').setInputFiles({
      name: 'invalid.png', mimeType: 'image/png', buffer: Buffer.from('not an image'),
    });
    const responsePromise = page.waitForResponse(response =>
      new URL(response.url()).pathname === '/api/inference' && response.request().method() === 'POST');
    await page.getByRole('button', { name: 'Analyze image' }).click();
    expect((await responsePromise).status()).toBe(400);
    await expect(page.getByRole('alert')).toContainText('Invalid or unsupported image');
    await expect(page.getByAltText('Predicted binary roof mask')).toHaveCount(0);
    await expect(page.getByRole('link', { name: 'Download predicted mask', exact: true })).toHaveCount(0);
  });
});

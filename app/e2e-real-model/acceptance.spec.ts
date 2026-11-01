import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';
import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const run = process.env.ROOF_QA_RUN_ID || 'real-model';
if (!/^[a-z0-9-]+$/.test(run)) throw new Error('Use a simple fresh QA run identifier.');
const directory = resolve('../output/playwright/roof-wave2-2026-11-01', run);
const fixture = resolve('../reports/wave2-serving-profile-2026-11-01/clean-windows-proof/generated-fixture.png');
const expectedCheckpoint = '2c5757995dafaa206abac4e22e384cdef78b4d93ab9026975050307ac2d9c7c5';
const sha = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');

test('registered real-model UI acceptance, layouts and failure boundaries', async ({ page, context, request, browser }) => {
  await mkdir(directory, { recursive: true });
  const observations: Record<string, unknown> = {
    scope: 'Real registered local checkpoint and generated numerical fixture; no geographic/accuracy claim',
    playwright: '1.58.2', browser: browser.version(), checkpoint_sha256: expectedCheckpoint,
    screenshots: [], layouts: [], downloads: [], injected_transport_failure: true,
  };
  const shots = observations.screenshots as string[];
  async function capture(name: string) {
    const snapshot = await page.locator('body').ariaSnapshot();
    await writeFile(resolve(directory, name + '.aria.txt'), snapshot);
    await page.screenshot({ path: resolve(directory, name + '.png'), fullPage: true });
    shots.push(name + '.png');
  }
  async function analyze() {
    const received = page.waitForResponse(response => new URL(response.url()).pathname === '/api/inference' && response.request().method() === 'POST');
    await page.getByRole('button', { name: 'Analyze image', exact: true }).click();
    const response = await received;
    expect(response.status()).toBe(200);
    const value = await response.json();
    expect(value.model.checkpoint_sha256).toBe(expectedCheckpoint);
    expect(value.input_sha256).toBe(sha(await readFile(fixture)));
    await expect(page.getByAltText('Predicted binary roof mask')).toBeVisible();
    await writeFile(resolve(directory, 'latest-response-private.json'), JSON.stringify(value, null, 2));
    return value;
  }

  const ready = await request.get('/api/health/ready');
  expect(ready.status()).toBe(200);
  const readiness = await ready.json();
  expect(readiness.model.checkpoint_sha256).toBe(expectedCheckpoint);
  expect(readiness.supported_profile_ready).toBe(true);
  observations.model_release_sha256 = readiness.model.model_release_sha256;
  observations.runtime = readiness.model.runtime;
  const index = await request.get('/');
  observations.index_sha256 = sha(await index.body());
  await page.goto('/');
  await expect(page.getByLabel('Aerial image')).toBeVisible();
  const assetRoutes = await page.locator('script[src], link[rel="stylesheet"]').evaluateAll(elements => elements
    .map(element => element.getAttribute('src') || element.getAttribute('href'))
    .filter((value): value is string => !!value && value.startsWith('/assets/')));
  observations.frontend_assets = await Promise.all(assetRoutes.map(async route => {
    const response = await request.get(route);
    expect(response.status()).toBe(200);
    return { route, sha256: sha(await response.body()) };
  }));
  await capture('desktop-empty');
  await page.getByLabel('Aerial image').setInputFiles(fixture);
  const contrast = await page.getByRole('button', { name: 'Analyze image', exact: true }).evaluate(element => {
    const style = getComputedStyle(element);
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = 1;
    const context = canvas.getContext('2d')!;
    function luminance(color: string) {
      context.fillStyle = color; context.fillRect(0, 0, 1, 1);
      const rgba = context.getImageData(0, 0, 1, 1).data;
      const linear = Array.from(rgba).slice(0, 3).map(value => { const v = value / 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4; });
      return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2];
    }
    const a = luminance(style.color), b = luminance(style.backgroundColor);
    return (Math.max(a, b) + .05) / (Math.min(a, b) + .05);
  });
  expect.soft(contrast, 'Enabled Analyze text contrast').toBeGreaterThanOrEqual(4.5);
  observations.enabled_button_text_contrast = contrast;
  const result = await analyze();
  expect(result.roof_area_px).toBe(0);
  await expect(page.getByText(/0 roof pixels \(0.00% coverage\)/)).toBeVisible();
  await expect(page.getByText(/Not applicable \(no positive pixels\)/)).toBeVisible();
  await expect(page.getByText(new RegExp('Run: ' + result.run_id))).toBeVisible();
  expect.soft(await page.getByAltText('Predicted foreground probability').count(), 'Probability must be inspectable in the UI').toBe(1);
  observations.actual_result = {
    run_id: result.run_id, input_sha256: result.input_sha256, width: result.width,
    height: result.height, foreground_pixels: result.roof_area_px,
    artifacts: result.artifacts, zero_foreground_valid: true,
  };
  await capture('desktop-real-result');

  for (const [label, kind] of [['Download predicted mask', 'mask'], ['Download float32 probability array', 'probability'], ['Download probability preview', 'probability_preview']] as const) {
    const event = page.waitForEvent('download');
    await page.getByRole('link', { name: label, exact: true }).click();
    const download = await event;
    const path = await download.path();
    expect(path).not.toBeNull();
    const bytes = await readFile(path!);
    expect(sha(bytes)).toBe(result.artifacts[kind].sha256);
    expect(download.suggestedFilename()).toBe(result.artifacts[kind].filename);
    (observations.downloads as unknown[]).push({ kind, bytes: bytes.length, sha256: sha(bytes), matches_displayed_run: true });
  }

  for (const [width, height] of [[720, 900], [390, 844], [320, 740]]) {
    await page.setViewportSize({ width, height });
    await page.locator('main').evaluate(element => { element.scrollTop = 0; });
    const layout = await page.evaluate(() => {
      const main = document.querySelector('main');
      return { width: innerWidth, page_overflow: document.documentElement.scrollWidth - innerWidth,
        content_overflow: main ? main.scrollWidth - main.clientWidth : 0 };
    });
    (observations.layouts as unknown[]).push(layout);
    expect.soft(layout.page_overflow, `Page overflow at width ${width}`).toBeLessThanOrEqual(1);
    expect.soft(layout.content_overflow, `Content overflow at width ${width}`).toBeLessThanOrEqual(1);
    await capture('result-' + width);
    await page.getByRole('link', { name: 'Download predicted mask', exact: true }).scrollIntoViewIfNeeded();
    await capture('result-' + width + '-details');
  }

  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByLabel('Aerial image').setInputFiles({ name: 'invalid.png', mimeType: 'image/png', buffer: Buffer.from('invalid PNG bytes') });
  await expect(page.getByAltText('Predicted binary roof mask')).toHaveCount(0);
  await page.getByRole('button', { name: 'Analyze image', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Invalid or unsupported image');
  await expect(page.getByAltText('Uploaded aerial imagery')).toHaveCount(0);
  await expect(page.getByRole('link', { name: 'Download predicted mask', exact: true })).toHaveCount(0);
  await capture('invalid-upload');

  await page.getByLabel('Aerial image').setInputFiles({ name: 'oversized.png', mimeType: 'image/png', buffer: Buffer.alloc(8_000_001) });
  await page.getByRole('button', { name: 'Analyze image', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('smaller than 8 MB');
  await expect(page.getByAltText('Uploaded aerial imagery')).toHaveCount(0);
  await expect(page.getByAltText('Predicted binary roof mask')).toHaveCount(0);
  await capture('oversize-upload');

  // Real transport failure only: successful model responses are never mocked.
  await page.route('**/api/**', route => route.abort('failed'));
  await page.goto('/');
  await expect(page.getByRole('alert')).toContainText('Model API unavailable');
  await page.getByLabel('Aerial image').setInputFiles(fixture);
  await page.getByRole('button', { name: 'Analyze image', exact: true }).click();
  await expect(page.getByAltText('Predicted binary roof mask')).toHaveCount(0);
  await capture('backend-transport-failure');
  await page.unroute('**/api/**');
  await analyze();
  expect.soft(await page.getByRole('alert').count(), 'Recovered real inference must clear stale startup failure').toBe(0);
  await capture('backend-recovered-real-result');

  // A new file cannot retain an old mask/download; pending work locks selection.
  await page.getByLabel('Aerial image').setInputFiles({ name: 'new-generated-fixture.png', mimeType: 'image/png', buffer: await readFile(fixture) });
  await expect(page.getByAltText('Predicted binary roof mask')).toHaveCount(0);
  let delayed = false;
  await page.route('**/api/inference', async route => {
    delayed = true;
    await new Promise(resolveWait => setTimeout(resolveWait, 1200));
    await route.continue();
  });
  const pending = page.waitForResponse(response => new URL(response.url()).pathname === '/api/inference' && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Analyze image', exact: true }).click();
  await expect(page.getByLabel('Aerial image')).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Analyzing...' })).toBeDisabled();
  await expect(page.getByAltText('Predicted binary roof mask')).toHaveCount(0);
  expect((await pending).status()).toBe(200);
  await page.unroute('**/api/inference');
  await expect(page.getByAltText('Predicted binary roof mask')).toBeVisible();
  observations.superseded_result_cleared = true;
  observations.pending_input_locked = delayed;

  await page.goto('/');
  await page.keyboard.press('Tab');
  const firstFocus = await page.evaluate(() => document.activeElement?.textContent?.trim());
  observations.keyboard_first_focus = firstFocus;
  await page.getByLabel('Aerial image').focus();
  await capture('keyboard-file-focus');
  await page.getByLabel('Aerial image').setInputFiles(fixture);
  await page.keyboard.press('Tab');
  expect.soft(await page.getByRole('button', { name: 'Analyze image', exact: true }).evaluate(element => element === document.activeElement)).toBe(true);
  const keyboardResponse = page.waitForResponse(response => new URL(response.url()).pathname === '/api/inference' && response.request().method() === 'POST');
  await page.keyboard.press('Enter');
  expect((await keyboardResponse).status()).toBe(200);
  await expect(page.getByAltText('Predicted binary roof mask')).toBeVisible();
  observations.keyboard_analysis_completed = true;
  await page.emulateMedia({ reducedMotion: 'reduce' });
  expect(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(true);
  await capture('reduced-motion-real-result');
  observations.failure_count = test.info().errors.length;
  await writeFile(resolve(directory, 'observations-private.json'), JSON.stringify(observations, null, 2));
  console.log('ROOF_QA_CAPTURE_READY ' + directory);
  if (process.env.ROOF_QA_HOLD_PATH) {
    const until = Date.now() + 600_000;
    while (Date.now() < until) {
      try { await access(process.env.ROOF_QA_HOLD_PATH); break; } catch { /* wait for own visual review */ }
      await new Promise(resolveWait => setTimeout(resolveWait, 500));
    }
  }
});

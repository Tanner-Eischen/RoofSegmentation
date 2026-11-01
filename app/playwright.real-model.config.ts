import { defineConfig } from '@playwright/test';

const baseURL = process.env.ROOF_REAL_MODEL_BASE_URL;
if (!baseURL || !['127.0.0.1', 'localhost', '[::1]'].includes(new URL(baseURL).hostname)) {
  throw new Error('ROOF_REAL_MODEL_BASE_URL must identify the explicitly started localhost real-model service.');
}

export default defineConfig({
  testDir: './e2e-real-model',
  fullyParallel: false,
  workers: 1,
  timeout: 900_000,
  expect: { timeout: 8_000 },
  reporter: 'line',
  outputDir: '../output/playwright/roof-wave2-2026-11-01/real-test-results',
  use: {
    baseURL,
    headless: false,
    acceptDownloads: true,
    viewport: { width: 1440, height: 1000 },
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
});

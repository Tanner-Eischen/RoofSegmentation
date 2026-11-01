import { defineConfig, devices } from '@playwright/test';

// Authored for the local-upload app; browser execution is a separate release gate.
// The API uses an actual temporary, untrained Tiny checkpoint, never a mock success.
export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: [
    {
      command: 'python -m scripts.serve_test_fixture --port 8000',
      cwd: '..',
      url: 'http://127.0.0.1:8000/api/health/ready',
      reuseExistingServer: false,
      timeout: 120 * 1000,
    },
    {
      command: 'npm run dev -- --host localhost',
      url: 'http://localhost:5173',
      reuseExistingServer: false,
      timeout: 120 * 1000,
    },
  ],
});

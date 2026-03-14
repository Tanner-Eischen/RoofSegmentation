import { test, expect } from '@playwright/test';

test.describe('Batch Page - Bulk Processing', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/batch');
  });

  test('displays file upload section', async ({ page }) => {
    await expect(page.getByRole('heading', { name: /drag and drop csv/i })).toBeVisible();
    await expect(page.getByText(/select file/i)).toBeVisible();
  });

  test('has file input for CSV upload', async ({ page }) => {
    // File input is hidden but exists
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toBeAttached();
    await expect(fileInput).toHaveAttribute('accept', '.csv');
  });

  test('shows sample CSV format help', async ({ page }) => {
    await expect(page.getByText(/address.*column/i)).toBeVisible();
    await expect(page.getByText(/template/i)).toBeVisible();
  });

  test('upload CSV and start batch job', async ({ page }) => {
    // Mock batch upload API
    await page.route('**/api/batch/upload*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          job_id: 'test-job-123',
          status: 'queued',
          total: 3
        })
      });
    });

    // Mock batch status API
    await page.route('**/api/batch/test-job-123*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          job_id: 'test-job-123',
          status: 'queued',
          total: 3,
          done: 0
        })
      });
    });

    // Create a test CSV file
    const csvContent = 'address\n100 Main St\n200 Oak Ave\n300 Pine Rd';

    // Find file input and upload
    const fileInput = page.locator('input[type="file"]');

    // Upload the file
    await fileInput.setInputFiles({
      name: 'addresses.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csvContent)
    });

    // Should show job in table
    await expect(page.getByText('addresses.csv')).toBeVisible({ timeout: 5000 });
  });

  test('shows error for invalid file type', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');

    // Try to upload non-CSV file
    await fileInput.setInputFiles({
      name: 'test.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('not a csv')
    });

    // Should show error
    await expect(page.getByText(/please select a csv file/i)).toBeVisible({ timeout: 3000 });
  });

  test('displays uploaded batches table', async ({ page }) => {
    await expect(page.getByRole('heading', { name: /uploaded batches/i })).toBeVisible();
    await expect(page.getByText(/no batches yet/i)).toBeVisible();
  });
});

test.describe('Batch Results', () => {
  test('displays job status in table after upload', async ({ page }) => {
    await page.goto('/batch');

    // Mock batch upload API
    await page.route('**/api/batch/upload*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          job_id: 'completed-job-123',
          status: 'queued',
          total: 2
        })
      });
    });

    // Mock batch status API
    await page.route('**/api/batch/completed-job-123*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          job_id: 'completed-job-123',
          status: 'complete',
          total: 2,
          done: 2,
          results: [
            { address: '100 Main St', roof_area_px: 45000, confidence: 85.5 },
            { address: '200 Oak Ave', roof_area_px: 32000, confidence: 92.1 }
          ]
        })
      });
    });

    const csvContent = 'address\n100 Main St\n200 Oak Ave';
    const fileInput = page.locator('input[type="file"]');

    await fileInput.setInputFiles({
      name: 'test.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csvContent)
    });

    // Wait for job to appear in table
    await expect(page.getByText('test.csv')).toBeVisible({ timeout: 5000 });
  });
});

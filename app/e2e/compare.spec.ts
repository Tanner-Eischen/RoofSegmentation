import { test, expect } from '@playwright/test';

test.describe('Compare Page - Property Comparison', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/compare');
  });

  test('displays two location cards', async ({ page }) => {
    await expect(page.getByText('Location A')).toBeVisible();
    await expect(page.getByText('Location B')).toBeVisible();
  });

  test('shows analyze buttons for each location', async ({ page }) => {
    const analyzeButtons = page.getByRole('button', { name: /analyze/i });
    await expect(analyzeButtons.first()).toBeVisible();
    expect(await analyzeButtons.count()).toBe(2);
  });

  test('has address inputs for both locations', async ({ page }) => {
    const addressInputs = page.getByPlaceholder(/enter address/i);
    await expect(addressInputs.first()).toBeVisible();
    expect(await addressInputs.count()).toBe(2);
  });

  test('full comparison flow - analyze both locations', async ({ page }) => {
    // Mock APIs
    await page.route('**/api/geocode*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          lat: 39.78,
          lng: -89.65,
          formatted_address: 'Test Property'
        })
      });
    });

    await page.route('**/api/satellite*', async (route) => {
      const url = new URL(route.request().url());
      if (url.pathname.includes('/image')) {
        await route.fulfill({
          status: 200,
          contentType: 'image/png',
          body: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64')
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ url: 'http://example.com/satellite.png' })
        });
      }
    });

    await page.route('**/api/inference*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          mask_base64: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
          polygons: [],
          roof_area_px: 50000,
          confidence: 90
        })
      });
    });

    // Enter addresses for both locations
    const addressInputs = page.getByPlaceholder(/enter address/i);
    await addressInputs.first().fill('100 Main St');
    await addressInputs.nth(1).fill('200 Oak Ave');

    // Click analyze for location A
    const analyzeButtons = page.getByRole('button', { name: /analyze/i });
    await analyzeButtons.first().click();

    // Wait for confidence indicator
    await expect(page.getByText(/90% confidence/i)).toBeVisible({ timeout: 10000 });

    // Click analyze for location B
    await analyzeButtons.nth(1).click();
    await expect(page.getByText(/90% confidence/i).nth(1)).toBeVisible({ timeout: 10000 });
  });

  test('shows error when analysis fails', async ({ page }) => {
    await page.route('**/api/geocode*', async (route) => {
      await route.fulfill({ status: 404 });
    });

    const addressInputs = page.getByPlaceholder(/enter address/i);
    await addressInputs.first().fill('Invalid Address');

    const analyzeButtons = page.getByRole('button', { name: /analyze/i });
    await analyzeButtons.first().click();

    // Should show error message
    await expect(page.getByText(/failed|error/i)).toBeVisible({ timeout: 5000 });
  });
});

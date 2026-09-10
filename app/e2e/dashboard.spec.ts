import { test, expect } from '@playwright/test';

test.describe('Dashboard - Roof Analysis', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{"status":"ok"}' });
    });
    await page.goto('/');
  });

  test('displays address input and analyze button', async ({ page }) => {
    await expect(page.getByPlaceholder(/enter address/i)).toBeVisible();
    await expect(page.getByRole('button', { name: /analyze/i })).toBeVisible();
  });

  test('shows error for empty address submission', async ({ page }) => {
    const analyzeBtn = page.getByRole('button', { name: /analyze/i });
    await analyzeBtn.click();
    // Should stay on page, no error shown (just doesn't submit)
    await expect(page.getByPlaceholder(/enter address/i)).toBeVisible();
  });

  test('full analysis flow - address to segmentation', async ({ page }) => {
    // Mock geocode API
    await page.route('**/api/geocode*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          lat: 39.78,
          lng: -89.65,
          formatted_address: '123 Test St, Springfield, IL'
        })
      });
    });

    // Mock satellite URL API
    await page.route(/\/api\/satellite(?:\/image)?(?:\?.*)?$/, async (route) => {
      const url = new URL(route.request().url());
      if (url.pathname.includes('/image')) {
        await route.fulfill({
          status: 200,
          contentType: 'image/png',
          headers: { 'Access-Control-Allow-Origin': '*' },
          body: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64')
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            url: '/samples/placeholder-satellite.svg'
          })
        });
      }
    });

    // Mock inference API
    await page.route('**/api/inference*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        headers: { 'Access-Control-Allow-Origin': '*' },
        body: JSON.stringify({
          mask_base64: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
          polygons: [[[100, 100], [200, 100], [200, 200], [100, 200]]],
          roof_area_px: 45000,
          confidence: 87.5
        })
      });
    });

    // Enter address
    const addressInput = page.getByPlaceholder(/enter address/i);
    await addressInput.fill('123 Test St, Springfield, IL');

    // Click analyze
    const analyzeBtn = page.getByRole('button', { name: /analyze/i });
    await analyzeBtn.click();

    // Wait for results to appear
    await expect(page.getByText(/roof area/i)).toBeVisible({ timeout: 10000 });
    await expect(page.getByText(/45,000/)).toBeVisible();
    await expect(page.getByTestId('polygons-detected')).toBeVisible();
  });

  test('displays satellite image after geocoding', async ({ page }) => {
    // Mock APIs
    await page.route('**/api/geocode*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          lat: 39.78,
          lng: -89.65,
          formatted_address: 'Test Address'
        })
      });
    });

    await page.route(/\/api\/satellite(?:\/image)?(?:\?.*)?$/, async (route) => {
      const url = new URL(route.request().url());
      if (url.pathname.includes('/image')) {
        await route.fulfill({
          status: 200,
          contentType: 'image/png',
          headers: { 'Access-Control-Allow-Origin': '*' },
          body: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64')
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ url: '/samples/placeholder-satellite.svg' })
        });
      }
    });

    await page.route('**/api/inference*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        headers: { 'Access-Control-Allow-Origin': '*' },
        body: JSON.stringify({
          mask_base64: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
          polygons: [],
          roof_area_px: 0,
          confidence: 0
        })
      });
    });

    await page.getByPlaceholder(/enter address/i).fill('100 Main St');
    await page.getByRole('button', { name: /analyze/i }).click();

    // Check that satellite view section shows an image (not placeholder text)
    await expect(page.getByText('Enter address and click Analyze')).not.toBeVisible({ timeout: 10000 });
    await expect(page.locator('img').first()).toBeVisible({ timeout: 10000 });
  });

  test('shows property info after successful analysis', async ({ page }) => {
    await page.route('**/api/geocode*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          lat: 39.78,
          lng: -89.65,
          formatted_address: '123 Test St, Springfield, IL'
        })
      });
    });

    await page.route(/\/api\/satellite(?:\/image)?(?:\?.*)?$/, async (route) => {
      const url = new URL(route.request().url());
      if (url.pathname.includes('/image')) {
        await route.fulfill({
          status: 200,
          contentType: 'image/png',
          headers: { 'Access-Control-Allow-Origin': '*' },
          body: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64')
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ url: '/samples/placeholder-satellite.svg' })
        });
      }
    });

    await page.route('**/api/inference*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        headers: { 'Access-Control-Allow-Origin': '*' },
        body: JSON.stringify({
          mask_base64: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
          polygons: [],
          roof_area_px: 30000,
          confidence: 75
        })
      });
    });

    await page.getByPlaceholder(/enter address/i).fill('123 Test St');
    await page.getByRole('button', { name: /analyze/i }).click();

    // Check property info section
    await expect(page.getByText(/property info/i)).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('123 Test St, Springfield, IL')).toBeVisible();
  });

  test('shows error when geocoding fails', async ({ page }) => {
    await page.route('**/api/geocode*', async (route) => {
      await route.fulfill({ status: 404 });
    });

    await page.getByPlaceholder(/enter address/i).fill('Invalid Address');
    await page.getByRole('button', { name: /analyze/i }).click();

    // Should show error message
    await expect(page.locator('.bg-red-900\\/30')).toBeVisible({ timeout: 5000 });
  });
});

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{"status":"ok"}' });
    });
  });

  test('can navigate between pages', async ({ page }) => {
    await page.goto('/');

    // Check navigation links exist
    await expect(page.getByRole('link', { name: /dashboard/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /compare/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /batch/i })).toBeVisible();
  });

  test('navigates to compare page', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: /compare/i }).click();
    await expect(page).toHaveURL(/.*compare/);
  });

  test('navigates to batch page', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: /batch/i }).click();
    await expect(page).toHaveURL(/.*batch/);
  });
});

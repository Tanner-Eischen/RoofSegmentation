import { test, expect } from '@playwright/test';

// A remote-address comparison cannot masquerade as actual model measurement.
test('remote comparison is unavailable and local image upload remains reachable', async ({ page, request }) => {
  const result = await request.get('/api/satellite/image?lat=1&lng=2');
  expect(result.status()).toBe(503);
  expect((await result.json()).mask_base64).toBeUndefined();
  await page.goto('/compare');
  await expect(page.getByText('This release supports local image uploads. Return to the home page.')).toBeVisible();
  await expect(page.getByPlaceholder(/enter address/i)).toHaveCount(0);
  await page.getByRole('link', { name: 'Dashboard', exact: true }).click();
  await expect(page.getByLabel('Aerial image')).toBeVisible();
});

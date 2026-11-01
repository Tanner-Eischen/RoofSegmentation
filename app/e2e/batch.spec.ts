import { test, expect } from '@playwright/test';

// Legacy address workflows are explicitly unavailable; no successful mock jobs.
test('address batches cannot create fake jobs or expose legacy controls', async ({ page, request }) => {
  const result = await request.post('/api/batch', { data: { addresses: ['not-a-real-address'] } });
  expect(result.status()).toBe(503);
  expect((await result.json()).job_id).toBeUndefined();
  await page.goto('/batch');
  await expect(page.getByText('This release supports local image uploads. Return to the home page.')).toBeVisible();
  await expect(page.getByText('Drag and drop CSV')).toHaveCount(0);
});

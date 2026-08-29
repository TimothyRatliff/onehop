import { test, expect } from '@playwright/test';

test('page loads with title and prose', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/onehop/);
  await expect(page.locator('h1')).toHaveText('onehop');
  await expect(page.locator('section')).toHaveCount(16); // intro + 14 sections + coda
});

test('weights load and the HUD shows the model card', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#loading')).toContainText('weights loaded', { timeout: 15000 });
  await expect(page.locator('#loading')).toContainText('109,376 params');
  await expect(page.locator('#hud .hud-card')).toContainText('toy model');
});

test('module 1 renders and responds', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() =>
    document.getElementById('loading').textContent.includes('weights loaded'));
  const fig = page.locator('#fig-onehop');
  await fig.scrollIntoViewIfNeeded();
  await expect(fig.locator('canvas')).toBeVisible();
  await expect(fig.locator('.badge')).toContainText('simulation');
  // scrubbing pauses playback and pins the clock
  await page.evaluate(() => {
    const inp = document.querySelectorAll('#fig-onehop .ctl-slider input')[2];
    inp.value = '100';
    inp.dispatchEvent(new Event('input'));
  });
  await expect(fig.locator('.ctl-btn')).toHaveText('play');
  // endpoint handles are keyboard-focusable
  await fig.locator('[data-h="src"]').focus();
  await expect(fig.locator('[data-h="src"]')).toBeFocused();
});

test('figure slots exist for all 15 modules', async ({ page }) => {
  await page.goto('/');
  for (let m = 1; m <= 15; m++) {
    await expect(page.locator(`figure[data-module="${m}"]`)).toHaveCount(1);
  }
});

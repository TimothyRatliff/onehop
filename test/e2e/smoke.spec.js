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

test('module 2 computes live attention from typed input', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() =>
    document.getElementById('loading').textContent.includes('weights loaded'));
  const fig = page.locator('#fig-sdpa');
  await fig.scrollIntoViewIfNeeded();
  await expect(fig.locator('.tok')).toHaveCount(23); // "the third of march 2012"
  // hover a weight cell -> readout prints a fully-qualified trace name
  await fig.locator('.strip').nth(2).locator('.cell').first().hover();
  await expect(fig.locator('.sdpa-readout')).toContainText('enc0.attn.weights');
  // switching heads updates the aria state
  await fig.locator('.head-btn').nth(1).click();
  await expect(fig.locator('.head-btn').nth(1)).toHaveAttribute('aria-checked', 'true');
  // typing a new date recomputes the token row
  await fig.locator('input').fill('03/03/12');
  await expect(fig.locator('.tok')).toHaveCount(8);
});

test('module 3 renders three panels and the scaling toggle responds', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() =>
    document.getElementById('loading').textContent.includes('weights loaded'));
  const fig = page.locator('#fig-sqrtdk');
  await fig.scrollIntoViewIfNeeded();
  await expect(fig.locator('canvas')).toHaveCount(3);
  const toggle = fig.locator('.ctl-btn').first();
  await expect(toggle).toHaveText('scaling off');
  await toggle.click();
  await expect(toggle).toHaveText(/scaling on/);
  await expect(toggle).toHaveAttribute('aria-pressed', 'true');
});

test('module 4 renders clocks and heatmaps with live readout', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() =>
    document.getElementById('loading').textContent.includes('weights loaded'));
  const fig = page.locator('#fig-pe');
  await fig.scrollIntoViewIfNeeded();
  await expect(fig.locator('canvas')).toHaveCount(3);
  await fig.locator('[data-p="pe"]').hover({ position: { x: 60, y: 100 } });
  await expect(fig.locator('.sdpa-readout')).toContainText('PE[');
});

test('module 5 mask toggle floods the future and breaks predictions', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() =>
    document.getElementById('loading').textContent.includes('weights loaded'));
  const fig = page.locator('#fig-mask');
  await fig.scrollIntoViewIfNeeded();
  // masked: upper triangle is -inf cells and predictions are correct
  await expect(fig.locator('.mg-masked').first()).toHaveText('−∞');
  await expect(fig.locator('.mp-bad')).toHaveCount(0);
  // unmasked: no -inf cells remain, predictions collapse
  await fig.locator('.mask-toggle').click();
  await expect(fig.locator('.mg-masked')).toHaveCount(0);
  await expect(fig.locator('.mp-bad').first()).toBeVisible();
});

test('figure slots exist for all 15 modules', async ({ page }) => {
  await page.goto('/');
  for (let m = 1; m <= 15; m++) {
    await expect(page.locator(`figure[data-module="${m}"]`)).toHaveCount(1);
  }
});

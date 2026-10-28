// Run after installing Playwright and serving docs/ on port 8765.
const assert = require('node:assert/strict');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport: {width: 1365, height: 900}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const base = process.env.LAB_URL || 'http://127.0.0.1:8765/';
    await page.goto(new URL('sensor-model.html', base).href, {waitUntil: 'domcontentloaded'});
    await page.waitForSelector('#landmark-12-1');
    assert.equal(await page.locator('#map-choice option').count(), 7);
    assert.equal(await page.getAttribute('#field-fov', 'min'), '1');
    assert.equal(await page.getAttribute('#field-fov', 'max'), '360');
    assert.match(await page.textContent('#landmark-state'), /Waiting for a run/);
    assert.match(await page.textContent('#field-state'), /Waiting for a run/);
    const blank = await page.locator('#landmark-detection-plot').evaluate(canvas => canvas.toDataURL());

    await page.fill('#lm-count', '2');
    await page.fill('#lm-x', '0');
    await page.fill('#lm-y', '0');
    await page.fill('#lm-theta', '0');
    await page.fill('#lm-fov', '360');
    await page.fill('#lm-range', '20');
    await page.click('#run-landmark');
    assert.match(await page.textContent('#landmark-state'), /Run 1 · 2 detected/);
    assert.match(await page.textContent('#landmark-sample-note'), /500 sampled poses from landmark 1/);
    assert.equal(await page.locator('#landmark-readings tr').count(), 3);
    assert.notEqual(await page.locator('#landmark-detection-plot').evaluate(canvas => canvas.toDataURL()), blank);

    await page.selectOption('#map-choice', 'map3.png');
    await page.click('#run-field');
    await page.waitForFunction(() => document.querySelector('#field-state').textContent.includes('Run 1'));
    assert.equal(await page.locator('#field-readings tr').count(), 13);
    assert.equal(await page.locator('#field-error').isHidden(), true);
    await page.fill('#weight-hit', '0');
    await page.fill('#weight-max', '1');
    await page.fill('#weight-rand', '0');
    assert.equal(await page.textContent('#weight-summary'), '0.000 / 1.000 / 0.000');
    await page.fill('#field-fov', '360');
    await page.fill('#field-num-rays', '24');
    await page.fill('#field-max-range', '6');
    await page.click('#run-field');
    assert.match(await page.textContent('#field-state'), /Run 2 · map3\.png · 24 rays/);
    assert.equal(await page.locator('#field-readings tr').count(), 25);
    const ranges = await page.locator('#field-readings tr td:nth-child(2)').allTextContents();
    assert.ok(ranges.every(value => Number(value) <= 6), 'No measured range may exceed the chosen maximum');
    await page.screenshot({path: 'sensor-desktop.png', fullPage: true});

    await page.setViewportSize({width: 390, height: 844});
    await page.waitForTimeout(100);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1), 'Sensor page must not overflow on mobile');
    await page.screenshot({path: 'sensor-mobile.png', fullPage: true});
    assert.deepEqual(errors, []);
    console.log('Sensor landmark, map selection, likelihood field, controls, and mobile checks passed.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

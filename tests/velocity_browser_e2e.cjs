// Run after: npm install --no-save playwright@1.56.1 && npx playwright install chromium
// Local server: python -m http.server 8765 --directory docs
const assert = require('node:assert/strict');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
  const context = await browser.newContext({viewport: {width: 1365, height: 1000}});
  await context.addInitScript(() => {
    window.createdWorkers = 0;
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker { constructor(...args) { super(...args); window.createdWorkers++; } };
  });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const base = process.env.LAB_URL || 'http://127.0.0.1:8765/';
  await page.goto(base, {waitUntil: 'domcontentloaded'});
  await page.waitForSelector('#v');
  assert.equal(await page.getAttribute('html', 'lang'), 'en');
  for (const [key, min, max] of [['v', '0', '2'], ['omega', '-1.5', '1.5'], ['dt', '0.01', '1']]) {
    for (const id of [key, `${key}-slider`]) {
      assert.equal(await page.getAttribute(`#${id}`, 'min'), min);
      assert.equal(await page.getAttribute(`#${id}`, 'max'), max);
    }
  }
  assert.equal(await page.locator('#how-it-works .theory-note').count(), 1);
  assert.match(await page.locator('#how-it-works').innerText(), /One command, many outcomes/);
  await page.fill('#v', '0.801');
  await page.locator('#v').dispatchEvent('change');
  await page.waitForTimeout(500);
  assert.equal(await page.evaluate(() => window.createdWorkers), 0, 'Editing parameters must not start Python');
  assert.equal(await page.getAttribute('body', 'data-run-count'), '0');
  assert.equal(await page.inputValue('#v-slider'), '0.801');
  await page.selectOption('#noise-mode', 'sigma');
  for (const [key, value] of Object.entries({v: '.8', omega: '0', sigma_v: '0', sigma_omega: '0', sigma_gamma: '0', samples: '50', steps: '2'})) await page.fill('#'+key, value);
  await page.click('#run');
  await page.waitForFunction(() => document.body.dataset.runCount === '1', null, {timeout: 240000});
  assert.equal(await page.textContent('#mean-x'), '2.800');
  assert.equal(await page.textContent('#mean-y'), '4.000');
  await page.fill('#v', '1.200');
  await page.waitForTimeout(500);
  assert.equal(await page.getAttribute('body', 'data-run-count'), '1');
  assert.equal(await page.textContent('#mean-x'), '2.800', 'Old output must remain labeled with its executed parameters');
  assert.match(await page.textContent('#result-state'), /Parameters changed/);
  await page.click('#run');
  await page.waitForFunction(() => document.body.dataset.runCount === '2', null, {timeout: 30000});
  assert.equal(await page.textContent('#mean-x'), '3.200');
  await page.click('#play'); await page.waitForTimeout(1500);
  assert.equal(await page.getAttribute('body', 'data-run-count'), '2', 'Playback must not rerun the algorithm');
  const downloadPromise = page.waitForEvent('download'); await page.click('#download');
  const download = await downloadPromise; assert.match(download.suggestedFilename(), /velocity-run-2/);
  await page.screenshot({path: process.env.LAB_SCREENSHOT || 'velocity-desktop.png', fullPage: true});
  await page.setViewportSize({width: 390, height: 844});
  await page.waitForTimeout(300);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1), 'Mobile layout must not overflow');
  await page.screenshot({path: 'velocity-mobile.png', fullPage: true});
  assert.deepEqual(errors, []);
  await browser.close();
  console.log('Browser execution, manual run, slider values, replay, export and mobile checks passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });

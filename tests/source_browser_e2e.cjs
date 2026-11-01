// Run after installing Playwright and serving docs/ on port 8765.
const assert=require('node:assert/strict');
const {chromium}=require('playwright');

(async()=>{
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  try{
    const page=await browser.newPage({viewport:{width:1200,height:850}});
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    const base=process.env.LAB_URL||'http://127.0.0.1:8765/';
    await page.goto(new URL('source-code.html',base).href,{waitUntil:'domcontentloaded'});
    await page.waitForSelector('#source-commit a');
    assert.equal(await page.locator('.source-module').count(),7);
    const fileCount=await page.evaluate(async()=>{const response=await fetch('./source/manifest.json');return (await response.json()).files.length;});
    assert.equal(await page.textContent('#source-file-count'),`${fileCount} FILES`);
    await page.getByRole('link',{name:'velocity_motion_model.py'}).click();
    await page.waitForFunction(()=>document.querySelector('#preview-content')?.textContent.includes('def sample_velocity_motion_model'));
    assert.match(await page.textContent('#preview-title'),/Motion_Models\/velocity_motion_model\.py/);
    assert.deepEqual(errors,[]);
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Source catalogue must fit mobile width');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

// Run after installing Playwright and serving docs/ on port 8765.
const assert=require('node:assert/strict');
const {chromium}=require('playwright');

(async()=>{
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  try{
    const page=await browser.newPage({viewport:{width:1365,height:900}});
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    const base=process.env.LAB_URL||'http://127.0.0.1:8765/';
    await page.goto(new URL('gaussian-filters.html',base).href,{waitUntil:'domcontentloaded'});
    await page.waitForSelector('#sigmaBearing');
    assert.equal(await page.locator('#ekf-landmark-rows .ekf-landmark-row').count(),10);
    assert.equal(await page.getAttribute('html','lang'),'en');
    assert.match(await page.textContent('#ekf-state'),/Waiting for a run/);
    for(const [id,min,max,step] of [
      ['v','0','2','0.01'],['omega','-1.5','1.5','0.01'],
      ['sigmaV','0','1','0.01'],['sigmaOmega','0','1','0.01'],
      ['sigmaRange','0.05','1','0.01'],['sigmaBearing','1','45','0.01']
    ]){
      for(const control of [id,id+'-slider']){
        assert.equal(await page.getAttribute('#'+control,'min'),min);
        assert.equal(await page.getAttribute('#'+control,'max'),max);
        assert.equal(await page.getAttribute('#'+control,'step'),step);
      }
    }
    const numerical=await page.evaluate(async()=>{
      const {runEkfSimulation}=await import(new URL('./gaussian-filters-core.js',location.href).href);
      const p={v:.8,omega:.03,sigmaV:.1,sigmaOmega:.02,fov:60,maxRange:8,sigmaRange:.1,sigmaBearing:1,ekfDt:1,duration:22,seed:42};
      const a=runEkfSimulation(p),b=runEkfSimulation(p),zero=runEkfSimulation({...p,omega:0,duration:2});
      const custom=runEkfSimulation({...p,landmarks:[[5,12]]});
      const detections=a.frames.filter(f=>f.detected.length>0);
      return {frames:a.frames.length,updates:a.ekfErrors.length,detected:detections.length,
        reduced:detections.some(f=>f.covariance[0][0]+f.covariance[1][1]<f.prior.covariance[0][0]+f.prior.covariance[1][1]),
        deterministic:JSON.stringify(a.ekfErrors)===JSON.stringify(b.ekfErrors),
        finite:zero.frames.every(f=>f.estimate.every(Number.isFinite)),
        snapshotTimes:a.snapshotIndices.map(i=>a.frames[i].time),
        snapshotPairs:a.snapshotIndices.every(i=>a.frames[i].prior&&a.frames[i].update),
        customMap:custom.landmarks.length===1&&custom.landmarks[0][0]===5};
    });
    assert.equal(numerical.frames,221);
    assert.equal(numerical.updates,22);
    assert.ok(numerical.detected>0&&numerical.reduced);
    assert.ok(numerical.deterministic&&numerical.finite);
    assert.equal(numerical.snapshotTimes.length,3);
    for(const [time,target] of numerical.snapshotTimes.map((time,i)=>[time,22*(i+1)/3])) assert.ok(Math.abs(time-target)<=1);
    assert.ok(numerical.snapshotPairs&&numerical.customMap);
    const blank=await page.locator('#ekf-trajectory').evaluate(canvas=>canvas.toDataURL());
    await page.click('#run-ekf');
    assert.match(await page.textContent('#ekf-state'),/Run 1 complete/);
    assert.equal(await page.getAttribute('#ekf-frame','max'),'220');
    await page.locator('#ekf-frame').evaluate(input=>{input.value='11';input.dispatchEvent(new Event('input',{bubbles:true}));});
    assert.match(await page.textContent('#ekf-event'),/landmark.*detected/i);
    assert.notEqual(await page.locator('#ekf-trajectory').evaluate(canvas=>canvas.toDataURL()),blank);
    assert.notEqual(await page.textContent('#ekf-position-error'),'—');
    await page.fill('#sigmaRange','0.56');
    assert.match(await page.textContent('#ekf-state'),/Parameters changed/);
    await page.click('#ekf-clear-landmarks');
    assert.equal(await page.locator('#ekf-landmark-rows .ekf-landmark-row').count(),0);
    await page.locator('#ekf-landmark-map').click({position:{x:100,y:80}});
    assert.equal(await page.locator('#ekf-landmark-rows .ekf-landmark-row').count(),1);
    await page.fill('#ekf-landmark-0-0','5.0');
    await page.fill('#ekf-landmark-0-1','12.0');
    await page.click('#run-ekf');
    assert.match(await page.textContent('#ekf-state'),/Run 2 complete/);
    const saved=JSON.parse(await page.textContent('#ekf-last-parameters'));
    assert.deepEqual(saved.landmarks,[[5,12]]);
    await page.click('#ekf-reset-landmarks');
    assert.equal(await page.locator('#ekf-landmark-rows .ekf-landmark-row').count(),10);
    assert.match(await page.textContent('#ekf-state'),/Parameters changed/);
    await page.screenshot({path:'ekf-desktop.png',fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.waitForTimeout(100);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'EKF page must not overflow on mobile');
    await page.screenshot({path:'ekf-mobile.png',fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('EKF model, uncertainty updates, playback, controls, and responsive layout checks passed.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

// Run after installing Playwright and serving docs/ on port 8765.
const assert=require('node:assert/strict');
const {chromium}=require('playwright');

(async()=>{
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  try{
    const page=await browser.newPage({viewport:{width:1365,height:900}});
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    const base=process.env.LAB_URL||'http://127.0.0.1:8765/';
    await page.goto(new URL('discrete-filters.html',base).href,{waitUntil:'domcontentloaded'});
    await page.waitForSelector('#ekf-landmark-9-1');
    await page.waitForFunction(()=>document.querySelector('#startX')?.max==='30');
    assert.equal(await page.getAttribute('html','lang'),'en');
    assert.equal(await page.locator('.sensor-experiment').count(),2);
    assert.doesNotMatch(await page.locator('#pf-landmark .legend').first().textContent(),/Predicted 95%|Corrected 95%/);
    assert.doesNotMatch(await page.locator('#pf-laser .legend').first().textContent(),/Predicted 95%|Corrected 95%/);
    assert.equal(await page.locator('#lm-map-choice').count(),0);
    assert.equal(await page.locator('#lm-scripted-turn').count(),0);
    assert.doesNotMatch(await page.locator('#pf-landmark').innerText(),/occupancy map|map3\.png|free cells/i);
    assert.equal(await page.locator('#laser-map-choice option').count(),7);
    for(const [id,min,max,step] of [
      ['lm-v','0','2','0.01'],['lm-omega','-1.5','1.5','0.01'],
      ['lm-sigma-v','0','1','0.01'],['lm-sigma-omega','0','1','0.01'],
      ['lm-sigma-range','0.05','1','0.01'],['lm-sigma-bearing','1','45','0.01'],
      ['laser-sigma-hit','0.05','2','0.01'],['laser-fov','1','360','1'],
      ['laser-num-rays','1','360','1'],['startX','0','30','0.1'],['startY','0','30','0.1'],
      ['lm-particles','50','2000','1'],['laser-particles','50','2000','1']
    ]){
      for(const control of [id,id+'-slider']){
        assert.equal(await page.getAttribute('#'+control,'min'),min);
        assert.equal(await page.getAttribute('#'+control,'max'),max);
        assert.equal(await page.getAttribute('#'+control,'step'),step);
      }
    }
    const model=await page.evaluate(async()=>{
      const {PF_LANDMARKS,loadPfMap,runPfSimulation}=await import(new URL('./pf-core.js',location.href).href);
      const laserMap=await loadPfMap('map31_grey.png');
      const common={v:.8,omega:.03,sigmaV:.1,sigmaOmega:.02,fov:90,maxRange:8,pfDt:1,duration:4,particles:100,seed:42,scriptedTurn:false};
      const p={...common,mode:'landmark',initialPose:[1,6,.3],sigmaRange:.1,sigmaBearing:1,landmarks:PF_LANDMARKS};
      const first=runPfSimulation(p),second=runPfSimulation(p);
      const laser=runPfSimulation({...common,mode:'laser',v:.33,omega:0,sigmaV:.05,sigmaOmega:.02,fov:360,scriptedTurn:true,
        initialPose:[19.5,4.5,5*Math.PI/6],numRays:12,sigmaLaser:.1,sigmaHit:1,weightHit:.8,weightMax:0,weightRand:.2},laserMap);
      const larger=runPfSimulation({...p,particles:600,duration:2});
      const {EKF_LANDMARKS}=await import(new URL('./gaussian-filters-core.js',location.href).href);
      const laserRays=laser.frames.find(f=>f.rays)?.rays||[];
      const validLaserRays=laserRays.filter(ray=>ray.probability!==null);
      return {landmarks:PF_LANDMARKS.length,sameAsEkf:JSON.stringify(PF_LANDMARKS)===JSON.stringify(EKF_LANDMARKS),
        cloud:larger.frames[0].cloud.length,laserMapSize:[laserMap.rows,laserMap.cols],landmarkWorld:first.world,landmarkMap:first.map,initialEstimate:first.frames[0].estimate,noLandmarkMapFile:!('mapFile' in first.parameters),frames:[first.frames.length,laser.frames.length],
        updates:[first.pfErrors.length,laser.pfErrors.length],
        detected:first.frames.reduce((n,f)=>n+f.detected.length,0),
        laserRays:laserRays.length,validLaserRays:validLaserRays.length,
        laserProbabilities:validLaserRays.every(ray=>Number.isFinite(ray.probability)&&ray.probability>=0&&ray.probability<=1),
        initialParticlesUniformArea:larger.frames[0].cloud.every(([x,y])=>x>=0&&x<20&&y>=0&&y<20),
        deterministic:JSON.stringify(first.pfErrors)===JSON.stringify(second.pfErrors),
        finite:[first,laser].every(run=>run.frames.every(f=>f.estimate.every(Number.isFinite))),
        simGtErrors:[first.simGtErrors.length,laser.simGtErrors.length],
        noOdometry:[first,laser].every(run=>!('odometry' in run.frames[0]))};
    });
    assert.equal(model.landmarks,10);
    assert.ok(model.sameAsEkf);
    assert.equal(model.cloud,500);
    assert.ok(model.initialParticlesUniformArea);
    assert.deepEqual(model.laserMapSize,[22,22]);
    assert.deepEqual(model.landmarkWorld,{rows:30,cols:30});
    assert.equal(model.landmarkMap,null);
    assert.deepEqual(model.initialEstimate,[1,6,.3]);
    assert.ok(model.noLandmarkMapFile);
    assert.deepEqual(model.frames,[41,41]);
    assert.deepEqual(model.updates,[4,4]);
    assert.ok(model.detected>0);
    assert.equal(model.laserRays,12);
    assert.ok(model.validLaserRays>0&&model.laserProbabilities);
    assert.ok(model.deterministic&&model.finite&&model.noOdometry);
    assert.deepEqual(model.simGtErrors,[41,41]);

    const blank=await page.locator('#lm-trajectory').evaluate(canvas=>canvas.toDataURL());
    await page.fill('#lm-duration','8');
    await page.fill('#lm-particles','101');
    await page.click('#lm-run');
    await page.waitForFunction(()=>document.querySelector('#lm-state').textContent.includes('Run 1'));
    assert.equal(await page.getAttribute('#lm-frame','max'),'80');
    await page.locator('#lm-frame').evaluate(input=>{input.value='11';input.dispatchEvent(new Event('input',{bubbles:true}));});
    assert.match(await page.textContent('#lm-event'),/effective particles/);
    assert.notEqual(await page.locator('#lm-trajectory').evaluate(canvas=>canvas.toDataURL()),blank);
    assert.notEqual(await page.textContent('#lm-pf-error'),'—');
    await page.click('#ekf-clear-landmarks');
    assert.equal(await page.locator('#ekf-landmark-rows .ekf-landmark-row').count(),0);
    assert.match(await page.textContent('#lm-state'),/Parameters changed/);
    await page.click('#ekf-reset-landmarks');
    assert.equal(await page.locator('#ekf-landmark-rows .ekf-landmark-row').count(),10);

    await page.click('#ekf-click-robot');
    assert.equal(await page.getAttribute('#ekf-click-robot','aria-pressed'),'true');
    await page.locator('#ekf-landmark-map').evaluate(canvas=>{
      const r=canvas.getBoundingClientRect(),x=r.left+29+2/30*(r.width-41),y=r.top+r.height-27-6.5/30*(r.height-39);
      canvas.dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:x,clientY:y}));
    });
    assert.equal(await page.inputValue('#startX'),'2.0');
    assert.equal(await page.inputValue('#startY'),'6.5');
    await page.fill('#laser-duration','4');
    await page.fill('#laser-particles','101');
    await page.fill('#laser-num-rays','12');
    await page.click('#laser-run');
    await page.waitForFunction(()=>document.querySelector('#laser-state').textContent.includes('Run 1'));
    assert.equal(await page.getAttribute('#laser-frame','max'),'40');
    await page.locator('#laser-frame').evaluate(input=>{input.value='11';input.dispatchEvent(new Event('input',{bubbles:true}));});
    assert.match(await page.textContent('#laser-event'),/12 laser rays/);
    assert.notEqual(await page.textContent('#laser-pf-error'),'—');
    await page.screenshot({path:'pf-desktop.png',fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.waitForTimeout(100);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Particle filter page must not overflow on mobile');
    await page.screenshot({path:'pf-mobile.png',fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('Landmark and laser PF simulations, controls, playback, map editor, and mobile layout passed.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

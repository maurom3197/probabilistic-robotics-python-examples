import {PF_LANDMARKS,loadPfMap,runPfSimulation} from './pf-core.js?v=4';
import {drawPfTrajectory,drawPfError,drawPfMapPreview,previewPoint} from './pf-plots.js?v=4';
import {createLandmarkEditor} from './gaussian-filters-map.js?v=5';

const $=id=>document.getElementById(id),DEG=Math.PI/180;
const DEFINITIONS=[
  ['lm-v','Linear velocity v (m/s)',0,2,.8,.01,'lm-command-controls','v'],
  ['lm-omega','Angular velocity ω (rad/s)',-1.5,1.5,.03,.01,'lm-command-controls','omega'],
  ['lm-sigma-v','Velocity σv (m/s)',0,1,.1,.01,'lm-motion-controls','sigmaV'],
  ['lm-sigma-omega','Angular velocity σω (rad/s)',0,1,.02,.01,'lm-motion-controls','sigmaOmega'],
  ['lm-fov','Field of view (°)',5,360,90,1,'lm-sensor-controls','fov'],
  ['lm-max-range','Maximum range (m)',.1,20,8,.1,'lm-sensor-controls','maxRange'],
  ['lm-sigma-range','Range σ (m)',.05,1,.1,.01,'lm-sensor-controls','sigmaRange'],
  ['lm-sigma-bearing','Bearing σ (°)',1,45,1,.01,'lm-sensor-controls','sigmaBearing'],
  ['startX','Start x (m)',0,30,1,.1,'ekf-start-pose-controls','startX'],
  ['startY','Start y (m)',0,30,6,.1,'ekf-start-pose-controls','startY'],
  ['startTheta','Start heading θ (°)',-180,180,17.19,.01,'ekf-start-pose-controls','startTheta'],
  ['lm-pf-dt','PF interval (s)',.1,1,1,.1,'lm-experiment-controls','pfDt'],
  ['lm-duration','Duration (s)',2,60,22,1,'lm-experiment-controls','duration'],
  ['lm-particles','Particle count N',50,2000,1000,1,'lm-particle-controls','particles'],
  ['lm-seed','Random seed',0,999999,42,1,'lm-experiment-controls','seed'],
  ['laser-x','Start x / map row',0,21.9,19.5,.1,'laser-pose-controls','startX'],
  ['laser-y','Start y / map column',0,21.9,4.5,.1,'laser-pose-controls','startY'],
  ['laser-theta','Start heading θ (°)',-180,180,150,.01,'laser-pose-controls','startTheta'],
  ['laser-v','Linear velocity v (cells/s)',0,2,.33,.01,'laser-command-controls','v'],
  ['laser-omega','Angular velocity ω (rad/s)',-1.5,1.5,0,.01,'laser-command-controls','omega'],
  ['laser-sigma-v','Velocity σv (cells/s)',0,1,.05,.01,'laser-motion-controls','sigmaV'],
  ['laser-sigma-omega','Angular velocity σω (rad/s)',0,1,.02,.01,'laser-motion-controls','sigmaOmega'],
  ['laser-fov','Field of view (°)',1,360,360,1,'laser-sensor-controls','fov'],
  ['laser-num-rays','Number of rays',1,360,30,1,'laser-sensor-controls','numRays'],
  ['laser-max-range','Maximum range (cells)',.5,30,8,.1,'laser-sensor-controls','maxRange'],
  ['laser-sigma-laser','Laser range σ (cells)',.05,1,.1,.01,'laser-noise-controls','sigmaLaser'],
  ['laser-sigma-hit','Hit σ (cells)',.05,2,1,.01,'laser-noise-controls','sigmaHit'],
  ['laser-weight-hit','z_hit',0,1,.8,.01,'laser-weight-controls','weightHit'],
  ['laser-weight-max','z_max',0,1,0,.01,'laser-weight-controls','weightMax'],
  ['laser-weight-rand','z_rand',0,1,.2,.01,'laser-weight-controls','weightRand'],
  ['laser-pf-dt','PF interval (s)',.1,1,1,.1,'laser-experiment-controls','pfDt'],
  ['laser-duration','Duration (s)',2,60,60,1,'laser-experiment-controls','duration'],
  ['laser-particles','Particle count N',50,2000,1000,1,'laser-particle-controls','particles'],
  ['laser-seed','Random seed',0,999999,42,1,'laser-experiment-controls','seed'],
];
const state={lm:{result:null,index:0,timer:null,runs:0},laser:{result:null,index:0,timer:null,runs:0,map:null,request:0}};
const mapCache=new Map();
let editor=null;
function markDraft(mode){
  const s=state[mode],badge=$(mode+'-state');
  badge.textContent=s.result?'Parameters changed · press Run':'Waiting for a run';
  badge.className='badge'+(s.result?' pending':'');
  if(mode==='laser')drawLaserPreview();
}
function makeControl([id,label,min,max,initial,step,host]){
  const mode=id.startsWith('laser-')?'laser':'lm';
  const row=document.createElement('div');row.className='parameter-row';
  const header=document.createElement('div');header.className='parameter-header';
  const text=document.createElement('label');text.htmlFor=id;text.textContent=label;
  const digits=step===.01?2:step===.1?1:0;
  const format=value=>step===1||step>=1?String(Math.round(Number(value))):Number(value).toFixed(digits);
  const number=document.createElement('input');
  Object.assign(number,{type:'number',id,min,max,step,value:format(initial)});
  const range=document.createElement('input');
  Object.assign(range,{type:'range',id:id+'-slider',min,max,step,value:String(initial)});
  range.setAttribute('aria-label',label+' slider');
  number.addEventListener('input',()=>{if(Number.isFinite(number.valueAsNumber))range.value=number.value;markDraft(mode);});
  number.addEventListener('change',()=>{
    const value=number.valueAsNumber;
    if(Number.isFinite(value)&&value>=min&&value<=max){
      number.value=format(Math.max(min,Math.min(max,min+Math.round((value-min)/step)*step)));
      range.value=number.value;
    }
    markDraft(mode);
  });
  range.addEventListener('input',()=>{number.value=format(range.value);markDraft(mode);});
  header.append(text,number);row.append(header,range);
  const limits=document.createElement('div');limits.className='range-limits';
  const lo=document.createElement('span'),hi=document.createElement('span');
  lo.textContent=String(min);hi.textContent=String(max);limits.append(lo,hi);row.append(limits);
  $(host).append(row);
}
function readParameters(mode){
  const p={mode:mode==='lm'?'landmark':'laser',scriptedTurn:mode==='laser'&&$('laser-scripted-turn').checked};
  if(mode==='laser')p.mapFile=$('laser-map-choice').value;
  for(const [id,label,min,max,initial,step,host,key] of DEFINITIONS){
    if(mode==='lm'&&id.startsWith('laser-')||mode==='laser'&&!id.startsWith('laser-'))continue;
    const input=$(id),value=input.valueAsNumber;
    if(!Number.isFinite(value)||value<min||value>max||input.validity.stepMismatch){
      input.focus();throw new Error(label+': enter a value between '+min+' and '+max+' in the slider increments.');
    }
    p[key]=value;
  }
  p.initialPose=[p.startX,p.startY,p.startTheta*DEG];
  delete p.startX;delete p.startY;delete p.startTheta;
  if(mode==='lm')p.landmarks=editor.read();
  else if(p.weightHit+p.weightRand<=0)throw new Error('Enter a positive hit or random weight; maximum-range beams are ignored.');
  return p;
}
async function getMap(filename){
  if(!mapCache.has(filename))mapCache.set(filename,loadPfMap(filename).catch(error=>{mapCache.delete(filename);throw error;}));
  return mapCache.get(filename);
}
function setInput(id,value){
  const input=$(id),digits=Number(input.step)===.01?2:1;
  input.value=Number(value).toFixed(digits);$(id+'-slider').value=String(value);
}
function freePose(map,pose){
  const [x,y]=pose;
  return x>=0&&x<map.rows&&y>=0&&y<map.cols&&map.cells[Math.floor(x)][Math.floor(y)]===0;
}
async function selectMap(){
  const s=state.laser,request=++s.request,filename=$('laser-map-choice').value;
  try{
    const map=await getMap(filename);
    if(request!==s.request)return;
    s.map=map;
    const ids=['laser-x','laser-y'];
    ids.forEach((id,axis)=>{
      const max=(axis===0?map.rows:map.cols)-.1;
      $(id).min='0';$(id).max=String(max);
      $(id+'-slider').min='0';$(id+'-slider').max=String(max);
    });
    const pose=ids.map(id=>$(id).valueAsNumber);
    if(!freePose(map,pose)){
      const nearest=map.free.reduce((best,[x,y])=>{
        const d=(x+.5-pose[0])**2+(y+.5-pose[1])**2;
        return !best||d<best.d?{x:x+.5,y:y+.5,d}:best;
      },null);
      if(nearest){setInput(ids[0],nearest.x);setInput(ids[1],nearest.y);}
    }
    markDraft('laser');
  }catch(error){
    if(request!==s.request)return;
    s.map=null;$('laser-form-error').textContent=error.message;$('laser-form-error').hidden=false;
  }
}
function drawLaserPreview(){
  const map=state.laser.map;
  drawPfMapPreview($('laser-map-preview'),map,[$('laser-x').valueAsNumber,$('laser-y').valueAsNumber,$('laser-theta').valueAsNumber]);
}
function stop(mode){
  const s=state[mode];if(s.timer!==null)clearInterval(s.timer);
  s.timer=null;$(mode+'-play').textContent='▶ Play';
}
function display(mode,index){
  const s=state[mode],result=s.result;if(!result)return;
  s.index=Math.max(0,Math.min(index,result.frames.length-1));
  const f=result.frames[s.index],latest=result.frames.slice(0,s.index+1).reverse().find(item=>item.update);
  $(mode+'-frame').value=String(s.index);$(mode+'-time').textContent='t = '+f.time.toFixed(1)+' s';
  const sensor=mode==='lm'?latest?.detected.length+' landmark'+(latest?.detected.length===1?'':'s')+' detected':latest?.rays?.length+' laser rays';
  $(mode+'-event').textContent=latest
    ? 'Latest update: '+sensor+' · effective particles '+latest.neff.toFixed(0)+(latest.resampled?' · resampled':'')+'.'
    : mode==='lm'?'Particles start uniformly within 0–20 m before the first filter update.':'Particles start over free map cells before the first filter update.';
  $(mode+'-pf-error').textContent=f.pfError.toFixed(2);
  $(mode+'-sim-gt-error').textContent=f.simGtError.toFixed(2);
  $(mode+'-sigma').textContent=Math.sqrt(Math.max(0,f.covariance[0][0])).toFixed(2)+' / '+Math.sqrt(Math.max(0,f.covariance[1][1])).toFixed(2);
  $(mode+'-neff').textContent=latest?latest.neff.toFixed(0):String(result.parameters.particles);
  drawPfTrajectory($(mode+'-trajectory'),result,s.index);
  drawPfError($(mode+'-error-chart'),result,s.index);
}
function play(mode){
  const s=state[mode],result=s.result;if(!result)return;
  if(s.timer!==null){stop(mode);return;}
  if(s.index>=result.frames.length-1)display(mode,0);
  $(mode+'-play').textContent='Ⅱ Pause';
  let last=performance.now(),carry=0;
  s.timer=setInterval(()=>{
    const now=performance.now();carry+=now-last;last=now;
    const period=Math.max(16,100/Number($(mode+'-speed').value));
    const count=Math.floor(carry/period);if(!count)return;
    carry-=count*period;display(mode,s.index+count);
    if(s.index>=result.frames.length-1)stop(mode);
  },16);
}
async function run(mode){
  const error=$(mode+'-form-error'),button=$(mode+'-run');
  error.hidden=true;button.disabled=true;
  try{
    const p=readParameters(mode),map=mode==='laser'?await getMap(p.mapFile):null;
    await new Promise(resolve=>setTimeout(resolve,0));
    const result=runPfSimulation(p,map),s=state[mode];
    stop(mode);s.result=result;s.runs++;
    const badge=$(mode+'-state');badge.textContent='Run '+s.runs+' complete';badge.className='badge done';
    $(mode+'-last-parameters').textContent=JSON.stringify(p,null,2);
    $(mode+'-frame').max=String(result.frames.length-1);$(mode+'-frame').disabled=false;$(mode+'-play').disabled=false;
    display(mode,0);play(mode);
  }catch(cause){error.textContent=cause.message;error.hidden=false;}
  finally{button.disabled=false;}
}
function init(){
  DEFINITIONS.forEach(makeControl);
  editor=createLandmarkEditor(()=>markDraft('lm'),{defaultLandmarks:PF_LANDMARKS,downloadName:'pf-landmarks.json',axisRange:[0,30]});
  for(const mode of ['lm','laser']){
    $(mode+'-form').addEventListener('submit',event=>{event.preventDefault();run(mode);});
    $(mode+'-play').addEventListener('click',()=>play(mode));
    $(mode+'-frame').addEventListener('input',()=>{stop(mode);display(mode,Number($(mode+'-frame').value));});
  }
  $('laser-map-choice').addEventListener('change',selectMap);
  $('laser-scripted-turn').addEventListener('change',()=>markDraft('laser'));
  selectMap();
  $('laser-map-preview').addEventListener('click',event=>{
    const map=state.laser.map;if(!map)return;
    const [x,y]=previewPoint(event.currentTarget,map,event.clientX,event.clientY);
    if(!freePose(map,[x,y])){
      const error=$('laser-form-error');error.textContent='Choose a free map cell.';error.hidden=false;return;
    }
    setInput('laser-x',x);setInput('laser-y',y);$('laser-form-error').hidden=true;markDraft('laser');
  });
  const redraw=()=>{
    for(const mode of ['lm','laser']){
      drawPfTrajectory($(mode+'-trajectory'),state[mode].result,state[mode].index);
      drawPfError($(mode+'-error-chart'),state[mode].result,state[mode].index);
    }
    drawLaserPreview();
  };
  if('ResizeObserver' in window){
    const observer=new ResizeObserver(redraw);
    for(const mode of ['lm','laser'])for(const suffix of ['trajectory','error-chart'])observer.observe($(mode+'-'+suffix));
    observer.observe($('laser-map-preview'));
  }else window.addEventListener('resize',redraw);
  redraw();
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();

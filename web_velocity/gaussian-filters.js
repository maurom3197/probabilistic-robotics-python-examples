import {runEkfSimulation} from './gaussian-filters-core.js?v=3';
import {drawTrajectory,drawErrorChart} from './gaussian-filters-plots.js?v=3';
import {createLandmarkEditor} from './gaussian-filters-map.js?v=4';

const PARAMETERS=[
  ['v','Linear velocity v (m/s)',0,2,.8,.01,'command'],
  ['omega','Angular velocity ω (rad/s)',-1.5,1.5,.03,.01,'command'],
  ['sigmaV','Velocity σv (m/s)',0,1,.1,.01,'motion-noise'],
  ['sigmaOmega','Angular velocity σω (rad/s)',0,1,.02,.01,'motion-noise'],
  ['fov','Field of view (°)',5,360,60,1,'sensor'],
  ['maxRange','Maximum range (m)',.1,20,8,.1,'sensor'],
  ['sigmaRange','Range σ (m)',.05,1,.1,.01,'measurement-noise'],
  ['sigmaBearing','Bearing σ (°)',1,45,1,.01,'measurement-noise'],
  ['startX','Start x (m)',-20,20,1,.1,'start-pose'],
  ['startY','Start y (m)',-20,20,6,.1,'start-pose'],
  ['startTheta','Start heading θ (°)',-180,180,17.19,.01,'start-pose'],
  ['ekfDt','EKF interval (s)',.1,1,1,.1,'experiment'],
  ['duration','Duration (s)',2,60,22,1,'experiment'],
  ['seed','Random seed',0,999999,42,1,'experiment'],
];
const $=id=>document.getElementById(id);
let result=null,frameIndex=0,runCount=0,timer=null,lastParameters=null,lastLandmarkSignature=null,editor=null;
function stop(){
  if(timer!==null)clearInterval(timer);
  timer=null;$('ekf-play').textContent='▶ Play';
}
function markDraft(){
  const badge=$('ekf-state');
  if(!result){badge.textContent='Waiting for a run';badge.className='badge';return;}
  const dirty=PARAMETERS.some(([key])=>Number($(key).value)!==lastParameters[key])||editor.signature()!==lastLandmarkSignature;
  badge.textContent=dirty?'Parameters changed · press Run':'Run '+runCount+' complete';
  badge.className='badge '+(dirty?'pending':'done');
}
function makeControl([key,label,min,max,initial,step,group]){
  const row=document.createElement('div');row.className='parameter-row';
  const header=document.createElement('div');header.className='parameter-header';
  const text=document.createElement('label');text.htmlFor=key;text.textContent=label;
  const decimals=step===.01?2:step===.1?1:0;
  const format=value=>step===1?String(Math.round(value)):Number(value).toFixed(decimals);
  const number=document.createElement('input');
  Object.assign(number,{type:'number',id:key,name:key,min,max,step,value:format(initial)});
  const range=document.createElement('input');
  Object.assign(range,{type:'range',id:key+'-slider',min,max,step,value:String(initial)});
  range.setAttribute('aria-label',label+' slider');
  number.addEventListener('input',()=>{
    if(Number.isFinite(number.valueAsNumber))range.value=number.value;
    markDraft();
  });
  number.addEventListener('change',()=>{
    const value=number.valueAsNumber;
    if(Number.isFinite(value)&&value>=min&&value<=max){
      const snapped=min+Math.round((value-min)/step)*step;
      number.value=format(Math.max(min,Math.min(max,snapped)));
      range.value=number.value;
    }
    markDraft();
  });
  range.addEventListener('input',()=>{number.value=format(range.value);markDraft();});
  header.append(text,number);row.append(header,range);
  const limits=document.createElement('div');limits.className='range-limits';
  const lo=document.createElement('span'),hi=document.createElement('span');
  lo.textContent=String(min);hi.textContent=String(max);limits.append(lo,hi);row.append(limits);
  $('ekf-'+group+'-controls').append(row);
}
function readParameters(){
  const p={};
  for(const [key,label,min,max,,,] of PARAMETERS){
    const input=$(key),value=input.valueAsNumber;
    if(!Number.isFinite(value)||value<min||value>max||input.validity.stepMismatch){
      input.focus();
      throw new Error(label+': enter a value between '+min+' and '+max+' in the slider increments.');
    }
    p[key]=value;
  }
  p.initialPose=[p.startX,p.startY,p.startTheta*Math.PI/180];
  p.landmarks=editor.read();
  return p;
}
function current(){
  return result?.frames[frameIndex];
}
function display(index){
  if(!result)return;
  frameIndex=Math.max(0,Math.min(index,result.frames.length-1));
  $('ekf-frame').value=String(frameIndex);
  const f=current(),latest=result.frames.slice(0,frameIndex+1).reverse().find(item=>item.update);
  $('ekf-time').textContent='t = '+f.time.toFixed(1)+' s';
  if(f.update){
    $('ekf-event').textContent=f.detected.length
      ? f.detected.length+' landmark'+(f.detected.length===1?'':'s')+' detected. Dashed ellipse: predicted; green ellipse: corrected 95% uncertainty.'
      : 'EKF prediction completed. No landmark was detected at this update.';
  }else if(latest){
    $('ekf-event').textContent='Moving between EKF updates · '+latest.detected.length+' landmark'+(latest.detected.length===1?'':'s')+' detected at the last update.';
  }else $('ekf-event').textContent='Moving toward the first EKF update.';
  $('ekf-position-error').textContent=f.ekfError.toFixed(2);
  $('ekf-odometry-error').textContent=f.odometryError.toFixed(2);
  $('ekf-position-sigma').textContent=Math.sqrt(Math.max(0,f.covariance[0][0])).toFixed(2)+' / '+Math.sqrt(Math.max(0,f.covariance[1][1])).toFixed(2);
  $('ekf-detections').textContent=String(latest?.detected.length??0);
  drawTrajectory($('ekf-trajectory'),result,frameIndex);
  drawErrorChart($('ekf-error-chart'),result,frameIndex);
}
function play(){
  if(!result)return;
  if(timer!==null){stop();return;}
  if(frameIndex>=result.frames.length-1)display(0);
  $('ekf-play').textContent='Ⅱ Pause';
  const period=()=>Math.max(16,100/Number($('ekf-speed').value));
  let last=performance.now(),carry=0;
  timer=setInterval(()=>{
    const now=performance.now();carry+=now-last;last=now;
    const count=Math.floor(carry/period());
    if(!count)return;
    carry-=count*period();
    display(frameIndex+count);
    if(frameIndex>=result.frames.length-1)stop();
  },16);
}
function init(){
  PARAMETERS.forEach(makeControl);
  editor=createLandmarkEditor(markDraft);
  $('ekf-form').addEventListener('submit',event=>{
    event.preventDefault();$('ekf-error').hidden=true;
    try{
      const p=readParameters();
      stop();result=runEkfSimulation(p);lastParameters={...p};lastLandmarkSignature=editor.signature();runCount++;
      $('ekf-state').textContent='Run '+runCount+' complete';$('ekf-state').className='badge done';
      $('ekf-last-parameters').textContent=JSON.stringify(p,null,2);
      $('ekf-frame').max=String(result.frames.length-1);$('ekf-frame').disabled=false;
      $('ekf-play').disabled=false;display(0);play();
    }catch(error){
      $('ekf-error').textContent=error.message;$('ekf-error').hidden=false;
    }
  });
  $('ekf-play').addEventListener('click',play);
  $('ekf-frame').addEventListener('input',()=>{stop();display(Number($('ekf-frame').value));});
  const redraw=()=>{drawTrajectory($('ekf-trajectory'),result,frameIndex);drawErrorChart($('ekf-error-chart'),result,frameIndex);};
  if('ResizeObserver' in window){
    const observer=new ResizeObserver(redraw);
    observer.observe($('ekf-trajectory'));observer.observe($('ekf-error-chart'));
  }else window.addEventListener('resize',redraw);
  redraw();
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();

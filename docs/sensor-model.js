import {DEFAULT_LANDMARKS, loadOccupancyMap, nearestFreeCell, runLandmarkModel, runLikelihoodField} from './sensor-model-core.js';
import {drawLandmarkDetection, drawLandmarkSamples, drawFieldRays, drawFieldHeatmap} from './sensor-plots.js';

const $ = id => document.getElementById(id);
const DEG = Math.PI/180;
const LANDMARK_POSE = [
  ['lm-x', 'Robot x (m)', -15, 15, 3, .1],
  ['lm-y', 'Robot y (m)', -15, 15, 4, .1],
  ['lm-theta', 'Heading θ (°)', -180, 180, -90, 1],
];
const LANDMARK_SENSOR = [
  ['lm-fov', 'Field of view (°)', 5, 360, 60, 1],
  ['lm-range', 'Maximum range (m)', .1, 20, 6, .1],
  ['lm-sigma-range', 'Model σ range (m)', .001, 3, .3, .001],
  ['lm-sigma-bearing', 'Model σ bearing (°)', .1, 90, 18, .1],
  ['lm-seed', 'Random seed', 0, 999999, 42, 1],
];
const FIELD_POSE = [
  ['field-x', 'Map row x (cells)', 0, 15.9, 8, .1],
  ['field-y', 'Map column y (cells)', 0, 15.9, 11, .1],
  ['field-theta', 'Heading θ (°)', -180, 180, 0, 1],
];
const FIELD_SENSOR = [
  ['field-fov', 'Field of view (°)', 1, 360, 45, 1],
  ['field-num-rays', 'Number of rays', 1, 360, 12, 1],
  ['field-max-range', 'Maximum range (cells)', .5, 30, 8, .1],
];
const FIELD_SIGMA = [['field-sigma', 'Hit σ (cells)', .05, 5, .75, .01]];
const FIELD_WEIGHTS = [
  ['weight-hit', 'z_hit', 0, 1, .9, .01],
  ['weight-max', 'z_max', 0, 1, .05, .01],
  ['weight-rand', 'z_rand', 0, 1, .05, .01],
  ['field-seed', 'Random seed', 0, 999999, 42, 1],
];
let landmarkRows = DEFAULT_LANDMARKS.map(p => [...p]);
let landmarkResult = null, fieldResult = null, currentMap = null;
let landmarkRun = 0, fieldRun = 0, mapRequest = 0;
const mapCache = new Map();

function control(parent, [id, label, min, max, value, step], onChange) {
  const row=document.createElement('div'); row.className='parameter-row';
  const header=document.createElement('div'); header.className='parameter-header';
  const text=document.createElement('label'); text.htmlFor=id; text.textContent=label;
  const number=document.createElement('input'); Object.assign(number,{type:'number',id,min,max,step,value:String(value)});
  const range=document.createElement('input'); Object.assign(range,{type:'range',id:`${id}-slider`,min,max,step,value:String(value)});
  range.setAttribute('aria-label',`${label} slider`);
  number.addEventListener('input',()=>{ if(Number.isFinite(number.valueAsNumber)) range.value=number.value; onChange(id); });
  range.addEventListener('input',()=>{ number.value=range.value; onChange(id); });
  header.append(text,number); row.append(header,range);
  const limits=document.createElement('div'); limits.className='range-limits';
  const left=document.createElement('span'),right=document.createElement('span'); left.textContent=String(min); right.textContent=String(max); limits.append(left,right); row.append(limits);
  parent.append(row);
}

function readNumber(id, min, max, integer=false) {
  const input=$(id), n=input.valueAsNumber;
  if (!Number.isFinite(n) || n<min || n>max || (integer && !Number.isInteger(n))) {
    input.focus(); throw new Error(`${input.labels?.[0]?.textContent || id}: enter ${integer?'an integer':'a value'} between ${min} and ${max}.`);
  }
  return n;
}

function markDraft(which) {
  const badge=$(which==='landmark'?'landmark-state':'field-state');
  const result=which==='landmark'?landmarkResult:fieldResult;
  badge.textContent=result?'Parameters changed · press Run':'Waiting for a run';
  badge.className=`badge${result?' pending':''}`;
}

function makeLandmarkRow(index, position) {
  const row=document.createElement('div'); row.className='landmark-edit-row';
  const name=document.createElement('span'); name.textContent=`Landmark ${index+1}`;
  for (const axis of [0,1]) {
    const input=document.createElement('input'); Object.assign(input,{type:'number',min:-20,max:20,step:.1,value:String(position[axis]),id:`landmark-${index}-${axis}`});
    input.setAttribute('aria-label',`Landmark ${index+1} ${axis===0?'x':'y'} position in meters`);
    input.addEventListener('input',()=>{ if(Number.isFinite(input.valueAsNumber)) landmarkRows[index][axis]=input.valueAsNumber; markDraft('landmark'); });
    row.append(input);
  }
  row.prepend(name); return row;
}

function renderLandmarkRows(count) {
  const host=$('landmark-rows'); host.replaceChildren();
  for(let i=0;i<count;i++) {
    if (!landmarkRows[i]) {
      const angle=2*Math.PI*(i-13)/7;
      landmarkRows[i]=[Math.round(60*Math.cos(angle))/10,Math.round(60*Math.sin(angle))/10];
    }
    host.append(makeLandmarkRow(i,landmarkRows[i]));
  }
}

function readLandmark() {
  const pose=LANDMARK_POSE.map(([id,,min,max])=>readNumber(id,min,max));
  const count=readNumber('lm-count',1,20,true);
  const landmarks=[];
  for(let i=0;i<count;i++) landmarks.push([
    readNumber(`landmark-${i}-0`,-20,20), readNumber(`landmark-${i}-1`,-20,20),
  ]);
  return {pose:[pose[0],pose[1],pose[2]*DEG], landmarks,
    fov:readNumber('lm-fov',5,360)*DEG,
    maxRange:readNumber('lm-range',.1,20),
    sigmaRange:readNumber('lm-sigma-range',.001,3),
    sigmaBearing:readNumber('lm-sigma-bearing',.1,90)*DEG,
    seed:readNumber('lm-seed',0,999999,true)};
}

function appendReadings(host, headings, rows) {
  const table=document.createElement('table'); table.className='sensor-table';
  const header=document.createElement('tr'); headings.forEach(h=>{const th=document.createElement('th'); th.textContent=h; header.append(th);});
  table.append(header);
  rows.forEach(values=>{ const tr=document.createElement('tr'); values.forEach(value=>{const td=document.createElement('td'); td.textContent=String(value); tr.append(td);}); table.append(tr); });
  host.replaceChildren(table);
}

function renderLandmark(result) {
  landmarkResult=result; landmarkRun++;
  const badge=$('landmark-state'); badge.textContent=`Run ${landmarkRun} · ${result.detections.length} detected`; badge.className='badge done';
  drawLandmarkDetection($('landmark-detection-plot'),result);
  drawLandmarkSamples($('landmark-samples-plot'),result);
  $('landmark-sample-note').textContent=result.sampledLandmark===null
    ? 'No landmark was detected. Change the robot pose, FOV, or maximum range to sample poses.'
    : `500 sampled poses from landmark 1, as in the Python example, using the selected model noise.`;
  if (!result.detections.length) { $('landmark-readings').textContent='No landmarks detected.'; return; }
  appendReadings($('landmark-readings'),['Landmark','Range [m]','Bearing [°]','p(z|x)'],result.detections.map(d=>[
    d.index+1,d.range.toFixed(3),(d.bearing/DEG).toFixed(2),d.probability.toFixed(4),
  ]));
}

async function getMap(filename) {
  if (!mapCache.has(filename)) mapCache.set(filename,loadOccupancyMap(filename).catch(error=>{mapCache.delete(filename);throw error;}));
  return mapCache.get(filename);
}

async function selectMap() {
  const filename=$('map-choice').value, request=++mapRequest;
  $('field-error').hidden=true;
  try {
    const map=await getMap(filename);
    if(request!==mapRequest) return;
    currentMap=map;
    const x=$('field-x').valueAsNumber,y=$('field-y').valueAsNumber;
    if (!(x>=0&&x<map.rows&&y>=0&&y<map.cols) || map.cells[Math.trunc(x)][Math.trunc(y)]!==0) {
      const free=nearestFreeCell(map,x,y);
      if (free) { $('field-x').value=free[0]; $('field-y').value=free[1]; $('field-x-slider').value=free[0]; $('field-y-slider').value=free[1]; }
    }
    markDraft('field');
  } catch(error) {
    if(request!==mapRequest) return;
    $('field-error').textContent=error.message; $('field-error').hidden=false; currentMap=null;
  }
}

function updateWeights() {
  const values=['weight-hit','weight-max','weight-rand'].map(id=>$(id).valueAsNumber);
  const sum=values.reduce((a,b)=>a+b,0);
  $('weight-summary').textContent=values.every(v=>Number.isFinite(v)&&v>=0)&&sum>0
    ? values.map(v=>(v/sum).toFixed(3)).join(' / ') : 'enter positive weights';
  markDraft('field');
}

function readField() {
  const pose=FIELD_POSE.map(([id,,min,max])=>readNumber(id,min,max));
  const fov=readNumber('field-fov',1,360)*DEG;
  const numRays=readNumber('field-num-rays',1,360,true);
  const maxRange=readNumber('field-max-range',.5,30);
  const sigmaHit=readNumber('field-sigma',.05,5);
  const weights=FIELD_WEIGHTS.slice(0,3).map(([id,,min,max])=>readNumber(id,min,max));
  if (weights.every(w=>w===0)) throw new Error('Enter at least one positive mixture weight.');
  return {mapFile:$('map-choice').value,pose:[pose[0],pose[1],pose[2]*DEG],fov,numRays,maxRange,sigmaHit,weights,
    seed:readNumber('field-seed',0,999999,true)};
}

function renderField(result) {
  fieldResult=result; fieldRun++;
  const badge=$('field-state'); badge.textContent=`Run ${fieldRun} · ${result.map.filename} · ${result.rays.length} rays`; badge.className='badge done';
  drawFieldRays($('field-rays-plot'),result); drawFieldHeatmap($('field-heatmap-plot'),result);
  appendReadings($('field-readings'),['Ray','Range [cells]','Obstacle distance','p(z|x)'],result.rays.map((r,i)=>[
    i+1,r.range.toFixed(3),r.distance.toFixed(3),r.probability.toFixed(4),
  ]));
}

function init() {
  const landmarkDraft=()=>markDraft('landmark');
  LANDMARK_POSE.forEach(def=>control($('landmark-pose-controls'),def,landmarkDraft));
  LANDMARK_SENSOR.forEach(def=>control($('landmark-sensor-controls'),def,landmarkDraft));
  control($('landmark-count-control'),['lm-count','Number of landmarks',1,20,13,1],()=>{
    const n=$('lm-count').valueAsNumber;
    if(Number.isInteger(n)&&n>=1&&n<=20) renderLandmarkRows(n);
    markDraft('landmark');
  });
  renderLandmarkRows(13);
  FIELD_POSE.forEach(def=>control($('field-pose-controls'),def,()=>markDraft('field')));
  FIELD_SENSOR.forEach(def=>control($('field-sensor-controls'),def,()=>markDraft('field')));
  FIELD_SIGMA.forEach(def=>control($('field-sigma-control'),def,()=>markDraft('field')));
  FIELD_WEIGHTS.forEach(def=>control($('field-weight-controls'),def,id=>id.startsWith('weight-')?updateWeights():markDraft('field')));
  $('landmark-form').addEventListener('submit',event=>{
    event.preventDefault(); $('landmark-error').hidden=true;
    try { renderLandmark(runLandmarkModel(readLandmark())); }
    catch(error) { $('landmark-error').textContent=error.message; $('landmark-error').hidden=false; }
  });
  $('field-form').addEventListener('submit',async event=>{
    event.preventDefault(); $('field-error').hidden=true; $('run-field').disabled=true;
    try {
      const p=readField(), map=await getMap(p.mapFile);
      renderField(runLikelihoodField(p,map));
    } catch(error) { $('field-error').textContent=error.message; $('field-error').hidden=false; }
    finally { $('run-field').disabled=false; }
  });
  $('map-choice').addEventListener('change',selectMap);
  selectMap();
  const redraw=()=>{
    drawLandmarkDetection($('landmark-detection-plot'),landmarkResult);
    drawLandmarkSamples($('landmark-samples-plot'),landmarkResult);
    drawFieldRays($('field-rays-plot'),fieldResult);
    drawFieldHeatmap($('field-heatmap-plot'),fieldResult);
  };
  const observer=new ResizeObserver(redraw);
  for(const id of ['landmark-detection-plot','landmark-samples-plot','field-rays-plot','field-heatmap-plot']) observer.observe($(id));
  redraw();
}
init();

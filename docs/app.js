import {createScene, drawPositionDistribution} from './scene.js';

export const PARAMETERS = [
  ['v', 'Linear velocity v (m/s)', 0, 2, .8, 'command', .001],
  ['omega', 'Angular velocity ω (rad/s)', -1.5, 1.5, .6, 'command', .001],
  ['dt', 'Time step Δt (s)', .01, 1, .5, 'command', .001],
  ['a1', 'α₁ · v² → σv', 0, 1, .001, 'alpha', .001],
  ['a2', 'α₂ · ω² → σv', 0, 1, .01, 'alpha', .001],
  ['a3', 'α₃ · v² → σω', 0, 1, .1, 'alpha', .001],
  ['a4', 'α₄ · ω² → σω', 0, 1, .2, 'alpha', .001],
  ['a5', 'α₅ · v² → σγ', 0, 1, .05, 'alpha', .001],
  ['a6', 'α₆ · ω² → σγ', 0, 1, .05, 'alpha', .001],
  ['sigma_v', 'σv (m/s)', 0, 1, .1, 'sigma', .001],
  ['sigma_omega', 'σω (rad/s)', 0, 1, .1, 'sigma', .001],
  ['sigma_gamma', 'σγ (rad/s)', 0, 1, .02, 'sigma', .001],
  ['x', 'Initial position x (m)', -10, 10, 2, 'pose', .001],
  ['y', 'Initial position y (m)', -10, 10, 4, 'pose', .001],
  ['theta', 'Initial orientation θ (rad)', -3.141, 3.141, 0, 'pose', .001],
  ['steps', 'Simulation steps', 1, 60, 12, 'experiment', 1],
  ['samples', 'Number of samples', 50, 5000, 600, 'experiment', 1],
  ['seed', 'Random seed', 0, 999999, 42, 'experiment', 1],
];

const $ = id => document.getElementById(id);
let scene = null, result = null, activeWorker = null, activeReject = null;
let runId = 0, completed = 0, playing = false, animation = null;
let lastParameters = null, provenance = null;

function setStatus(text, error=false) {
  $('execution-status').textContent = text;
  $('execution-status').classList.toggle('error', error);
}

function readParameters(focusInvalid = false) {
  const values = {noise_mode: $('noise-mode').value};
  for (const [key, label, low, high, , group, step] of PARAMETERS) {
    const input = $(key), value = input.valueAsNumber;
    const relevant = group !== 'alpha' && group !== 'sigma' || group === values.noise_mode;
    if (relevant && (!Number.isFinite(value) || value < low || value > high || (step === 1 && !Number.isInteger(value)))) {
      if (focusInvalid) input.focus();
      throw new Error(`${label}: enter ${step === 1 ? 'an integer' : 'a value'} between ${low} and ${high}.`);
    }
    values[key] = Number.isFinite(value) ? value : 0;
  }
  return values;
}

function signature(p) { return JSON.stringify(p); }

function refreshDraftStatus() {
  try {
    const p = readParameters();
    const dirty = lastParameters && signature(p) !== signature(lastParameters);
    $('result-state').textContent = dirty ? 'Parameters changed · press Run' : result ? `Run ${completed} complete` : 'Waiting for a run';
    $('result-state').className = `badge${dirty ? ' pending' : result ? ' done' : ''}`;
    const v2 = p.v*p.v, w2 = p.omega*p.omega;
    const sigmas = p.noise_mode === 'alpha' ? [p.a1*v2+p.a2*w2, p.a3*v2+p.a4*w2, p.a5*v2+p.a6*w2] : [p.sigma_v, p.sigma_omega, p.sigma_gamma];
    ['effective-v', 'effective-omega', 'effective-gamma'].forEach((id, i) => $(id).textContent = sigmas[i].toFixed(3));
  } catch { $('result-state').textContent = 'Check the parameters'; }
}

function updateMode() {
  const alpha = $('noise-mode').value === 'alpha';
  $('alpha-controls').hidden = !alpha; $('sigma-controls').hidden = alpha;
  $('noise-explanation').textContent = alpha
    ? 'Repository parameterization: σv = α₁v² + α₂ω², σω = α₃v² + α₄ω², σγ = α₅v² + α₆ω². These combinations are used as standard deviations.'
    : 'Set the standard deviations of v, ω, and drift γ directly. This is an alternative noise parameterization.';
  refreshDraftStatus();
}

function createControls() {
  for (const [key, label, low, high, initial, group, step] of PARAMETERS) {
    const row = document.createElement('div'); row.className = 'parameter-row';
    const header = document.createElement('div'); header.className = 'parameter-header';
    const text = document.createElement('label'); text.htmlFor = key; text.textContent = label;
    const number = document.createElement('input'); Object.assign(number, {type: 'number', id: key, name: key, min: low, max: high, step, value: step === 1 ? String(initial) : initial.toFixed(3)});
    const range = document.createElement('input'); Object.assign(range, {type: 'range', id: `${key}-slider`, min: low, max: high, step, value: initial}); range.setAttribute('aria-label', `${label} · slider`);
    number.addEventListener('input', () => { if (Number.isFinite(number.valueAsNumber)) range.value = number.value; refreshDraftStatus(); });
    number.addEventListener('change', () => { if (Number.isFinite(number.valueAsNumber) && step !== 1) { number.value = (Math.round(number.valueAsNumber*1000)/1000).toFixed(3); range.value = number.value; } refreshDraftStatus(); });
    range.addEventListener('input', () => { number.value = step === 1 ? range.value : Number(range.value).toFixed(3); refreshDraftStatus(); });
    header.append(text, number); row.append(header, range);
    const limits = document.createElement('div'); limits.className = 'range-limits';
    const left = document.createElement('span'), right = document.createElement('span'); left.textContent = String(low); right.textContent = String(high); limits.append(left, right); row.append(limits);
    $(`${group}-controls`).append(row);
  }
  $('noise-mode').addEventListener('change', updateMode); updateMode();
}

function stopPlayback() { playing = false; cancelAnimationFrame(animation); $('play').textContent = '▶ Play'; }

function displayFrame(index) {
  if (!result) return;
  const frame = result.frames[index]; $('frame').value = index;
  $('frame-label').textContent = `t = ${frame.time.toFixed(3)} s · ${index}/${result.parameters.steps}`;
  scene?.setFrame(index);
}

function play() {
  if (!result) return;
  if (playing) { stopPlayback(); return; }
  let index = Number($('frame').value);
  if (index >= result.frames.length-1) index = 0;
  displayFrame(index); playing = true; $('play').textContent = 'Ⅱ Pause';
  let previous = performance.now();
  function tick(now) {
    if (!playing) return;
    const interval = result.parameters.dt*1000/Number($('playback-speed').value);
    if (now-previous >= interval) {
      index = Math.min(index+Math.floor((now-previous)/interval), result.frames.length-1);
      previous = now; displayFrame(index);
      if (index === result.frames.length-1) { stopPlayback(); return; }
    }
    animation = requestAnimationFrame(tick);
  }
  animation = requestAnimationFrame(tick);
}

function getWorker() {
  if (!activeWorker) {
    activeWorker = new Worker(new URL('./worker.js', import.meta.url), {type: 'module'});
  }
  return activeWorker;
}

function execute(parameters) {
  const id = ++runId, worker = getWorker();
  return new Promise((resolve, reject) => {
    activeReject = reject;
    worker.onmessage = ({data}) => {
      if (data.id !== id) return;
      if (data.type === 'status') setStatus(data.text);
      if (data.type === 'progress') { $('progress').value = data.step/data.total; setStatus(`Python running · step ${data.step}/${data.total}`); }
      if (data.type === 'result') { activeReject = null; resolve(data.result); }
      if (data.type === 'error') { activeReject = null; reject(new Error(data.text)); }
    };
    worker.onerror = event => { activeReject = null; reject(new Error(event.message || 'Failed to load the Web Worker')); };
    worker.postMessage({type: 'run', id, parameters});
  });
}

function renderResult(value, elapsed) {
  result = value; result.provenance = provenance;
  lastParameters = structuredClone(value.parameters); completed += 1;
  document.body.dataset.runCount = String(completed);
  scene?.setResult(result);
  $('scene-empty').hidden = true;
  $('mean-x').textContent = result.mean[0].toFixed(3); $('mean-y').textContent = result.mean[1].toFixed(3);
  $('position-sigma').textContent = `${result.sigma_x.toFixed(3)} / ${result.sigma_y.toFixed(3)}`;
  $('heading-sigma').textContent = result.sigma_theta.toFixed(3);
  $('frame').max = result.frames.length-1; $('frame').disabled = false;
  $('play').disabled = false; $('download').disabled = false;
  displayFrame(result.frames.length-1);
  const distribution = result.position_distribution;
  $('distribution-stats').textContent = `mean = ${distribution.mean.toFixed(3)} m · std_dev = ${distribution.std_dev.toFixed(3)} m`;
  drawPositionDistribution($('position-distribution'), distribution);
  $('last-parameters').textContent = JSON.stringify({parameters: lastParameters, runtime: result.runtime, source: provenance}, null, 2);
  setStatus(`Run ${completed} · ${result.sample_count} samples computed, ${result.shown_count} displayed · ${(elapsed/1000).toFixed(2)} s${completed === 1 ? ' (includes initial loading)' : ''}.`);
  refreshDraftStatus();
}

async function run(event) {
  event.preventDefault(); if ($('run').disabled) return;
  $('form-error').hidden = true;
  let parameters;
  try { parameters = readParameters(true); } catch (error) { $('form-error').textContent = error.message; $('form-error').hidden = false; return; }
  stopPlayback(); $('run').disabled = true; $('cancel').hidden = false;
  $('progress').hidden = false; $('progress').value = 0;
  setStatus('Starting the experiment…');
  const started = performance.now();
  try { renderResult(await execute(parameters), performance.now()-started); }
  catch (error) {
    setStatus(`${error.message}. If the runtime does not load, check your connection and CDN access.`, true);
    activeWorker?.terminate(); activeWorker = null;
  } finally { $('run').disabled = false; $('cancel').hidden = true; $('progress').hidden = true; }
}

function init() {
  document.body.dataset.runCount = '0'; createControls();
  $('parameters').addEventListener('submit', run);
  $('reset-parameters').addEventListener('click', () => {
    for (const [key, , , , initial, , step] of PARAMETERS) { $(key).value = step === 1 ? initial : initial.toFixed(3); $(`${key}-slider`).value = initial; }
    $('noise-mode').value = 'alpha'; $('form-error').hidden = true; updateMode();
  });
  $('cancel').addEventListener('click', () => {
    activeWorker?.terminate(); activeWorker = null;
    const reject = activeReject; activeReject = null;
    reject?.(new Error('Run canceled. Previous results are still available'));
  });
  $('play').addEventListener('click', play);
  $('frame').addEventListener('input', () => { stopPlayback(); displayFrame(Number($('frame').value)); });
  $('show-headings').addEventListener('change', () => scene?.setHeadings($('show-headings').checked));
  $('fit-view').addEventListener('click', () => scene?.fit());
  $('download').addEventListener('click', () => {
    if (!result) return;
    const json = JSON.stringify(result, (_, value) => ArrayBuffer.isView(value) ? Array.from(value) : value);
    const url = URL.createObjectURL(new Blob([json], {type: 'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = `velocity-run-${completed}-seed-${result.parameters.seed}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  const resize = () => drawPositionDistribution($('position-distribution'), result?.position_distribution);
  new ResizeObserver(resize).observe($('position-distribution')); resize();
  createScene($('scene')).then(view => {
    scene = view; $('renderer-status').textContent = `${view.mode} · drag to pan, scroll to zoom`;
    if (result) { scene.setResult(result); displayFrame(Number($('frame').value)); }
  }).catch(error => $('renderer-status').textContent = `View unavailable: ${error.message}`);
  fetch(new URL('./provenance.json', import.meta.url)).then(r => r.ok ? r.json() : null).then(value => {
    provenance = value;
    if (value) $('source-info').textContent = `Source: main · commit ${value.commit?.slice(0, 10) || 'local'} · ${value.source} · SHA ${value.sha256.slice(0, 10)}`;
  }).catch(() => {});
}

init();

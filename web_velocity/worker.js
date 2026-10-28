import { PYODIDE_BASE } from './runtime-config.js';

let runtimePromise = null;
let busy = false;

async function runtime(id) {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      self.postMessage({type: 'status', id, text: 'Caricamento di Python nel browser…'});
      const {loadPyodide} = await import(`${PYODIDE_BASE}pyodide.mjs`);
      const py = await loadPyodide({indexURL: PYODIDE_BASE});
      self.postMessage({type: 'status', id, text: 'Caricamento di NumPy…'});
      await py.loadPackage('numpy');
      py.FS.mkdirTree('/velocity_lab');
      for (const name of ['velocity_core.py', 'experiment.py']) {
        const response = await fetch(new URL(`./python/${name}`, import.meta.url));
        if (!response.ok) throw new Error(`Impossibile caricare ${name}: HTTP ${response.status}`);
        py.FS.writeFile(`/velocity_lab/${name}`, await response.text());
      }
      py.runPython("import sys\nimport numpy as np\nsys.path.insert(0, '/velocity_lab')\nfrom experiment import start_job, advance_job, finish_job");
      return py;
    })().catch(error => { runtimePromise = null; throw error; });
  }
  return runtimePromise;
}

function frameFromJSON(text) {
  const frame = JSON.parse(text);
  frame.cloud = Float32Array.from(frame.cloud);
  return frame;
}

self.onmessage = async ({data}) => {
  if (data.type !== 'run') return;
  const {id, parameters} = data;
  if (busy) { self.postMessage({type: 'error', id, text: 'Una simulazione è già in corso'}); return; }
  busy = true;
  try {
    const py = await runtime(id);
    py.globals.set('parameters_json', JSON.stringify(parameters));
    const frames = [frameFromJSON(py.runPython('start_job(parameters_json)'))];
    for (let step = 1; step <= parameters.steps; step++) {
      frames.push(frameFromJSON(py.runPython('advance_job()')));
      self.postMessage({type: 'progress', id, step, total: parameters.steps});
      // Keep work in bounded chunks. Cancellation terminates this worker.
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    const result = JSON.parse(py.runPython('finish_job()'));
    result.frames = frames;
    result.runtime = {engine: 'Pyodide', python: py.runPython('import sys; sys.version.split()[0]'), numpy: py.runPython('np.__version__')};
    self.postMessage({type: 'result', id, result}, frames.map(frame => frame.cloud.buffer));
  } catch (error) {
    self.postMessage({type: 'error', id, text: error?.message || String(error)});
  } finally {
    busy = false;
  }
};

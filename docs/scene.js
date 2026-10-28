import {THREE_MODULE} from './runtime-config.js';

const COLORS = {blue: '#2454d7', orange: '#ec8b30', purple: '#8156c5', pale: '#c5b3df'};

export function resultBounds(result) {
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const frame of result.frames) {
    const data = frame.cloud;
    for (let i = 0; i < data.length; i += 3) {
      xmin = Math.min(xmin, data[i]); xmax = Math.max(xmax, data[i]);
      ymin = Math.min(ymin, data[i+1]); ymax = Math.max(ymax, data[i+1]);
    }
    xmin = Math.min(xmin, frame.nominal[0]); xmax = Math.max(xmax, frame.nominal[0]);
    ymin = Math.min(ymin, frame.nominal[1]); ymax = Math.max(ymax, frame.nominal[1]);
    for (let i = 0; i < frame.ellipse.length; i += 2) {
      xmin = Math.min(xmin, frame.ellipse[i]); xmax = Math.max(xmax, frame.ellipse[i]);
      ymin = Math.min(ymin, frame.ellipse[i+1]); ymax = Math.max(ymax, frame.ellipse[i+1]);
    }
  }
  return {cx: (xmin+xmax)/2, cy: (ymin+ymax)/2,
    span: Math.max(xmax-xmin, ymax-ymin, 1)*1.25};
}

function niceStep(span) {
  const raw = span/7, power = 10**Math.floor(Math.log10(raw));
  const fraction = raw/power;
  return power*(fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10);
}

export async function createScene(container) {
  let THREE = null;
  try { THREE = await import(THREE_MODULE); } catch (error) { console.warn('Three.js unavailable:', error); }
  return new MotionScene(container, THREE);
}

class MotionScene {
  constructor(container, THREE) {
    this.container = container; this.THREE = THREE;
    this.result = null; this.index = 0; this.headings = false;
    this.view = {cx: 0, cy: 0, span: 5}; this.drag = null;
    this.canvas = document.createElement('canvas');
    this.canvas.setAttribute('aria-hidden', 'true');
    container.prepend(this.canvas);
    this.labels = document.createElement('canvas'); this.labels.className = 'axis-labels';
    this.labels.setAttribute('aria-hidden', 'true'); container.append(this.labels);
    this.context = null;
    if (THREE) {
      try {
        this.renderer = new THREE.WebGLRenderer({canvas: this.canvas, antialias: true, alpha: false});
        this.renderer.setClearColor('#fbfdff');
        this.world = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(-5, 5, 5, -5, .1, 100);
        this.camera.position.set(0, 0, 50);
        this.grid = new THREE.Group(); this.content = new THREE.Group();
        this.world.add(this.grid, this.content);
      } catch (error) { console.warn('WebGL unavailable:', error); this.renderer = null; }
    }
    if (!this.renderer) {
      // A failed WebGL context can prevent a 2D context on the same canvas.
      const old = this.canvas; this.canvas = document.createElement('canvas');
      this.canvas.setAttribute('aria-hidden', 'true'); old.replaceWith(this.canvas);
      this.context = this.canvas.getContext('2d');
    }
    this.mode = this.renderer ? 'Three.js · x–y plane' : 'Canvas 2D · compatibility mode';
    this.canvas.dataset.renderer = this.renderer ? 'three' : 'canvas';
    this.canvas.addEventListener('pointerdown', event => {
      this.drag = {x: event.clientX, y: event.clientY, cx: this.view.cx, cy: this.view.cy};
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener('pointermove', event => {
      if (!this.drag) return;
      const unit = this.view.span/this.height;
      this.view.cx = this.drag.cx-(event.clientX-this.drag.x)*unit;
      this.view.cy = this.drag.cy+(event.clientY-this.drag.y)*unit;
      this.draw();
    });
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
      this.canvas.addEventListener(name, () => { this.drag = null; });
    }
    this.canvas.addEventListener('wheel', event => {
      event.preventDefault();
      this.view.span = Math.max(.05, Math.min(10000, this.view.span*Math.exp(event.deltaY*.001)));
      this.draw();
    }, {passive: false});
    this.canvas.addEventListener('dblclick', () => this.fit());
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container); this.resize();
  }

  resize() {
    this.width = Math.max(1, this.container.clientWidth);
    this.height = Math.max(1, this.container.clientHeight);
    const ratio = Math.min(devicePixelRatio || 1, 2);
    this.labels.width = Math.round(this.width*ratio); this.labels.height = Math.round(this.height*ratio);
    this.labels.getContext('2d').setTransform(ratio, 0, 0, ratio, 0, 0);
    if (this.renderer) { this.renderer.setPixelRatio(ratio); this.renderer.setSize(this.width, this.height, false); }
    else { this.canvas.width = Math.round(this.width*ratio); this.canvas.height = Math.round(this.height*ratio); this.context.setTransform(ratio, 0, 0, ratio, 0, 0); }
    this.draw();
  }

  setResult(result) { this.result = result; this.index = result.frames.length-1; this.fit(); }
  setFrame(index) { this.index = index; this.draw(); }
  setHeadings(value) { this.headings = value; this.draw(); }
  fit() {
    this.view = this.result ? resultBounds(this.result) : {cx: 0, cy: 0, span: 5};
    // span is the vertical extent; allow narrow screens to fit x as well.
    if (this.width < this.height) this.view.span *= this.height/this.width;
    this.draw();
  }

  screen(x, y) {
    const unit = this.height/this.view.span;
    return [(x-this.view.cx)*unit+this.width/2, this.height/2-(y-this.view.cy)*unit];
  }

  clearGroup(group) {
    for (const object of [...group.children]) {
      object.geometry?.dispose();
      if (Array.isArray(object.material)) object.material.forEach(m => m.dispose());
      else object.material?.dispose();
      group.remove(object);
    }
  }

  line(points, color, opacity=1, segments=false) {
    if (this.renderer) {
      const T = this.THREE;
      const geometry = new T.BufferGeometry().setFromPoints(points.map(([x, y]) => new T.Vector3(x, y, .02)));
      const material = new T.LineBasicMaterial({color, transparent: opacity < 1, opacity});
      this.content.add(segments ? new T.LineSegments(geometry, material) : new T.Line(geometry, material));
    } else {
      const c = this.context; c.strokeStyle = color; c.globalAlpha = opacity; c.lineWidth = 1.5;
      c.beginPath();
      points.forEach(([x, y], i) => { const p = this.screen(x, y); if (i === 0 || (segments && i%2 === 0)) c.moveTo(...p); else c.lineTo(...p); });
      c.stroke(); c.globalAlpha = 1;
    }
  }

  drawGrid() {
    const {cx, cy, span} = this.view, xspan = span*this.width/this.height;
    const step = niceStep(Math.max(span, xspan));
    const xmin = cx-xspan/2, xmax = cx+xspan/2, ymin = cy-span/2, ymax = cy+span/2;
    const labels = this.labels.getContext('2d'); labels.clearRect(0, 0, this.width, this.height);
    labels.font = '10px system-ui'; labels.fillStyle = '#8290a4';
    const points = [];
    const format = v => Math.abs(v) < 1e-8 ? '0' : Number(v.toPrecision(4)).toString();
    for (let x = Math.ceil(xmin/step)*step; x <= xmax; x += step) {
      points.push([x, ymin], [x, ymax]); const [sx] = this.screen(x, ymin);
      labels.fillText(format(x), sx+3, this.height-8);
    }
    for (let y = Math.ceil(ymin/step)*step; y <= ymax; y += step) {
      points.push([xmin, y], [xmax, y]); const [, sy] = this.screen(xmin, y);
      if (sy > 12 && sy < this.height-20) labels.fillText(format(y), 8, sy-4);
    }
    labels.fillStyle = '#52657e'; labels.fillText('y (m)', 8, 17); labels.fillText('x (m)', this.width-42, this.height-8);
    this.line(points, '#e5ecf6', 1, true);
  }

  draw() {
    if (!this.width) return;
    if (this.renderer) {
      this.clearGroup(this.content);
      const h = this.view.span/2, w = h*this.width/this.height;
      Object.assign(this.camera, {left: -w, right: w, top: h, bottom: -h});
      this.camera.position.set(this.view.cx, this.view.cy, 50); this.camera.updateProjectionMatrix();
    } else { this.context.clearRect(0, 0, this.width, this.height); this.context.fillStyle = '#fbfdff'; this.context.fillRect(0, 0, this.width, this.height); }
    this.drawGrid();
    if (this.result) {
      const frame = this.result.frames[this.index], data = frame.cloud;
      // Show at most 40 sampled trajectories; all displayed endpoints remain.
      const stride = Math.max(1, Math.ceil(data.length/3/40));
      for (let particle = 0; particle < data.length/3; particle += stride) {
        this.line(this.result.frames.slice(0, this.index+1).map(f => [f.cloud[3*particle], f.cloud[3*particle+1]]), COLORS.blue, .07);
      }
      this.line(this.result.frames.slice(0, this.index+1).map(f => f.nominal), COLORS.orange);
      this.line(this.result.frames.slice(0, this.index+1).map(f => f.mean), COLORS.purple);
      const ellipse = []; for (let i = 0; i < frame.ellipse.length; i += 2) ellipse.push([frame.ellipse[i], frame.ellipse[i+1]]);
      this.line(ellipse, COLORS.pale, .8);
      if (this.renderer) {
        const T = this.THREE, positions = new Float32Array(data.length);
        for (let i = 0; i < data.length; i += 3) { positions[i] = data[i]; positions[i+1] = data[i+1]; positions[i+2] = .04; }
        const geometry = new T.BufferGeometry(); geometry.setAttribute('position', new T.BufferAttribute(positions, 3));
        this.content.add(new T.Points(geometry, new T.PointsMaterial({color: COLORS.blue, size: 4, sizeAttenuation: false, transparent: true, opacity: .48})));
      } else {
        const c = this.context; c.fillStyle = COLORS.blue; c.globalAlpha = .4;
        for (let i = 0; i < data.length; i += 3) { const [x, y] = this.screen(data[i], data[i+1]); c.beginPath(); c.arc(x, y, 2.2, 0, Math.PI*2); c.fill(); }
        c.globalAlpha = 1;
      }
      if (this.headings) {
        const segments = [], length = this.view.span*.023;
        for (let i = 0; i < data.length; i += 3*stride) {
          segments.push([data[i], data[i+1]], [data[i]+length*Math.cos(data[i+2]), data[i+1]+length*Math.sin(data[i+2])]);
        }
        this.line(segments, COLORS.blue, .65, true);
      }
      const length = this.view.span*.04, pose = frame.nominal;
      this.line([[pose[0], pose[1]], [pose[0]+length*Math.cos(pose[2]), pose[1]+length*Math.sin(pose[2])]], COLORS.orange);
    }
    if (this.renderer) this.renderer.render(this.world, this.camera);
  }
}

export function drawPositionDistribution(canvas, distribution) {
  if (!canvas) return;
  const width = Math.max(canvas.clientWidth, 150), height = 180, ratio = Math.min(devicePixelRatio || 1, 2);
  canvas.width = width*ratio; canvas.height = height*ratio;
  const c = canvas.getContext('2d'); c.setTransform(ratio, 0, 0, ratio, 0, 0);
  c.clearRect(0, 0, width, height);
  c.font = '10px system-ui'; c.fillStyle = '#8290a4';
  if (!distribution) { c.fillText('The distribution will appear after a run.', 8, 55); return; }
  const left = 42, bottom = height-37, right = width-12, top = 12;
  const {counts, edges, mean, std_dev} = distribution;
  const halfSpan = std_dev > 0 ? 5*std_dev : .5;
  const xmin = mean-halfSpan, xmax = mean+halfSpan;
  const xToPixel = x => left+(x-xmin)/(xmax-xmin)*(right-left);
  const maxCount = Math.max(...counts, 1);
  c.strokeStyle = '#e5ecf6';
  for (let i = 0; i <= 4; i++) {
    const y = bottom-i/4*(bottom-top); c.beginPath(); c.moveTo(left, y); c.lineTo(right, y); c.stroke();
    c.fillText(String(Math.round(i/4*maxCount)), 2, y+3);
  }
  c.textAlign = 'center';
  for (let i = 0; i <= 2; i++) {
    const x = left+i/2*(right-left); c.beginPath(); c.moveTo(x, top); c.lineTo(x, bottom); c.stroke();
    c.fillText((xmin+i/2*(xmax-xmin)).toFixed(2), x, bottom+13);
  }
  c.save(); c.beginPath(); c.rect(left, top, right-left, bottom-top); c.clip();
  c.fillStyle = '#6f98eb';
  counts.forEach((count, i) => {
    const x0 = xToPixel(edges[i]), x1 = xToPixel(edges[i+1]);
    const barHeight = count/maxCount*(bottom-top);
    c.fillRect(x0+.25, bottom-barHeight, Math.max(x1-x0-.5, .5), barHeight);
  });
  c.restore(); c.fillStyle = '#64748b';
  c.fillText('x-position [m]', (left+right)/2, height-3);
}

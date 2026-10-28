const COLORS = {blue: '#2454d7', orange: '#d64b39', purple: '#8156c5', grid: '#e5ecf6'};
const VIRIDIS = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'];

function colorAt(t) {
  const x = Math.max(0, Math.min(.999999, t))*(VIRIDIS.length-1), i = Math.floor(x), u = x-i;
  const a = VIRIDIS[i].slice(1).match(/../g).map(v => parseInt(v, 16));
  const b = VIRIDIS[i+1].slice(1).match(/../g).map(v => parseInt(v, 16));
  return `rgb(${a.map((v, n) => Math.round(v+(b[n]-v)*u)).join(',')})`;
}

function canvasContext(canvas) {
  const width = Math.max(canvas.clientWidth, 200), height = Math.max(canvas.clientHeight, 260);
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width*ratio); canvas.height = Math.round(height*ratio);
  const c = canvas.getContext('2d'); c.setTransform(ratio, 0, 0, ratio, 0, 0);
  c.clearRect(0, 0, width, height);
  return {c, width, height};
}

function empty(c, width, height, message) {
  c.fillStyle = '#fbfdff'; c.fillRect(0, 0, width, height);
  c.fillStyle = '#64748b'; c.font = '12px system-ui'; c.textAlign = 'center';
  c.fillText(message, width/2, height/2);
}

function niceStep(span) {
  const raw = span/6, base = 10**Math.floor(Math.log10(raw));
  return base*(raw/base <= 1 ? 1 : raw/base <= 2 ? 2 : raw/base <= 5 ? 5 : 10);
}

function worldView(c, width, height, points, rightPadding=20) {
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  let xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const spanX = Math.max(xmax-xmin, 1), spanY = Math.max(ymax-ymin, 1);
  xmin -= .12*spanX; xmax += .12*spanX; ymin -= .12*spanY; ymax += .12*spanY;
  const plot = {left: 42, top: 16, right: width-rightPadding, bottom: height-35};
  const scale = Math.min((plot.right-plot.left)/(xmax-xmin), (plot.bottom-plot.top)/(ymax-ymin));
  const cx=(xmin+xmax)/2, cy=(ymin+ymax)/2;
  const xy = (x, y) => [(plot.left+plot.right)/2+(x-cx)*scale, (plot.top+plot.bottom)/2-(y-cy)*scale];
  c.fillStyle = '#fbfdff'; c.fillRect(0, 0, width, height);
  c.save(); c.beginPath(); c.rect(plot.left, plot.top, plot.right-plot.left, plot.bottom-plot.top); c.clip();
  c.strokeStyle = COLORS.grid; c.lineWidth = 1;
  const step = niceStep(Math.max(xmax-xmin, ymax-ymin));
  for (let x=Math.ceil(xmin/step)*step; x<=xmax; x+=step) {
    const [px]=xy(x,0); c.beginPath(); c.moveTo(px,plot.top); c.lineTo(px,plot.bottom); c.stroke();
  }
  for (let y=Math.ceil(ymin/step)*step; y<=ymax; y+=step) {
    const [,py]=xy(0,y); c.beginPath(); c.moveTo(plot.left,py); c.lineTo(plot.right,py); c.stroke();
  }
  c.restore();
  c.fillStyle = '#64748b'; c.font = '10px system-ui'; c.textAlign = 'center';
  for (let x=Math.ceil(xmin/step)*step; x<=xmax; x+=step) {
    const [px]=xy(x,0); if (px>=plot.left+12 && px<=plot.right-12) c.fillText(Number(x.toPrecision(3)),px,height-19);
  }
  c.textAlign='right';
  for (let y=Math.ceil(ymin/step)*step; y<=ymax; y+=step) {
    const [,py]=xy(0,y); if (py>=plot.top+8 && py<=plot.bottom-8) c.fillText(Number(y.toPrecision(3)),plot.left-6,py+3);
  }
  c.textAlign='center'; c.fillText('x-position [m]',(plot.left+plot.right)/2,height-3);
  c.save(); c.translate(11,(plot.top+plot.bottom)/2); c.rotate(-Math.PI/2); c.fillText('y-position [m]',0,0); c.restore();
  return {xy, plot};
}

function poseArrow(c, x, y, theta, size, color, fill=false) {
  c.save(); c.translate(x,y); c.rotate(-theta); c.strokeStyle=color; c.fillStyle=color;
  c.lineWidth=fill?2.5:1.2; c.beginPath(); c.moveTo(-size*.55,0); c.lineTo(size*.65,0); c.moveTo(size*.65,0);
  c.lineTo(size*.2,-size*.35); c.moveTo(size*.65,0); c.lineTo(size*.2,size*.35); c.stroke();
  if (fill) { c.beginPath(); c.arc(0,0,3,0,Math.PI*2); c.fill(); }
  c.restore();
}

function colorScale(c, x, top, height, label, low, high) {
  for (let i=0; i<height; i++) { c.fillStyle=colorAt(1-i/height); c.fillRect(x,top+i,10,1); }
  c.fillStyle='#64748b'; c.font='10px system-ui'; c.textAlign='left';
  c.fillText(high.toFixed(2),x+14,top+8); c.fillText(low.toFixed(2),x+14,top+height);
  c.save(); c.translate(x+35,top+height/2); c.rotate(-Math.PI/2); c.textAlign='center'; c.fillText(label,0,0); c.restore();
}

export function drawLandmarkDetection(canvas, result) {
  const {c,width,height}=canvasContext(canvas);
  if (!result) return empty(c,width,height,'Run the landmark model to see detections.');
  const p=result.parameters, [x,y,theta]=p.pose;
  const points=[...p.landmarks,[x,y]];
  for (let i=0; i<=24; i++) {
    const angle=theta-p.fov/2+i*p.fov/24;
    points.push([x+p.maxRange*Math.cos(angle),y+p.maxRange*Math.sin(angle)]);
  }
  const {xy}=worldView(c,width,height,points,62);
  c.save(); c.strokeStyle=COLORS.blue; c.fillStyle='#2454d70d'; c.lineWidth=1.3;
  c.beginPath(); c.moveTo(...xy(x,y));
  for (let i=0; i<=48; i++) { const angle=theta-p.fov/2+i*p.fov/48; c.lineTo(...xy(x+p.maxRange*Math.cos(angle),y+p.maxRange*Math.sin(angle))); }
  c.closePath(); c.fill(); c.stroke();
  p.landmarks.forEach(([mx,my],i)=>{ const [px,py]=xy(mx,my); c.fillStyle='#17243d'; c.fillRect(px-4,py-4,8,8); c.fillStyle='#64748b'; c.font='10px system-ui'; c.fillText(String(i+1),px+7,py-6); });
  const probabilities=result.detections.map(d=>d.probability), low=Math.min(...probabilities,0), high=Math.max(...probabilities,1e-12);
  result.detections.forEach(d=>{ const [px,py]=xy(...d.endpoint); c.fillStyle=colorAt(high>low?(d.probability-low)/(high-low):.5); c.strokeStyle='#17243d'; c.beginPath(); c.arc(px,py,5,0,Math.PI*2); c.fill(); c.stroke(); });
  poseArrow(c,...xy(x,y),theta,22,COLORS.orange,true);
  if (result.detections.length) colorScale(c,width-48,32,Math.min(height-80,210),'p(z|x)',low,high);
  c.restore();
}

export function drawLandmarkSamples(canvas, result) {
  const {c,width,height}=canvasContext(canvas);
  if (!result) return empty(c,width,height,'Run the landmark model to sample poses.');
  if (!result.samples.length) return empty(c,width,height,'No landmark detected. Change the pose, FOV, or range.');
  const p=result.parameters, landmark=p.landmarks[result.sampledLandmark];
  const points=[...result.samples.map(s=>s.slice(0,2)),p.pose.slice(0,2),landmark];
  const {xy}=worldView(c,width,height,points);
  for (const [x,y,theta] of result.samples) poseArrow(c,...xy(x,y),theta,8,COLORS.blue);
  const [lx,ly]=xy(...landmark); c.fillStyle='#17243d'; c.fillRect(lx-5,ly-5,10,10);
  poseArrow(c,...xy(p.pose[0],p.pose[1]),p.pose[2],23,COLORS.orange,true);
}

function mapView(c,width,height,map,withScale=false) {
  const pad={left:40,top:15,right:withScale?75:20,bottom:38};
  const cell=Math.min((width-pad.left-pad.right)/map.cols,(height-pad.top-pad.bottom)/map.rows);
  const left=pad.left+(width-pad.left-pad.right-map.cols*cell)/2;
  const top=pad.top+(height-pad.top-pad.bottom-map.rows*cell)/2;
  const xy=(row,col)=>[left+col*cell,top+row*cell];
  c.fillStyle='#fbfdff'; c.fillRect(0,0,width,height);
  c.fillStyle='#64748b'; c.font='10px system-ui'; c.textAlign='center';
  for (let col=0; col<=map.cols; col+=2) c.fillText(col,left+col*cell,top+map.rows*cell+15);
  c.textAlign='right';
  for (let row=0; row<=map.rows; row+=2) c.fillText(row,left-6,top+row*cell+3);
  c.textAlign='center'; c.fillText('Y [cells]',left+map.cols*cell/2,height-3);
  c.save(); c.translate(11,top+map.rows*cell/2); c.rotate(-Math.PI/2); c.fillText('X [cells]',0,0); c.restore();
  return {xy,cell,left,top};
}

function drawCells(c,map,view,field=null) {
  const {xy,cell}=view;
  for (let row=0; row<map.rows; row++) for (let col=0; col<map.cols; col++) {
    const state=map.cells[row][col], [x,y]=xy(row,col);
    if (field) {
      if (state===-1) c.fillStyle='#9aa2ae';
      else { const value=Math.round(255*field[row][col]); c.fillStyle=`rgb(${value},${value},${value})`; }
    } else c.fillStyle=state===1?'#263247':state===-1?'#9aa2ae':'#fff';
    c.fillRect(x,y,cell,cell);
    c.strokeStyle=field?'#ffffff20':'#dfe6f0'; c.lineWidth=.6; c.strokeRect(x,y,cell,cell);
  }
}

function mapPose(c,view,pose) {
  const [row,col,theta]=pose, [x,y]=view.xy(row,col);
  c.save(); c.translate(x,y); c.rotate(Math.PI/2-theta); c.strokeStyle=COLORS.orange; c.fillStyle=COLORS.orange;
  c.lineWidth=2.5; c.beginPath(); c.arc(0,0,5,0,Math.PI*2); c.fill();
  c.beginPath(); c.moveTo(0,0); c.lineTo(16,0); c.moveTo(16,0); c.lineTo(10,-5); c.moveTo(16,0); c.lineTo(10,5); c.stroke(); c.restore();
}

export function drawFieldRays(canvas,result) {
  const {c,width,height}=canvasContext(canvas);
  if (!result) return empty(c,width,height,'Run the likelihood field to see laser rays.');
  const view=mapView(c,width,height,result.map,true); drawCells(c,result.map,view);
  const probabilities=result.rays.map(r=>r.probability), low=Math.min(...probabilities), high=Math.max(...probabilities);
  const start=view.xy(...result.parameters.pose.slice(0,2));
  result.rays.forEach(ray=>{
    const end=view.xy(...ray.endpoint); c.strokeStyle='#2454d7a0'; c.lineWidth=1.3;
    c.beginPath(); c.moveTo(...start); c.lineTo(...end); c.stroke();
    c.fillStyle=colorAt(high>low?(ray.probability-low)/(high-low):.5);
    c.strokeStyle='#17243d'; c.beginPath(); c.arc(...end,4.5,0,Math.PI*2); c.fill(); c.stroke();
  });
  mapPose(c,view,result.parameters.pose);
  colorScale(c,width-58,32,Math.min(height-80,210),'p(z|x)',low,high);
}

export function drawFieldHeatmap(canvas,result) {
  const {c,width,height}=canvasContext(canvas);
  if (!result) return empty(c,width,height,'Run the likelihood field to see the map.');
  const view=mapView(c,width,height,result.map,true); drawCells(c,result.map,view,result.field);
  mapPose(c,view,result.parameters.pose);
  c.fillStyle='#64748b'; c.font='10px system-ui'; c.textAlign='left';
  c.fillText('high',width-60,48);
  const gradient=c.createLinearGradient(0,122,0,56); gradient.addColorStop(0,'#000'); gradient.addColorStop(1,'#fff');
  c.fillStyle=gradient; c.fillRect(width-60,57,6,63);
  c.fillStyle='#64748b'; c.fillText('low',width-60,134);
}

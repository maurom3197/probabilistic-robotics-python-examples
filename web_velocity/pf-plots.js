const C={truth:'#ec8b30',pf:'#8156c5',green:'#1a9a75',muted:'#64748b',grid:'#e5ecf6'};
const VIRIDIS=['#440154','#3b528b','#21918c','#5ec962','#fde725'];
function probabilityColor(t){
  const x=Math.max(0,Math.min(.999999,t))*(VIRIDIS.length-1),i=Math.floor(x),u=x-i;
  const a=VIRIDIS[i].slice(1).match(/../g).map(v=>parseInt(v,16));
  const b=VIRIDIS[i+1].slice(1).match(/../g).map(v=>parseInt(v,16));
  return `rgb(${a.map((value,n)=>Math.round(value+(b[n]-value)*u)).join(',')})`;
}
function probabilityScale(c,width,height,low,high){
  const x=width-58,top=32,scaleHeight=Math.min(height-80,210);
  for(let i=0;i<scaleHeight;i++){c.fillStyle=probabilityColor(1-i/scaleHeight);c.fillRect(x,top+i,10,1);}
  c.fillStyle=C.muted;c.font='10px system-ui';c.textAlign='left';
  c.fillText(high.toFixed(2),x+14,top+8);c.fillText(low.toFixed(2),x+14,top+scaleHeight);
  c.save();c.translate(x+35,top+scaleHeight/2);c.rotate(-Math.PI/2);c.textAlign='center';c.fillText('p(z|x)',0,0);c.restore();
}
function surface(canvas){
  const rect=canvas.getBoundingClientRect(),width=Math.max(1,rect.width),height=Math.max(1,rect.height);
  const ratio=Math.min(window.devicePixelRatio||1,2),px=Math.round(width*ratio),py=Math.round(height*ratio);
  if(canvas.width!==px||canvas.height!==py){canvas.width=px;canvas.height=py;}
  const c=canvas.getContext('2d');c.setTransform(ratio,0,0,ratio,0,0);
  c.clearRect(0,0,width,height);c.fillStyle='#fbfdff';c.fillRect(0,0,width,height);
  c.font='11px system-ui';c.lineCap='round';c.lineJoin='round';
  return {c,width,height};
}
function path(c,points,color,width=2,dash=[]){
  if(points.length<2)return;
  c.beginPath();c.moveTo(...points[0]);for(let i=1;i<points.length;i++)c.lineTo(...points[i]);
  c.strokeStyle=color;c.lineWidth=width;c.setLineDash(dash);c.stroke();c.setLineDash([]);
}
function dot(c,x,y,r,color){c.beginPath();c.arc(x,y,r,0,2*Math.PI);c.fillStyle=color;c.fill();}
function mapGeometry(width,height,map,compact=false,rightPadding=20){
  const left=compact?30:46,right=compact?13:rightPadding,top=compact?12:19,bottom=compact?28:39;
  const scale=Math.min((width-left-right)/map.rows,(height-top-bottom)/map.cols);
  const extraX=(width-left-right-map.rows*scale)/2,extraY=(height-top-bottom-map.cols*scale)/2;
  const x0=left+extraX,y0=height-bottom-extraY;
  return {scale,left:x0,top:y0-map.cols*scale,width:map.rows*scale,height:map.cols*scale,
    toX:x=>x0+x*scale,toY:y=>y0-y*scale,
    toWorld:(x,y)=>[(x-x0)/scale,(y0-y)/scale]};
}
function mapBase(c,map,g,compact=false){
  for(let row=0;row<map.rows;row++)for(let col=0;col<map.cols;col++){
    const state=map.cells[row][col];
    c.fillStyle=state===1?'#2d394d':state===-1?'#a4adb8':'#fff';
    c.fillRect(g.toX(row),g.toY(col+1),g.scale+.3,g.scale+.3);
  }
  c.strokeStyle='#dbe4f0';c.lineWidth=.7;
  for(let row=0;row<=map.rows;row+=2)path(c,[[g.toX(row),g.top],[g.toX(row),g.top+g.height]],'#dbe4f0',.7);
  for(let col=0;col<=map.cols;col+=2)path(c,[[g.left,g.toY(col)],[g.left+g.width,g.toY(col)]],'#dbe4f0',.7);
  c.fillStyle=C.muted;c.textAlign='center';
  for(let row=0;row<=map.rows;row+=compact?5:4)c.fillText(String(row),g.toX(row),g.top+g.height+15);
  c.textAlign='right';
  for(let col=0;col<=map.cols;col+=compact?5:4)c.fillText(String(col),g.left-7,g.toY(col)+3);
  if(!compact){
    c.textAlign='center';c.fillText('map row x [cells]',g.left+g.width/2,g.top+g.height+31);
    c.save();c.translate(12,g.top+g.height/2);c.rotate(-Math.PI/2);c.fillText('map column y [cells]',0,0);c.restore();
  }
}
function landmarkBase(c,g,world){
  c.fillStyle='#fff';c.fillRect(g.left,g.top,g.width,g.height);
  for(let x=0;x<=world.rows;x+=5){
    path(c,[[g.toX(x),g.top],[g.toX(x),g.top+g.height]],C.grid,1);
    c.fillStyle=C.muted;c.textAlign='center';c.fillText(String(x),g.toX(x),g.top+g.height+15);
  }
  for(let y=0;y<=world.cols;y+=5){
    path(c,[[g.left,g.toY(y)],[g.left+g.width,g.toY(y)]],C.grid,1);
    c.fillStyle=C.muted;c.textAlign='right';c.fillText(String(y),g.left-7,g.toY(y)+3);
  }
  c.fillStyle=C.muted;c.textAlign='center';c.fillText('x [m]',g.left+g.width/2,g.top+g.height+31);
  c.save();c.translate(12,g.top+g.height/2);c.rotate(-Math.PI/2);c.fillText('y [m]',0,0);c.restore();
}
function robot(c,pose,g,color){
  const x=g.toX(pose[0]),y=g.toY(pose[1]);
  dot(c,x,y,4.8,color);
  path(c,[[x,y],[x+12*Math.cos(pose[2]),y-12*Math.sin(pose[2])]],color,2);
}
export function drawPfTrajectory(canvas,result,index){
  const {c,width,height}=surface(canvas);
  if(!result){c.fillStyle=C.muted;c.textAlign='center';c.fillText('Run the particle filter to see trajectories.',width/2,height/2);return;}
  const domain=result.map??result.world;
  const g=mapGeometry(width,height,domain,false,result.mode==='laser'?85:20),f=result.frames[Math.max(0,Math.min(index,result.frames.length-1))];
  if(result.mode==='laser')mapBase(c,result.map,g);else landmarkBase(c,g,result.world);
  c.save();c.beginPath();c.rect(g.left,g.top,g.width,g.height);c.clip();
  const latestCloud=result.frames.slice(0,index+1).reverse().find(frame=>frame.cloud)?.cloud||[];
  for(const point of latestCloud.slice(0,500))dot(c,g.toX(point[0]),g.toY(point[1]),1.45,'#dc26268c');
  if(result.mode==='landmark'){
    const detected=new Set(f.detected);
    result.landmarks.forEach((point,i)=>{
      const x=g.toX(point[0]),y=g.toY(point[1]);
      c.fillStyle=detected.has(i)?C.green:'#566980';c.fillRect(x-4,y-4,8,8);
      if(detected.has(i))path(c,[[g.toX(f.truth[0]),g.toY(f.truth[1])],[x,y]],'#1a9a7588',1);
    });
  }else{
    const observed=result.frames.slice(0,index+1).reverse().find(frame=>frame.rays);
    if(observed&&f.time-observed.time<=result.parameters.pfDt+.1){
      const pose=observed.truth,probabilities=observed.rays.map(ray=>ray.probability).filter(Number.isFinite);
      const low=probabilities.length?Math.min(...probabilities):0,high=probabilities.length?Math.max(...probabilities):1;
      for(const ray of observed.rays){
        path(c,[[g.toX(pose[0]),g.toY(pose[1])],[g.toX(ray.endpoint[0]),g.toY(ray.endpoint[1])]],'#2454d755',.8);
        const color=ray.probability===null?'#94a3b8':probabilityColor(high>low?(ray.probability-low)/(high-low):.5);
        dot(c,g.toX(ray.endpoint[0]),g.toY(ray.endpoint[1]),3.2,color);
      }
    }
  }
  const frames=result.frames.slice(0,index+1);
  path(c,frames.map(item=>[g.toX(item.truth[0]),g.toY(item.truth[1])]),C.truth,2.3);
  path(c,frames.filter((item,i)=>i===0||item.update).map(item=>[g.toX(item.estimate[0]),g.toY(item.estimate[1])]),C.pf,2.3);
  robot(c,f.truth,g,C.truth);robot(c,f.estimate,g,C.pf);
  c.restore();
  if(result.mode==='laser'){
    const observed=result.frames.slice(0,index+1).reverse().find(frame=>frame.rays);
    if(observed&&f.time-observed.time<=result.parameters.pfDt+.1){
      const probabilities=observed.rays.map(ray=>ray.probability).filter(Number.isFinite);
      if(probabilities.length)probabilityScale(c,width,height,Math.min(...probabilities),Math.max(...probabilities));
    }
  }
}
export function drawPfError(canvas,result,index){
  const {c,width,height}=surface(canvas);
  if(!result){c.fillStyle=C.muted;c.textAlign='center';c.fillText('Run the particle filter to see position error.',width/2,height/2);return;}
  const left=55,right=18,top=18,bottom=38,plotW=width-left-right,plotH=height-top-bottom;
  const duration=result.frames.at(-1).time,maxError=Math.max(.2,...result.pfErrors.map(p=>p.error),...result.simGtErrors.map(p=>p.error))*1.12;
  const x=t=>left+t/duration*plotW,y=e=>height-bottom-e/maxError*plotH;
  for(let i=0;i<=4;i++){
    const e=i*maxError/4,py=y(e);path(c,[[left,py],[width-right,py]],C.grid,1);
    c.fillStyle=C.muted;c.textAlign='right';c.fillText(e.toFixed(1),left-7,py+3);
    const t=i*duration/4,px=x(t);path(c,[[px,top],[px,height-bottom]],C.grid,1);
    c.textAlign='center';c.fillText(t.toFixed(0),px,height-bottom+17);
  }
  c.textAlign='center';c.fillText('time [s]',width/2,height-5);
  c.save();c.translate(12,height/2);c.rotate(-Math.PI/2);c.fillText(result.mode==='landmark'?'position error [m]':'position error [cells]',0,0);c.restore();
  path(c,result.simGtErrors.map(p=>[x(p.time),y(p.error)]),C.truth,1.8);
  path(c,result.pfErrors.map(p=>[x(p.time),y(p.error)]),C.pf,2.3);
  const current=result.frames[index],px=x(current.time);
  path(c,[[px,top],[px,height-bottom]],'#2454d788',1,[3,3]);
  dot(c,px,y(current.simGtError),4,C.truth);
  if(current.update)dot(c,px,y(current.pfError),4,C.pf);
}
export function drawPfMapPreview(canvas,map,pose){
  const {c,width,height}=surface(canvas);
  if(!map){c.fillStyle=C.muted;c.textAlign='center';c.fillText('Select a map to preview the robot pose.',width/2,height/2);return;}
  const g=mapGeometry(width,height,map,true);mapBase(c,map,g,true);
  if(pose.every(Number.isFinite))robot(c,[pose[0],pose[1],pose[2]*Math.PI/180],g,C.truth);
}
export function previewPoint(canvas,map,clientX,clientY){
  const rect=canvas.getBoundingClientRect(),g=mapGeometry(rect.width,rect.height,map,true);
  const [x,y]=g.toWorld(clientX-rect.left,clientY-rect.top);
  return [Math.max(0,Math.min(map.rows-.1,Math.round(x*10)/10)),Math.max(0,Math.min(map.cols-.1,Math.round(y*10)/10))];
}

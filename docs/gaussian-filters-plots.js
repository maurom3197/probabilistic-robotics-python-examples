const COLORS={truth:'#2454d7',odometry:'#ec8b30',estimate:'#8156c5',detected:'#1a9a75',grid:'#e9eef5',ink:'#17243d',muted:'#64748b'};
function canvasContext(canvas){
  const rect=canvas.getBoundingClientRect(),width=Math.max(1,rect.width),height=Math.max(1,rect.height);
  const dpr=Math.min(window.devicePixelRatio||1,2),px=Math.round(width*dpr),py=Math.round(height*dpr);
  if(canvas.width!==px||canvas.height!==py){canvas.width=px;canvas.height=py;}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,width,height);ctx.fillStyle='#fbfdff';ctx.fillRect(0,0,width,height);
  ctx.font='11px Inter, system-ui, sans-serif';ctx.lineCap='round';ctx.lineJoin='round';
  return {ctx,width,height};
}
function line(ctx,points,color,width=2,dash=[]){
  if(points.length<2)return;
  ctx.beginPath();ctx.moveTo(points[0][0],points[0][1]);
  for(let i=1;i<points.length;i++)ctx.lineTo(points[i][0],points[i][1]);
  ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);
}
function circle(ctx,x,y,r,color){
  ctx.beginPath();ctx.arc(x,y,r,0,2*Math.PI);ctx.fillStyle=color;ctx.fill();
}
function ellipse(ctx,mean,cov,toX,toY,scale,color,dashed=false){
  const a=cov[0][0],b=(cov[0][1]+cov[1][0])/2,d=cov[1][1];
  const trace=a+d,disc=Math.sqrt(Math.max(0,(a-d)**2+4*b*b));
  const l1=Math.max(0,(trace+disc)/2),l2=Math.max(0,(trace-disc)/2);
  const angle=.5*Math.atan2(2*b,a-d),radius=2.448;
  ctx.beginPath();
  for(let i=0;i<=64;i++){
    const t=i*2*Math.PI/64;
    const dx=radius*(Math.sqrt(l1)*Math.cos(t)*Math.cos(angle)-Math.sqrt(l2)*Math.sin(t)*Math.sin(angle));
    const dy=radius*(Math.sqrt(l1)*Math.cos(t)*Math.sin(angle)+Math.sqrt(l2)*Math.sin(t)*Math.cos(angle));
    const x=toX(mean[0]+dx),y=toY(mean[1]+dy);
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  }
  ctx.strokeStyle=color;ctx.lineWidth=dashed?1.4:2;ctx.setLineDash(dashed?[5,4]:[]);ctx.stroke();ctx.setLineDash([]);
}
function robot(ctx,x,y,theta,color,scale){
  circle(ctx,x,y,5,color);
  line(ctx,[[x,y],[x+11*Math.cos(theta),y-11*Math.sin(theta)]],color,2);
}
export function drawTrajectory(canvas,result,index){
  const {ctx,width,height}=canvasContext(canvas);
  if(!result){ctx.fillStyle=COLORS.muted;ctx.textAlign='center';ctx.fillText('Run the simulation to see the trajectories.',width/2,height/2);return;}
  const shown=result.frames[Math.max(0,Math.min(index,result.frames.length-1))];
  const all=[...result.landmarks,...result.frames.map(f=>f.truth),...result.frames.map(f=>f.odometry),...result.frames.map(f=>f.estimate)];
  const margin=Math.max(1.5,Math.min(6,2.448*Math.sqrt(Math.max(...result.frames.map(f=>Math.max(f.covariance[0][0],f.covariance[1][1]))))));
  let minX=Math.min(...all.map(p=>p[0]))-margin,maxX=Math.max(...all.map(p=>p[0]))+margin;
  let minY=Math.min(...all.map(p=>p[1]))-margin,maxY=Math.max(...all.map(p=>p[1]))+margin;
  const left=44,right=18,top=18,bottom=36,plotW=width-left-right,plotH=height-top-bottom;
  const scale=Math.min(plotW/(maxX-minX),plotH/(maxY-minY));
  const extraX=(plotW-(maxX-minX)*scale)/2,extraY=(plotH-(maxY-minY)*scale)/2;
  const toX=x=>left+extraX+(x-minX)*scale,toY=y=>height-bottom-extraY-(y-minY)*scale;
  const axisX=Math.floor(minX/5)*5,axisY=Math.floor(minY/5)*5;
  ctx.strokeStyle=COLORS.grid;ctx.fillStyle=COLORS.muted;ctx.lineWidth=1;
  for(let x=axisX;x<=maxX;x+=5){const px=toX(x);if(px<left||px>width-right)continue;line(ctx,[[px,top],[px,height-bottom]],COLORS.grid,1);ctx.textAlign='center';ctx.fillText(String(x),px,height-bottom+17);}
  for(let y=axisY;y<=maxY;y+=5){const py=toY(y);if(py<top||py>height-bottom)continue;line(ctx,[[left,py],[width-right,py]],COLORS.grid,1);ctx.textAlign='right';ctx.fillText(String(y),left-8,py+3);}
  ctx.textAlign='center';ctx.fillText('x [m]',width/2,height-5);ctx.save();ctx.translate(13,height/2);ctx.rotate(-Math.PI/2);ctx.fillText('y [m]',0,0);ctx.restore();
  const detected=new Set(shown.detected);
  result.landmarks.forEach((p,j)=>{
    const x=toX(p[0]),y=toY(p[1]);ctx.fillStyle=detected.has(j)?COLORS.detected:'#62758e';
    ctx.fillRect(x-4,y-4,8,8);
    if(detected.has(j)){circle(ctx,x,y,8,'#1a9a7529');line(ctx,[[toX(shown.truth[0]),toY(shown.truth[1])],[x,y]],'#1a9a7555',1);}
  });
  const frames=result.frames.slice(0,index+1);
  line(ctx,frames.map(f=>[toX(f.truth[0]),toY(f.truth[1])]),COLORS.truth,2.4);
  line(ctx,frames.map(f=>[toX(f.odometry[0]),toY(f.odometry[1])]),COLORS.odometry,1.8,[5,4]);
  line(ctx,frames.filter((f,i)=>i===0||f.update).map(f=>[toX(f.estimate[0]),toY(f.estimate[1])]),COLORS.estimate,2.4);
  // Keep three prediction/correction pairs visible as playback passes each selected update.
  const snapshots=result.snapshotIndices||[];
  for(const snapshotIndex of snapshots){
    if(snapshotIndex>index)continue;
    const f=result.frames[snapshotIndex];
    if(!f.prior)continue;
    ctx.save();ctx.globalAlpha=snapshotIndex===index?.95:.58;
    ellipse(ctx,f.prior.mean,f.prior.covariance,toX,toY,scale,'#64748b',true);
    if(f.detected.length)ellipse(ctx,f.estimate,f.covariance,toX,toY,scale,COLORS.detected);
    ctx.restore();
    ctx.fillStyle='#227548';ctx.textAlign='left';
    ctx.fillText('t = '+f.time.toFixed(1)+' s',toX(f.estimate[0])+8,toY(f.estimate[1])-10);
  }
  if(!snapshots.includes(index)){
    if(shown.prior)ellipse(ctx,shown.prior.mean,shown.prior.covariance,toX,toY,scale,'#64748b',true);
    if(!shown.prior||shown.detected.length)ellipse(ctx,shown.estimate,shown.covariance,toX,toY,scale,shown.detected.length?COLORS.detected:'#bba5dc');
  }
  robot(ctx,toX(shown.truth[0]),toY(shown.truth[1]),shown.truth[2],COLORS.truth,scale);
  robot(ctx,toX(shown.odometry[0]),toY(shown.odometry[1]),shown.odometry[2],COLORS.odometry,scale);
  robot(ctx,toX(shown.estimate[0]),toY(shown.estimate[1]),shown.estimate[2],COLORS.estimate,scale);
}
export function drawErrorChart(canvas,result,index){
  const {ctx,width,height}=canvasContext(canvas);
  if(!result){ctx.fillStyle=COLORS.muted;ctx.textAlign='center';ctx.fillText('Run the simulation to see position error.',width/2,height/2);return;}
  const left=55,right=19,top=18,bottom=39,plotW=width-left-right,plotH=height-top-bottom;
  const duration=result.frames.at(-1).time;
  const maxError=Math.max(.2,...result.ekfErrors.map(p=>p.error),...result.odometryErrors.map(p=>p.error))*1.12;
  const toX=t=>left+t/duration*plotW,toY=e=>height-bottom-e/maxError*plotH;
  for(let i=0;i<=4;i++){
    const y=i*maxError/4,py=toY(y);line(ctx,[[left,py],[width-right,py]],COLORS.grid,1);
    ctx.fillStyle=COLORS.muted;ctx.textAlign='right';ctx.fillText(y.toFixed(1),left-7,py+3);
  }
  for(let i=0;i<=4;i++){
    const t=i*duration/4,px=toX(t);line(ctx,[[px,top],[px,height-bottom]],COLORS.grid,1);
    ctx.fillStyle=COLORS.muted;ctx.textAlign='center';ctx.fillText(t.toFixed(0),px,height-bottom+17);
  }
  ctx.fillText('time [s]',width/2,height-5);
  ctx.save();ctx.translate(12,height/2);ctx.rotate(-Math.PI/2);ctx.fillText('position error [m]',0,0);ctx.restore();
  line(ctx,result.odometryErrors.map(p=>[toX(p.time),toY(p.error)]),COLORS.odometry,1.8);
  line(ctx,result.ekfErrors.map(p=>[toX(p.time),toY(p.error)]),COLORS.estimate,2.3);
  const current=result.frames[index],x=toX(current.time);
  line(ctx,[[x,top],[x,height-bottom]],'#2454d788',1,[3,3]);
  circle(ctx,x,toY(current.odometryError),4,COLORS.odometry);
  if(current.update)circle(ctx,x,toY(current.ekfError),4,COLORS.estimate);
}

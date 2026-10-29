import {EKF_LANDMARKS} from './gaussian-filters-core.js?v=3';

const $=id=>document.getElementById(id);
const MAX_POINTS=20;
const copy=points=>points.map(point=>[...point]);

export function createLandmarkEditor(onChange,{defaultLandmarks=EKF_LANDMARKS,downloadName='ekf-landmark-map.json',gridMap=null,axisRange=[-20,20]}={}){
  let points=copy(defaultLandmarks),map=gridMap,clickMode='landmark';
  const host=$('ekf-landmark-rows'),canvas=$('ekf-landmark-map'),error=$('ekf-map-error');
  const bounds=axis=>map?[0,(axis===0?map.rows:map.cols)-.1]:axisRange;
  const inBounds=(value,axis)=>Number.isFinite(value)&&value>=bounds(axis)[0]&&value<=bounds(axis)[1];
  const validPoint=point=>Array.isArray(point)&&point.length===2&&point.every(inBounds);
  const clamp=(value,axis)=>Math.max(bounds(axis)[0],Math.min(bounds(axis)[1],value));
  const number=value=>Number(value.toFixed(1));
  function showError(message){error.textContent=message;error.hidden=!message;}
  function geometry(width,height){
    const left=29,right=12,top=12,bottom=27,plotW=width-left-right,plotH=height-top-bottom;
    const xMin=map?0:axisRange[0],xMax=map?map.rows:axisRange[1],yMin=map?0:axisRange[0],yMax=map?map.cols:axisRange[1];
    return {left,top,plotW,plotH,
      toX:x=>left+(x-xMin)/(xMax-xMin)*plotW,
      toY:y=>height-bottom-(y-yMin)/(yMax-yMin)*plotH,
      toWorld:(px,py)=>[xMin+(px-left)/plotW*(xMax-xMin),yMin+(height-bottom-py)/plotH*(yMax-yMin)]};
  }
  function draw(){
    const rect=canvas.getBoundingClientRect(),width=Math.max(1,rect.width),height=Math.max(1,rect.height);
    const dpr=Math.min(window.devicePixelRatio||1,2);
    canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.fillStyle='#fbfdff';ctx.fillRect(0,0,width,height);
    const g=geometry(width,height);
    ctx.font='10px Inter, system-ui, sans-serif';ctx.lineWidth=1;
    if(map){
      for(let x=0;x<map.rows;x++)for(let y=0;y<map.cols;y++){
        ctx.fillStyle=map.cells[x][y]===1?'#2d394d':map.cells[x][y]===-1?'#a4adb8':'#fff';
        ctx.fillRect(g.toX(x),g.toY(y+1),g.plotW/map.rows+.3,g.plotH/map.cols+.3);
      }
      ctx.strokeStyle='#dbe4f0';
      for(let x=0;x<=map.rows;x+=2){ctx.beginPath();ctx.moveTo(g.toX(x),g.top);ctx.lineTo(g.toX(x),g.top+g.plotH);ctx.stroke();}
      for(let y=0;y<=map.cols;y+=2){ctx.beginPath();ctx.moveTo(g.left,g.toY(y));ctx.lineTo(g.left+g.plotW,g.toY(y));ctx.stroke();}
      ctx.fillStyle='#64748b';
      ctx.textAlign='center';for(let x=0;x<=map.rows;x+=5)ctx.fillText(String(x),g.toX(x),height-6);
      ctx.textAlign='right';for(let y=0;y<=map.cols;y+=5)ctx.fillText(String(y),g.left-5,g.toY(y)+3);
    }else{
      const tick=axisRange[1]-axisRange[0]<=30?5:10;
      for(let n=axisRange[0];n<=axisRange[1];n+=tick){
        const x=g.toX(n),y=g.toY(n);
        ctx.beginPath();ctx.moveTo(x,g.top);ctx.lineTo(x,g.top+g.plotH);ctx.moveTo(g.left,y);ctx.lineTo(g.left+g.plotW,y);
        ctx.strokeStyle=n===0?'#b9c8da':'#e6edf6';ctx.stroke();
        ctx.fillStyle='#64748b';ctx.textAlign='center';ctx.fillText(String(n),x,height-6);
        ctx.textAlign='right';ctx.fillText(String(n),g.left-5,y+3);
      }
    }
    points.forEach((point,index)=>{
      if(!point.every(inBounds))return;
      const x=g.toX(point[0]),y=g.toY(point[1]);
      ctx.fillStyle='#2454d7';ctx.fillRect(x-4,y-4,8,8);
      ctx.textAlign='left';ctx.fillStyle='#17243d';ctx.fillText(String(index+1),x+6,y-5);
    });
    const startX=$('startX').valueAsNumber,startY=$('startY').valueAsNumber,heading=$('startTheta').valueAsNumber;
    if(inBounds(startX,0)&&inBounds(startY,1)&&Number.isFinite(heading)){
      const x=g.toX(startX),y=g.toY(startY),angle=heading*Math.PI/180;
      const dx=Math.cos(angle),dy=-Math.sin(angle),px=-dy,py=dx;
      ctx.beginPath();ctx.moveTo(x+13*dx,y+13*dy);
      ctx.lineTo(x-6*dx+5*px,y-6*dy+5*py);ctx.lineTo(x-6*dx-5*px,y-6*dy-5*py);ctx.closePath();
      ctx.fillStyle='#ec8b30';ctx.fill();ctx.textAlign='left';ctx.fillText('start',x+9,y+16);
    }
    if(!points.length){ctx.textAlign='center';ctx.fillStyle='#64748b';ctx.fillText('Click to add a landmark',width/2,height/2+40);}
  }
  function render(){
    host.replaceChildren();
    points.forEach((point,index)=>{
      const row=document.createElement('div');row.className='ekf-landmark-row';
      const name=document.createElement('span');name.textContent='Landmark '+(index+1);row.append(name);
      for(let axis=0;axis<2;axis++){
        const input=document.createElement('input');
        Object.assign(input,{type:'number',min:bounds(axis)[0],max:bounds(axis)[1],step:.1,value:point[axis].toFixed(1),id:'ekf-landmark-'+index+'-'+axis});
        input.setAttribute('aria-label','Landmark '+(index+1)+' '+(axis===0?'x':'y')+' position in '+(map?'map cells':'meters'));
        input.addEventListener('input',()=>{if(inBounds(input.valueAsNumber,axis)){points[index][axis]=input.valueAsNumber;draw();}onChange();});
        input.addEventListener('change',()=>{if(inBounds(input.valueAsNumber,axis)){input.value=number(input.valueAsNumber).toFixed(1);points[index][axis]=input.valueAsNumber;draw();}onChange();});
        row.append(input);
      }
      const remove=document.createElement('button');remove.type='button';remove.className='button quiet';
      remove.textContent='×';remove.setAttribute('aria-label','Remove landmark '+(index+1));
      remove.addEventListener('click',()=>{points.splice(index,1);render();onChange();});
      row.append(remove);host.append(row);
    });
    draw();
  }
  function setPoints(value){
    if(!Array.isArray(value)||value.length>MAX_POINTS||!value.every(validPoint))throw new Error('Map must contain up to 20 valid [x, y] points within the displayed map.');
    points=value.map(point=>point.map(number));render();showError('');onChange();
  }
  function add(point){
    if(points.length>=MAX_POINTS){showError('The map can contain at most 20 landmarks.');return;}
    points.push(point);render();showError('');onChange();
  }
  function read(){
    return Array.from(host.children,(row,index)=>Array.from(row.querySelectorAll('input'),(input,axis)=>{
      const value=input.valueAsNumber;
      if(!inBounds(value,axis)||input.validity.stepMismatch){input.focus();throw new Error('Landmark '+(index+1)+' '+(axis===0?'x':'y')+' must be within the displayed map in 0.1 steps.');}
      return value;
    }));
  }
  function signature(){return JSON.stringify(Array.from(host.children,row=>Array.from(row.querySelectorAll('input'),input=>input.value)));}
  function readPose(){
    const values=['startX','startY','startTheta'].map((id,axis)=>{
      const input=$(id),value=input.valueAsNumber,valid=axis===2?Number.isFinite(value)&&Math.abs(value)<=180:inBounds(value,axis);
      if(!valid||input.validity.stepMismatch){input.focus();throw new Error('Robot starting pose must be within the displayed map.');}
      return value;
    });
    return {x:values[0],y:values[1],headingDegrees:values[2]};
  }
  function setPose(value){
    if(!value||typeof value!=='object'||!inBounds(value.x,0)||!inBounds(value.y,1)||!Number.isFinite(value.headingDegrees)||Math.abs(value.headingDegrees)>180)throw new Error('Robot start must be within the displayed map.');
    for(const [id,raw,decimals] of [['startX',value.x,1],['startY',value.y,1],['startTheta',value.headingDegrees,2]]){
      const rounded=Number(raw.toFixed(decimals));$(id).value=rounded.toFixed(decimals);$(id+'-slider').value=String(rounded);
    }
    draw();onChange();
  }
  function setGridMap(value){map=value;render();showError('');}
  function setClickMode(value){
    clickMode=value;
    for(const [id,mode] of [['ekf-click-landmark','landmark'],['ekf-click-robot','robot']]){
      const button=$(id);if(button){button.setAttribute('aria-pressed',String(mode===value));button.classList.toggle('active',mode===value);}
    }
  }
  $('ekf-add-landmark').addEventListener('click',()=>add(map?[.5,.5]:[0,0]));
  $('ekf-reset-landmarks').addEventListener('click',()=>setPoints(defaultLandmarks));
  $('ekf-clear-landmarks').addEventListener('click',()=>setPoints([]));
  $('ekf-click-landmark')?.addEventListener('click',()=>setClickMode('landmark'));
  $('ekf-click-robot')?.addEventListener('click',()=>setClickMode('robot'));
  canvas.addEventListener('click',event=>{
    const rect=canvas.getBoundingClientRect(),g=geometry(rect.width,rect.height);
    const [rawX,rawY]=g.toWorld(event.clientX-rect.left,event.clientY-rect.top);
    const x=number(clamp(rawX,0)),y=number(clamp(rawY,1));
    if(clickMode==='robot'){
      if(map&&map.cells[Math.floor(x)][Math.floor(y)]!==0){showError('Choose a free map cell for the robot.');return;}
      setPose({x,y,headingDegrees:$('startTheta').valueAsNumber});showError('');
    }else add([x,y]);
  });
  $('ekf-save-map').addEventListener('click',()=>{
    try{
      const data=JSON.stringify({startPose:readPose(),landmarks:read()},null,2);
      const url=URL.createObjectURL(new Blob([data],{type:'application/json'}));
      const link=document.createElement('a');link.href=url;link.download=downloadName;
      document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
      showError('');
    }catch(cause){showError(cause.message);}
  });
  $('ekf-load-map').addEventListener('change',async event=>{
    const file=event.target.files?.[0];if(!file)return;
    try{
      const parsed=JSON.parse(await file.text()),landmarks=Array.isArray(parsed)?parsed:parsed.landmarks;
      if(!Array.isArray(landmarks)||landmarks.length>MAX_POINTS||!landmarks.every(validPoint))throw new Error('Map must contain up to 20 landmarks within the displayed map.');
      if(!Array.isArray(parsed)&&parsed.startPose!==undefined){
        const pose=parsed.startPose;
        if(!pose||!inBounds(pose.x,0)||!inBounds(pose.y,1)||!Number.isFinite(pose.headingDegrees)||Math.abs(pose.headingDegrees)>180)throw new Error('Invalid robot starting pose in map file.');
      }
      setPoints(landmarks);
      if(!Array.isArray(parsed)&&parsed.startPose!==undefined)setPose(parsed.startPose);
    }catch(cause){showError(cause.message);}
    event.target.value='';
  });
  for(const id of ['startX','startY','startTheta'])for(const suffix of ['', '-slider'])$(id+suffix).addEventListener('input',draw);
  if('ResizeObserver' in window)new ResizeObserver(draw).observe(canvas);
  else window.addEventListener('resize',draw);
  render();setClickMode('landmark');
  return {read,signature,readPose,draw,setGridMap};
}

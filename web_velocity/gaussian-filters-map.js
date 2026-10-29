import {EKF_LANDMARKS} from './gaussian-filters-core.js?v=3';

const $=id=>document.getElementById(id);
const LIMIT=20,MAX_POINTS=20;
const validPoint=point=>Array.isArray(point)&&point.length===2&&point.every(value=>typeof value==='number'&&Number.isFinite(value)&&value>=-LIMIT&&value<=LIMIT);
const copy=points=>points.map(point=>[...point]);
const clamp=value=>Math.max(-LIMIT,Math.min(LIMIT,value));

export function createLandmarkEditor(onChange){
  let points=copy(EKF_LANDMARKS);
  const host=$('ekf-landmark-rows'),canvas=$('ekf-landmark-map'),error=$('ekf-map-error');
  function showError(message){error.textContent=message;error.hidden=!message;}
  function geometry(width,height){
    const left=27,right=12,top=12,bottom=26;
    return {left,top,plotW:width-left-right,plotH:height-top-bottom,
      toX:x=>left+(x+LIMIT)/(2*LIMIT)*(width-left-right),
      toY:y=>height-bottom-(y+LIMIT)/(2*LIMIT)*(height-top-bottom)};
  }
  function draw(){
    const rect=canvas.getBoundingClientRect(),width=Math.max(1,rect.width),height=Math.max(1,rect.height);
    const dpr=Math.min(window.devicePixelRatio||1,2);
    canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.fillStyle='#fbfdff';ctx.fillRect(0,0,width,height);
    const g=geometry(width,height);
    ctx.font='10px Inter, system-ui, sans-serif';ctx.lineWidth=1;
    for(let n=-20;n<=20;n+=10){
      const x=g.toX(n),y=g.toY(n);
      ctx.beginPath();ctx.moveTo(x,g.top);ctx.lineTo(x,g.top+g.plotH);ctx.moveTo(g.left,y);ctx.lineTo(g.left+g.plotW,y);
      ctx.strokeStyle=n===0?'#b9c8da':'#e6edf6';ctx.stroke();
      ctx.fillStyle='#64748b';ctx.textAlign='center';ctx.fillText(String(n),x,height-6);
      ctx.textAlign='right';ctx.fillText(String(n),g.left-5,y+3);
    }
    points.forEach((point,index)=>{
      const x=g.toX(point[0]),y=g.toY(point[1]);
      ctx.fillStyle='#2454d7';ctx.fillRect(x-4,y-4,8,8);
      ctx.textAlign='left';ctx.fillStyle='#17243d';ctx.fillText(String(index+1),x+6,y-5);
    });
    const startX=$('startX').valueAsNumber,startY=$('startY').valueAsNumber,heading=$('startTheta').valueAsNumber;
    if(Number.isFinite(startX)&&Number.isFinite(startY)&&Number.isFinite(heading)&&Math.abs(startX)<=LIMIT&&Math.abs(startY)<=LIMIT){
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
        Object.assign(input,{type:'number',min:-LIMIT,max:LIMIT,step:.1,value:point[axis].toFixed(1),id:'ekf-landmark-'+index+'-'+axis});
        input.setAttribute('aria-label','Landmark '+(index+1)+' '+(axis===0?'x':'y')+' position in meters');
        input.addEventListener('input',()=>{
          if(Number.isFinite(input.valueAsNumber)&&input.valueAsNumber>=-LIMIT&&input.valueAsNumber<=LIMIT){
            points[index][axis]=input.valueAsNumber;draw();
          }
          onChange();
        });
        input.addEventListener('change',()=>{
          if(Number.isFinite(input.valueAsNumber)&&input.valueAsNumber>=-LIMIT&&input.valueAsNumber<=LIMIT){
            input.value=(Math.round(input.valueAsNumber*10)/10).toFixed(1);
            points[index][axis]=input.valueAsNumber;draw();
          }
          onChange();
        });
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
    if(!Array.isArray(value)||value.length>MAX_POINTS||!value.every(validPoint))throw new Error('Map must contain up to 20 [x, y] points between −20 and 20 m.');
    points=value.map(point=>point.map(coordinate=>Math.round(coordinate*10)/10));render();showError('');onChange();
  }
  function add(point){
    if(points.length>=MAX_POINTS){showError('The map can contain at most 20 landmarks.');return;}
    points.push(point);render();showError('');onChange();
  }
  function read(){
    return Array.from(host.children,(row,index)=>Array.from(row.querySelectorAll('input'),(input,axis)=>{
      const value=input.valueAsNumber;
      if(!Number.isFinite(value)||value< -LIMIT||value>LIMIT||input.validity.stepMismatch){
        input.focus();throw new Error('Landmark '+(index+1)+' '+(axis===0?'x':'y')+': enter a value from −20 to 20 in 0.1 m steps.');
      }
      return value;
    }));
  }
  function signature(){
    return JSON.stringify(Array.from(host.children,row=>Array.from(row.querySelectorAll('input'),input=>input.value)));
  }
  function readPose(){
    const values=[['startX',-20,20],['startY',-20,20],['startTheta',-180,180]].map(([id,min,max])=>{
      const input=$(id),value=input.valueAsNumber;
      if(!Number.isFinite(value)||value<min||value>max||input.validity.stepMismatch){
        input.focus();throw new Error('Robot starting pose: enter valid x, y, and heading values.');
      }
      return value;
    });
    return {x:values[0],y:values[1],headingDegrees:values[2]};
  }
  function setPose(value){
    if(!value||typeof value!=='object'||!Number.isFinite(value.x)||!Number.isFinite(value.y)||!Number.isFinite(value.headingDegrees)||Math.abs(value.x)>LIMIT||Math.abs(value.y)>LIMIT||Math.abs(value.headingDegrees)>180){
      throw new Error('Robot start must contain x and y from −20 to 20 m and heading from −180° to 180°.');
    }
    for(const [id,raw,decimals] of [['startX',value.x,1],['startY',value.y,1],['startTheta',value.headingDegrees,2]]){
      const rounded=Number(raw.toFixed(decimals));
      $(id).value=rounded.toFixed(decimals);$(id+'-slider').value=String(rounded);
    }
    draw();onChange();
  }
  $('ekf-add-landmark').addEventListener('click',()=>add([0,0]));
  $('ekf-reset-landmarks').addEventListener('click',()=>setPoints(EKF_LANDMARKS));
  $('ekf-clear-landmarks').addEventListener('click',()=>setPoints([]));
  canvas.addEventListener('click',event=>{
    const rect=canvas.getBoundingClientRect(),g=geometry(rect.width,rect.height);
    const x=clamp((event.clientX-rect.left-g.left)/g.plotW*2*LIMIT-LIMIT);
    const y=clamp((g.top+g.plotH-(event.clientY-rect.top))/g.plotH*2*LIMIT-LIMIT);
    add([Math.round(x*10)/10,Math.round(y*10)/10]);
  });
  $('ekf-save-map').addEventListener('click',()=>{
    try{
      const data=JSON.stringify({startPose:readPose(),landmarks:read()},null,2);
      const url=URL.createObjectURL(new Blob([data],{type:'application/json'}));
      const link=document.createElement('a');link.href=url;link.download='ekf-landmark-map.json';
      document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
      showError('');
    }catch(cause){showError(cause.message);}
  });
  $('ekf-load-map').addEventListener('change',async event=>{
    const file=event.target.files?.[0];if(!file)return;
    try{
      const parsed=JSON.parse(await file.text()),landmarks=Array.isArray(parsed)?parsed:parsed.landmarks;
      if(!Array.isArray(landmarks)||landmarks.length>MAX_POINTS||!landmarks.every(validPoint))throw new Error('Map must contain up to 20 [x, y] points between −20 and 20 m.');
      if(!Array.isArray(parsed)&&parsed.startPose!==undefined){
        const pose=parsed.startPose;
        if(!pose||!Number.isFinite(pose.x)||!Number.isFinite(pose.y)||!Number.isFinite(pose.headingDegrees)||Math.abs(pose.x)>LIMIT||Math.abs(pose.y)>LIMIT||Math.abs(pose.headingDegrees)>180)throw new Error('Invalid robot starting pose in map file.');
      }
      setPoints(landmarks);
      if(!Array.isArray(parsed)&&parsed.startPose!==undefined)setPose(parsed.startPose);
    }
    catch(cause){showError(cause.message);}
    event.target.value='';
  });
  for(const id of ['startX','startY','startTheta'])for(const suffix of ['', '-slider'])$(id+suffix).addEventListener('input',draw);
  if('ResizeObserver' in window)new ResizeObserver(draw).observe(canvas);
  else window.addEventListener('resize',draw);
  render();
  return {read,signature,readPose};
}

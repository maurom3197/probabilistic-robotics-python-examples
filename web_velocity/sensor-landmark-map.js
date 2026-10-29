const LIMIT=20;
const clamp=value=>Math.max(-LIMIT,Math.min(LIMIT,value));
const round=value=>Math.round(value*10)/10;

export function createLandmarkMapEditor({getPose,getLandmarks,placeRobot,setHeading,addLandmark,moveLandmark,reset}){
  const canvas=document.getElementById('landmark-map-editor');
  const instruction=document.getElementById('landmark-map-instruction');
  const robotButton=document.getElementById('landmark-place-robot');
  const landmarkButton=document.getElementById('landmark-add-point');
  const headingButton=document.getElementById('landmark-set-heading');
  let mode='robot',drag=null;

  function geometry(width,height){
    const side=Math.max(1,Math.min(width-46,height-44));
    const left=(width-side)/2,top=(height-side)/2-4;
    return {left,top,side,
      toX:x=>left+(x+LIMIT)/(2*LIMIT)*side,
      toY:y=>top+(LIMIT-y)/(2*LIMIT)*side,
      toWorld:(x,y)=>[round(clamp((x-left)/side*2*LIMIT-LIMIT)),round(clamp(LIMIT-(y-top)/side*2*LIMIT))]};
  }
  function draw(){
    const rect=canvas.getBoundingClientRect(),width=Math.max(rect.width,1),height=Math.max(rect.height,1);
    const ratio=Math.min(window.devicePixelRatio||1,2);
    const pixelWidth=Math.round(width*ratio),pixelHeight=Math.round(height*ratio);
    if(canvas.width!==pixelWidth||canvas.height!==pixelHeight){canvas.width=pixelWidth;canvas.height=pixelHeight;}
    const context=canvas.getContext('2d');context.setTransform(ratio,0,0,ratio,0,0);
    context.fillStyle='#fbfdff';context.fillRect(0,0,width,height);
    const g=geometry(width,height);
    context.font='10px system-ui';context.lineWidth=1;
    for(let n=-20;n<=20;n+=10){
      const x=g.toX(n),y=g.toY(n);
      context.beginPath();context.moveTo(x,g.top);context.lineTo(x,g.top+g.side);
      context.moveTo(g.left,y);context.lineTo(g.left+g.side,y);
      context.strokeStyle=n===0?'#b8c7da':'#e5ecf6';context.stroke();
      context.fillStyle='#64748b';context.textAlign='center';context.fillText(String(n),x,g.top+g.side+14);
      context.textAlign='right';context.fillText(String(n),g.left-5,y+3);
    }
    const pose=getPose(),landmarks=getLandmarks();
    if(pose.every(Number.isFinite)){
      const [x,y,heading]=pose,px=g.toX(x),py=g.toY(y),theta=heading*Math.PI/180;
      const range=Number(document.getElementById('lm-range').value),fov=Number(document.getElementById('lm-fov').value)*Math.PI/180;
      if(Number.isFinite(range)&&Number.isFinite(fov)){
        context.beginPath();context.moveTo(px,py);
        for(let i=0;i<=48;i++){
          const angle=theta-fov/2+i*fov/48;
          context.lineTo(g.toX(x+range*Math.cos(angle)),g.toY(y+range*Math.sin(angle)));
        }
        context.closePath();context.fillStyle='#2454d711';context.strokeStyle='#2454d766';
        context.fill();context.stroke();
      }
    }
    landmarks.forEach((point,index)=>{
      if(!point.every(Number.isFinite))return;
      const px=g.toX(point[0]),py=g.toY(point[1]);
      context.fillStyle='#2454d7';context.fillRect(px-5,py-5,10,10);
      context.textAlign='left';context.fillStyle='#17243d';context.fillText(String(index+1),px+7,py-7);
    });
    if(pose.every(Number.isFinite)){
      const px=g.toX(pose[0]),py=g.toY(pose[1]),angle=pose[2]*Math.PI/180;
      const dx=Math.cos(angle),dy=-Math.sin(angle),sideX=-dy,sideY=dx;
      context.beginPath();context.moveTo(px+14*dx,py+14*dy);
      context.lineTo(px-6*dx+5*sideX,py-6*dy+5*sideY);
      context.lineTo(px-6*dx-5*sideX,py-6*dy-5*sideY);
      context.closePath();context.fillStyle='#ec8b30';context.fill();
    }
  }
  function setMode(next){
    mode=next;
    robotButton.setAttribute('aria-pressed',String(mode==='robot'));
    landmarkButton.setAttribute('aria-pressed',String(mode==='landmark'));
    headingButton.setAttribute('aria-pressed',String(mode==='heading'));
    instruction.textContent=mode==='robot'
      ? 'Click an empty spot to place the robot. Drag the robot or a landmark to move it.'
      : mode==='heading' ? 'Click the direction the robot should face. The heading control updates too.'
      : 'Click an empty spot to add a landmark. Drag any existing marker to move it.';
  }
  function eventPoint(event){
    const rect=canvas.getBoundingClientRect();
    return {x:event.clientX-rect.left,y:event.clientY-rect.top,geometry:geometry(rect.width,rect.height)};
  }
  function nearestMarker(x,y,g){
    const pose=getPose(),landmarks=getLandmarks();
    let nearest=null,best=12**2;
    if(pose.every(Number.isFinite)){
      const d=(x-g.toX(pose[0]))**2+(y-g.toY(pose[1]))**2;
      if(d<best){nearest={kind:'robot'};best=d;}
    }
    landmarks.forEach((point,index)=>{
      if(!point.every(Number.isFinite))return;
      const d=(x-g.toX(point[0]))**2+(y-g.toY(point[1]))**2;
      if(d<best){nearest={kind:'landmark',index};best=d;}
    });
    return nearest;
  }
  canvas.addEventListener('pointerdown',event=>{
    const {x,y,geometry:g}=eventPoint(event);
    const marker=mode==='heading'?null:nearestMarker(x,y,g),point=g.toWorld(x,y);
    if(mode==='heading'){
      const pose=getPose(),dx=point[0]-pose[0],dy=point[1]-pose[1];
      if(pose.every(Number.isFinite)&&Math.hypot(dx,dy)>.1)setHeading(Math.round(Math.atan2(dy,dx)*180/Math.PI));
    }else if(marker){
      drag={...marker,pointerId:event.pointerId};
      canvas.setPointerCapture(event.pointerId);
    }else if(mode==='robot')placeRobot(...point);
    else addLandmark(...point);
    event.preventDefault();
  });
  canvas.addEventListener('pointermove',event=>{
    if(!drag||event.pointerId!==drag.pointerId)return;
    const {x,y,geometry:g}=eventPoint(event),point=g.toWorld(x,y);
    if(drag.kind==='robot')placeRobot(...point);
    else moveLandmark(drag.index,...point);
    event.preventDefault();
  });
  const finish=event=>{if(drag&&event.pointerId===drag.pointerId)drag=null;};
  canvas.addEventListener('pointerup',finish);
  canvas.addEventListener('pointercancel',finish);
  robotButton.addEventListener('click',()=>setMode('robot'));
  landmarkButton.addEventListener('click',()=>setMode('landmark'));
  headingButton.addEventListener('click',()=>setMode('heading'));
  document.getElementById('landmark-reset-map').addEventListener('click',reset);
  if('ResizeObserver' in window)new ResizeObserver(draw).observe(canvas);
  else window.addEventListener('resize',draw);
  setMode('robot');draw();
  return {draw};
}

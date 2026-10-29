export const PF_LANDMARKS=[
  [10.5,12.5],[10.5,13.5],[19.5,7.5],[19.5,8.5],[19.5,9.5],[18.5,9.5],
  [10.5,16.5],[11.5,16.5],[12.5,16.5],[16.5,16.5],[17.5,16.5],
  [9.5,7.5],[11.5,4.5],[15.5,2.5],[.5,12.5],[.5,16.5],
];
export const PF_MAP_FILES=['map0.png','map1.png','map2.png','map3.png','map31.png','map31_grey.png','map4.png'];
const TAU=2*Math.PI,DEG=Math.PI/180;
const wrap=angle=>Math.atan2(Math.sin(angle),Math.cos(angle));
const distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
const sinc=x=>Math.abs(x)<1e-5?1-x*x/6+x*x*x*x/120:Math.sin(x)/x;
const isFree=(map,pose)=>pose[0]>=0&&pose[0]<map.rows&&pose[1]>=0&&pose[1]<map.cols&&map.cells[Math.floor(pose[0])][Math.floor(pose[1])]===0;

export async function loadPfMap(filename){
  if(!PF_MAP_FILES.includes(filename))throw new Error('Unknown map file.');
  const image=new Image();
  await new Promise((resolve,reject)=>{
    image.onload=resolve;image.onerror=()=>reject(new Error('Could not load '+filename));
    image.src=new URL('./maps/'+filename,import.meta.url).href;
  });
  const canvas=document.createElement('canvas');
  canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
  const context=canvas.getContext('2d',{willReadFrequently:true});
  context.drawImage(image,0,0);
  const pixels=context.getImageData(0,0,canvas.width,canvas.height).data;
  const rows=Math.ceil(canvas.height/3),cols=Math.ceil(canvas.width/3);
  const cells=Array.from({length:rows},()=>Array(cols).fill(0)),free=[],obstacles=[];
  for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
    let highest=0;
    for(let dr=0;dr<3;dr++)for(let dc=0;dc<3;dc++){
      const r=row*3+dr,c=col*3+dc;if(r>=canvas.height||c>=canvas.width)continue;
      const i=4*(r*canvas.width+c);
      highest=Math.max(highest,Math.round(.299*pixels[i]+.587*pixels[i+1]+.114*pixels[i+2]));
    }
    const cell=highest===255?1:highest===128?-1:0;
    cells[row][col]=cell;
    if(cell===0)free.push([row,col]);else if(cell===1)obstacles.push([row,col]);
  }
  if(!free.length||!obstacles.length)throw new Error('Map must contain free and occupied cells.');
  const distances=cells.map((line,row)=>line.map((cell,col)=>{
    let best=Infinity;
    for(const [r,c] of obstacles)best=Math.min(best,Math.hypot(r-row,c-col));
    return best;
  }));
  const maxDistance=Math.max(...distances.flat());
  for(let row=0;row<rows;row++)for(let col=0;col<cols;col++)if(cells[row][col]===-1)distances[row][col]=maxDistance;
  return {filename,rows,cols,cells,free,obstacles,distances,maxDistance};
}
export function nearestPfFree(map,x,y){
  let best=null,score=Infinity;
  for(const [row,col] of map.free){
    const d=(row+.5-x)**2+(col+.5-y)**2;
    if(d<score){score=d;best=[row+.5,col+.5];}
  }
  return best;
}
function randomGenerator(seed){
  let state=seed>>>0,spare=null;
  const uniform=()=>{
    state=(state+0x6D2B79F5)|0;
    let t=Math.imul(state^(state>>>15),1|state);
    t^=t+Math.imul(t^(t>>>7),61|t);
    return ((t^(t>>>14))>>>0)/4294967296;
  };
  const normal=sigma=>{
    if(sigma===0)return 0;
    if(spare!==null){const z=spare;spare=null;return sigma*z;}
    const radius=Math.sqrt(-2*Math.log(Math.max(uniform(),1e-12))),angle=TAU*uniform();
    spare=radius*Math.sin(angle);
    return sigma*radius*Math.cos(angle);
  };
  return {uniform,normal};
}
function sampleMotion(pose,v,omega,sigmaV,sigmaOmega,dt,rng){
  const noisyV=v+rng.normal(sigmaV),noisyOmega=omega+rng.normal(sigmaOmega);
  const half=noisyOmega*dt/2,mid=pose[2]+half,travel=noisyV*dt*sinc(half);
  return [pose[0]+travel*Math.cos(mid),pose[1]+travel*Math.sin(mid),pose[2]+noisyOmega*dt+rng.normal(sigmaOmega/2)*dt];
}
function initializeParticles(map,count,rng){
  return Array.from({length:count},()=>{
    const cell=map.free[Math.floor(rng.uniform()*map.free.length)];
    return [cell[0]+.05+.9*rng.uniform(),cell[1]+.05+.9*rng.uniform(),-Math.PI+TAU*rng.uniform()];
  });
}
function estimate(particles,weights){
  let x=0,y=0,sine=0,cosine=0;
  for(let i=0;i<particles.length;i++){
    const w=weights[i],p=particles[i];x+=w*p[0];y+=w*p[1];sine+=w*Math.sin(p[2]);cosine+=w*Math.cos(p[2]);
  }
  let xx=0,xy=0,yy=0;
  for(let i=0;i<particles.length;i++){
    const dx=particles[i][0]-x,dy=particles[i][1]-y,w=weights[i];
    xx+=w*dx*dx;xy+=w*dx*dy;yy+=w*dy*dy;
  }
  return {mean:[x,y,Math.atan2(sine,cosine)],covariance:[[xx,xy],[xy,yy]]};
}
function normalize(logWeights){
  const max=Math.max(...logWeights);
  if(!Number.isFinite(max))throw new Error('No particle has a valid sensor likelihood. Change the noise or sensor settings.');
  const weights=logWeights.map(value=>Math.exp(value-max));
  const total=weights.reduce((sum,w)=>sum+w,0);
  if(!(total>0))throw new Error('Particle weights vanished. Change the noise or sensor settings.');
  return weights.map(w=>w/total);
}
function resample(particles,weights,rng,stratified){
  const count=particles.length,cdf=[],next=[];
  let total=0,index=0;
  weights.forEach(w=>{total+=w;cdf.push(total);});cdf[count-1]=1;
  const offset=stratified?0:rng.uniform();
  for(let i=0;i<count;i++){
    const target=(i+offset+(stratified?rng.uniform():0))/count;
    while(index<count-1&&target>cdf[index])index++;
    next.push([...particles[index]]);
  }
  return next;
}
function snapshotCloud(particles){
  const stride=Math.max(1,Math.ceil(particles.length/350));
  return particles.filter((_,index)=>index%stride===0).map(p=>[p[0],p[1]]);
}
function landmarkMeasurement(pose,landmark,p,rng){
  const dx=landmark[0]-pose[0],dy=landmark[1]-pose[1];
  return [Math.hypot(dx,dy)+rng.normal(p.sigmaRange),Math.atan2(dy,dx)-pose[2]+rng.normal(p.sigmaBearing*DEG)];
}
function applyLandmarks(particles,logWeights,truth,p,landmarks,rng){
  const detected=[],bearingSigma=p.sigmaBearing*DEG;
  for(let j=0;j<landmarks.length;j++){
    const z=landmarkMeasurement(truth,landmarks[j],p,rng);
    if(z[0]>p.maxRange||Math.abs(z[1])>p.fov*DEG/2)continue;
    detected.push(j);
    for(let i=0;i<particles.length;i++){
      const dx=landmarks[j][0]-particles[i][0],dy=landmarks[j][1]-particles[i][1];
      const predictedRange=Math.hypot(dx,dy),predictedBearing=Math.atan2(dy,dx)-particles[i][2];
      const dr=(z[0]-predictedRange)/p.sigmaRange,db=wrap(z[1]-predictedBearing)/bearingSigma;
      logWeights[i]-=.5*(dr*dr+db*db);
    }
  }
  return detected;
}
function castRays(map,pose,numRays,fov,maxRange){
  const rays=[];
  for(let i=0;i<numRays;i++){
    const angle=pose[2]-fov*DEG/2+i*fov*DEG/numRays;
    let range=maxRange;
    for(let length=.25;length<maxRange;length+=.1){
      const x=pose[0]+length*Math.cos(angle),y=pose[1]+length*Math.sin(angle);
      if(x<0||x>=map.rows||y<0||y>=map.cols)break;
      if(map.cells[Math.floor(x)][Math.floor(y)]===1){range=length;break;}
    }
    rays.push({angle,range,endpoint:[pose[0]+range*Math.cos(angle),pose[1]+range*Math.sin(angle)]});
  }
  return rays;
}
function applyLaser(particles,logWeights,rays,p,map){
  const total=p.weightHit+p.weightMax+p.weightRand;
  const hit=p.weightHit/total,random=p.weightRand/total;
  if(hit+random<=0)throw new Error('Laser PF needs a positive hit or random weight; maximum-range beams are ignored.');
  const valid=rays.map((ray,index)=>ray.range<p.maxRange?index:-1).filter(index=>index>=0);
  for(let i=0;i<particles.length;i++){
    const pose=particles[i];
    for(const index of valid){
      const range=rays[index].range,angle=pose[2]-p.fov*DEG/2+index*p.fov*DEG/p.numRays;
      const x=pose[0]+range*Math.cos(angle),y=pose[1]+range*Math.sin(angle);
      const row=Math.floor(x),col=Math.floor(y);
      const d=row>=0&&row<map.rows&&col>=0&&col<map.cols?map.distances[row][col]:map.maxDistance;
      const probability=hit*Math.exp(-.5*(d/p.sigmaHit)**2)+random/p.maxRange;
      logWeights[i]+=Math.log(Math.max(probability,1e-300));
    }
  }
  return valid.length;
}
function makeFrame(time,truth,odometry,estimateState,detected,update,prior,cloud,rays,neff,resampled){
  return {time,truth:[...truth],odometry:[...odometry],estimate:[...estimateState.mean],
    covariance:estimateState.covariance.map(row=>[...row]),detected:[...detected],update,prior,cloud,rays,neff,resampled,
    pfError:distance(truth,estimateState.mean),odometryError:distance(truth,odometry)};
}
export function runPfSimulation(p,map){
  if(!['landmark','laser'].includes(p.mode))throw new Error('Choose a particle filter model.');
  if(!Number.isInteger(p.particles)||p.particles<50||p.particles>2000)throw new Error('Particle count must be between 50 and 2000.');
  if(!Number.isInteger(p.seed)||p.seed<0)throw new Error('Random seed must be a non-negative integer.');
  const defaultPose=p.mode==='landmark'?[19,2,120*DEG]:[19.5,4.5,150*DEG];
  const initialPose=p.initialPose??defaultPose,landmarks=p.mode==='landmark'?(p.landmarks??PF_LANDMARKS).map(point=>[...point]):[];
  if(!Array.isArray(initialPose)||initialPose.length!==3||!initialPose.every(Number.isFinite)||!isFree(map,initialPose))throw new Error('Place the starting robot pose in a free map cell.');
  if(p.mode==='landmark'&&(!Array.isArray(landmarks)||landmarks.length>20||!landmarks.every(point=>point.length===2&&point.every(value=>Number.isFinite(value)&&value>=-20&&value<=20))))throw new Error('Enter up to 20 valid landmarks.');
  const rng=randomGenerator(p.seed),dt=.1,steps=Math.round(p.duration/dt),pfStep=Math.max(1,Math.round(p.pfDt/dt));
  let truth=[...initialPose],odometry=[...initialPose],particles=initializeParticles(map,p.particles,rng);
  let weights=Array(p.particles).fill(1/p.particles),state=estimate(particles,weights);
  const frames=[makeFrame(0,truth,odometry,state,[],false,null,snapshotCloud(particles),null,p.particles,false)];
  const pfErrors=[{time:0,error:frames[0].pfError}],odometryErrors=[{time:0,error:0}];
  for(let i=0;i<steps;i++){
    const turned=p.scriptedTurn&&(p.mode==='landmark'?i>steps/2:i>2*steps/3);
    const v=turned?(p.mode==='landmark'?.65:.4):p.v;
    const omega=turned?(p.mode==='landmark'?-.03:-.05):p.omega;
    const simSigmaV=p.mode==='laser'?.01:p.sigmaV,simSigmaOmega=p.mode==='laser'?.001:p.sigmaOmega;
    truth=sampleMotion(truth,v,omega,simSigmaV,simSigmaOmega,dt,rng);
    odometry=sampleMotion(odometry,v,omega,simSigmaV,simSigmaOmega,dt,rng);
    let prior=null,detected=[],rays=null,neff=null,resampled=false,update=false,cloud=null;
    if(i>0&&i%pfStep===0){
      update=true;
      particles=particles.map(particle=>sampleMotion(particle,v,omega,p.sigmaV,p.sigmaOmega,p.pfDt,rng));
      prior=estimate(particles,weights);
      const logWeights=weights.map(weight=>Math.log(Math.max(weight,1e-300)));
      if(p.mode==='landmark')detected=applyLandmarks(particles,logWeights,truth,p,landmarks,rng);
      else{
        rays=castRays(map,truth,p.numRays,p.fov,p.maxRange);
        for(const ray of rays)if(ray.range<p.maxRange){
          ray.range=Math.max(0,Math.min(p.maxRange,ray.range+rng.normal(p.sigmaLaser)));
          ray.endpoint=[truth[0]+ray.range*Math.cos(ray.angle),truth[1]+ray.range*Math.sin(ray.angle)];
        }
        applyLaser(particles,logWeights,rays,p,map);
      }
      weights=normalize(logWeights);
      neff=1/weights.reduce((sum,w)=>sum+w*w,0);
      if(neff<p.particles/2){
        particles=resample(particles,weights,rng,p.mode==='landmark');
        weights=Array(p.particles).fill(1/p.particles);resampled=true;
      }
      state=estimate(particles,weights);
      cloud=snapshotCloud(particles);
      pfErrors.push({time:(i+1)*dt,error:distance(truth,state.mean)});
    }
    const next=makeFrame((i+1)*dt,truth,odometry,state,detected,update,prior,cloud,rays,neff,resampled);
    frames.push(next);odometryErrors.push({time:next.time,error:next.odometryError});
    if(!next.estimate.every(Number.isFinite)||!next.covariance.flat().every(Number.isFinite))throw new Error('Particle filter became unstable. Change the noise or sensor settings.');
  }
  const updates=frames.map((frame,index)=>frame.update?index:null).filter(index=>index!==null);
  const snapshotIndices=Array.from({length:4},(_,i)=>p.duration*(i+1)/4)
    .map(target=>updates.reduce((best,index)=>best===null||Math.abs(frames[index].time-target)<Math.abs(frames[best].time-target)?index:best,null))
    .filter((index,position,array)=>index!==null&&array.indexOf(index)===position);
  return {parameters:{...p,initialPose:[...initialPose],landmarks:landmarks.map(point=>[...point])},mode:p.mode,map,landmarks,frames,pfErrors,odometryErrors,snapshotIndices};
}

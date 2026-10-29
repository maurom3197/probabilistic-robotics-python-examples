export const EKF_LANDMARKS = [
  [5,12],[10.5,7.5],[16.5,15],[10,14],[5,6],
  [14.5,11.5],[14,9],[8,15.5],[13.5,17],[18.4,18],
];
const TAU=2*Math.PI;
const identity=()=>[[1,0,0],[0,1,0],[0,0,1]];
const transpose=a=>a[0].map((_,i)=>a.map(row=>row[i]));
const multiply=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((sum,v,k)=>sum+v*b[k][j],0)));
const add=(a,b)=>a.map((row,i)=>row.map((v,j)=>v+b[i][j]));
const subtract=(a,b)=>a.map((row,i)=>row.map((v,j)=>v-b[i][j]));
const diagonal=(a,b)=>[[a,0],[0,b]];
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
const distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
const sinc=x=>Math.abs(x)<1e-5?1-x*x/6+x*x*x*x/120:Math.sin(x)/x;
const sincDerivative=x=>Math.abs(x)<1e-4?-x/3+x*x*x/30:(x*Math.cos(x)-Math.sin(x))/(x*x);

function randomGenerator(seed) {
  let state=seed>>>0, spare=null;
  const uniform=()=>{
    state=(state+0x6D2B79F5)|0;
    let t=Math.imul(state^(state>>>15),1|state);
    t^=t+Math.imul(t^(t>>>7),61|t);
    return ((t^(t>>>14))>>>0)/4294967296;
  };
  return sigma=>{
    if(sigma===0) return 0;
    if(spare!==null){const z=spare;spare=null;return sigma*z;}
    const radius=Math.sqrt(-2*Math.log(Math.max(uniform(),1e-12)));
    const angle=TAU*uniform();
    spare=radius*Math.sin(angle);
    return sigma*radius*Math.cos(angle);
  };
}

function motion(pose,v,w,dt) {
  const turn=w*dt, half=turn/2, mid=pose[2]+half, travel=v*dt*sinc(half);
  return [pose[0]+travel*Math.cos(mid),pose[1]+travel*Math.sin(mid),pose[2]+turn];
}
function sampleMotion(pose,p,dt,normal) {
  const v=p.v+normal(p.sigmaV), w=p.omega+normal(p.sigmaOmega);
  const moved=motion(pose,v,w,dt);
  moved[2]+=normal(p.sigmaOmega/2)*dt;
  return moved;
}
function motionJacobians(pose,v,w,dt) {
  const half=w*dt/2, mid=pose[2]+half, s=sinc(half), ds=sincDerivative(half);
  const travel=v*dt*s, c=Math.cos(mid), sn=Math.sin(mid);
  const G=[[1,0,-travel*sn],[0,1,travel*c],[0,0,1]];
  const dw=v*dt*dt/2;
  const V=[[dt*s*c,dw*(ds*c-s*sn)],[dt*s*sn,dw*(ds*sn+s*c)],[0,dt]];
  return {G,V};
}
function predict(estimate,covariance,p,dt,normal) {
  // The Python RobotEKF.predict samples its mean before evaluating Jacobians.
  const mean=sampleMotion(estimate,p,dt,normal);
  const {G,V}=motionJacobians(mean,p.v,p.omega,dt);
  const M=diagonal(p.sigmaV**2,p.sigmaOmega**2);
  return {mean,covariance:add(multiply(multiply(G,covariance),transpose(G)),multiply(multiply(V,M),transpose(V)))};
}
function noisyMeasurement(pose,landmark,p,normal) {
  const dx=landmark[0]-pose[0],dy=landmark[1]-pose[1];
  return [Math.hypot(dx,dy)+normal(p.sigmaRange),Math.atan2(dy,dx)-pose[2]+normal(p.sigmaBearing*Math.PI/180)];
}
function correct(mean,covariance,z,landmark,p,normal) {
  const dx=landmark[0]-mean[0],dy=landmark[1]-mean[1],q=dx*dx+dy*dy;
  if(q<1e-10) return {mean,covariance};
  const r=Math.sqrt(q),H=[[-dx/r,-dy/r,0],[dy/q,-dx/q,-1]];
  const Q=diagonal(p.sigmaRange**2,(p.sigmaBearing*Math.PI/180)**2);
  const PHt=multiply(covariance,transpose(H)),S=add(multiply(H,PHt),Q);
  const determinant=S[0][0]*S[1][1]-S[0][1]*S[1][0];
  if(!(determinant>1e-15)) return {mean,covariance};
  const inverse=[[S[1][1]/determinant,-S[0][1]/determinant],[-S[1][0]/determinant,S[0][0]/determinant]];
  const K=multiply(PHt,inverse);
  // The source calls its noisy landmark model for both observed and expected z.
  const expected=noisyMeasurement(mean,landmark,p,normal);
  const residual=[z[0]-expected[0],wrap(z[1]-expected[1])];
  const updatedMean=mean.map((v,i)=>v+K[i][0]*residual[0]+K[i][1]*residual[1]);
  const I_KH=subtract(identity(),multiply(K,H));
  const joseph=add(multiply(multiply(I_KH,covariance),transpose(I_KH)),multiply(multiply(K,Q),transpose(K)));
  return {mean:updatedMean,covariance:joseph};
}
function frame(time,truth,odometry,estimate,covariance,detected,update,prior=null) {
  return {time,truth:[...truth],odometry:[...odometry],estimate:[...estimate],
    covariance:covariance.map(row=>[...row]),detected:[...detected],update,prior,
    ekfError:distance(truth,estimate),odometryError:distance(truth,odometry)};
}
export function runEkfSimulation(p) {
  if(!Number.isInteger(p.seed)||p.seed<0) throw new Error('Random seed must be a non-negative integer.');
  const landmarks=p.landmarks===undefined?EKF_LANDMARKS:p.landmarks;
  if(!Array.isArray(landmarks)||landmarks.length>20||!landmarks.every(point=>Array.isArray(point)&&point.length===2&&point.every(value=>Number.isFinite(value)&&value>=-20&&value<=20))) {
    throw new Error('The landmark map must contain up to 20 points with x and y between −20 and 20 m.');
  }
  const map=landmarks.map(point=>[...point]);
  const normal=randomGenerator(p.seed), simStep=.1, steps=Math.round(p.duration/simStep);
  const ekfStep=Math.round(p.ekfDt/simStep);
  let truth=[1,6,.3],odometry=[1,6,.3],estimate=[1,6,.3];
  let covariance=[[.1,0,0],[0,.1,0],[0,0,.1]];
  const frames=[frame(0,truth,odometry,estimate,covariance,[],false)];
  const ekfErrors=[{time:0,error:0}],odometryErrors=[{time:0,error:0}];
  for(let i=0;i<steps;i++){
    truth=sampleMotion(truth,p,simStep,normal);
    odometry=sampleMotion(odometry,p,simStep,normal);
    let detected=[],prior=null,update=false;
    if(i>0&&i%ekfStep===0){
      update=true;
      ({mean:estimate,covariance}=predict(estimate,covariance,p,p.ekfDt,normal));
      prior={mean:[...estimate],covariance:covariance.map(row=>[...row])};
      for(let j=0;j<map.length;j++){
        const z=noisyMeasurement(truth,map[j],p,normal);
        if(z[0]<=p.maxRange&&Math.abs(z[1])<=p.fov*Math.PI/360){
          ({mean:estimate,covariance}=correct(estimate,covariance,z,map[j],p,normal));
          detected.push(j);
        }
      }
      ekfErrors.push({time:(i+1)*simStep,error:distance(truth,estimate)});
    }
    const next=frame((i+1)*simStep,truth,odometry,estimate,covariance,detected,update,prior);
    frames.push(next);odometryErrors.push({time:next.time,error:next.odometryError});
    if(!next.estimate.every(Number.isFinite)||!covariance.flat().every(Number.isFinite)) throw new Error('EKF became unstable. Try lower noise or a different command.');
  }
  const updateIndices=frames.map((f,index)=>f.update?index:null).filter(index=>index!==null);
  const snapshotIndices=[p.duration/3,2*p.duration/3,p.duration]
    .map(target=>updateIndices.reduce((best,index)=>best===null||Math.abs(frames[index].time-target)<Math.abs(frames[best].time-target)?index:best,null))
    .filter((index,position,array)=>index!==null&&array.indexOf(index)===position);
  return {parameters:{...p,landmarks:map.map(point=>[...point])},landmarks:map,frames,ekfErrors,odometryErrors,snapshotIndices};
}

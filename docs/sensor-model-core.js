// Numerical adapters of the landmark and likelihood-field examples on main.
export const DEFAULT_LANDMARKS = [
  [5, 2], [-2.5, 3], [3, 1.5], [4, -1], [-2, -2], [2, -1], [-4, -2],
  [-3, 1], [5, 5], [-5, 5], [-5, -5], [0, -5], [0, 5],
];
export const MAP_FILES = ['map0.png', 'map1.png', 'map2.png', 'map3.png', 'map31.png', 'map31_grey.png', 'map4.png'];
const TAU = 2*Math.PI;

export function makeRandom(seed) {
  let state = seed >>> 0, spare = null;
  const uniform = () => {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
  const normal = sigma => {
    if (sigma === 0) return 0;
    if (spare !== null) { const value = spare; spare = null; return sigma*value; }
    const radius = Math.sqrt(-2*Math.log(Math.max(uniform(), Number.MIN_VALUE)));
    const angle = TAU*uniform();
    spare = radius*Math.sin(angle);
    return sigma*radius*Math.cos(angle);
  };
  return {uniform, normal};
}

export function normalPdf(error, sigma) {
  return Math.exp(-.5*(error/sigma)**2)/(Math.sqrt(TAU)*sigma);
}

export function runLandmarkModel(p) {
  const rng = makeRandom(p.seed), [x, y, theta] = p.pose;
  const detections = p.landmarks.map(([mx, my], index) => {
    const expectedRange = Math.hypot(mx-x, my-y);
    const expectedBearing = Math.atan2(my-y, mx-x)-theta;
    const range = expectedRange+rng.normal(.05);
    const bearing = expectedBearing+rng.normal(3*Math.PI/180);
    if (range > p.maxRange || Math.abs(bearing) > p.fov/2) return null;
    const probability = normalPdf(range-expectedRange, p.sigmaRange)*normalPdf(bearing-expectedBearing, p.sigmaBearing);
    return {index, range, bearing, probability, endpoint: [x+range*Math.cos(theta+bearing), y+range*Math.sin(theta+bearing)]};
  }).filter(Boolean);
  const samples = [];
  let sampledLandmark = null;
  if (detections.length) {
    // The example checks for a detection, then samples poses from landmarks[0].
    sampledLandmark = 0;
    const [mx, my] = p.landmarks[sampledLandmark];
    const range = Math.hypot(mx-x, my-y)+rng.normal(p.sigmaRange);
    const bearing = Math.atan2(my-y, mx-x)-theta+rng.normal(p.sigmaBearing);
    for (let i = 0; i < 500; i++) {
      const gamma = TAU*rng.uniform();
      const sampledRange = range+rng.normal(p.sigmaRange);
      const sampledBearing = bearing+rng.normal(p.sigmaBearing);
      samples.push([mx+sampledRange*Math.cos(gamma), my+sampledRange*Math.sin(gamma), gamma-Math.PI-sampledBearing]);
    }
  }
  return {parameters: p, detections, samples, sampledLandmark};
}

// Equivalent to get_map(..., xy_reso=4): grayscale and 4x4 max pooling.
export async function loadOccupancyMap(filename) {
  if (!MAP_FILES.includes(filename)) throw new Error('Unknown map file');
  const image = new Image();
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error(`Could not load ${filename}`));
    image.src = new URL(`./maps/${filename}`, import.meta.url).href;
  });
  const canvas = document.createElement('canvas'); canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
  const context = canvas.getContext('2d', {willReadFrequently: true}); context.drawImage(image, 0, 0);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const rows = Math.ceil(canvas.height/4), cols = Math.ceil(canvas.width/4);
  const cells = Array.from({length: rows}, () => Array(cols).fill(0));
  for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
    let highest = 0;
    for (let dy = 0; dy < 4; dy++) for (let dx = 0; dx < 4; dx++) {
      const py = row*4+dy, px = col*4+dx;
      if (py >= canvas.height || px >= canvas.width) continue;
      const i = 4*(py*canvas.width+px);
      const gray = Math.round(.299*pixels[i]+.587*pixels[i+1]+.114*pixels[i+2]);
      highest = Math.max(highest, gray);
    }
    cells[row][col] = highest === 255 ? 1 : highest === 128 ? -1 : 0;
  }
  return {filename, rows, cols, cells};
}

export function nearestFreeCell(map, x=8, y=11) {
  let best = null, distance = Infinity;
  for (let row = 0; row < map.rows; row++) for (let col = 0; col < map.cols; col++) {
    if (map.cells[row][col] !== 0) continue;
    const d = (row-x)**2+(col-y)**2;
    if (d < distance) { best = [row+.5, col+.5]; distance = d; }
  }
  return best;
}

function castRays(map, pose, numRays=12, fov=Math.PI/4, zMax=8) {
  const [x, y, theta] = pose, rays = [];
  for (let i = 0; i < numRays; i++) {
    const angle = theta-fov/2+i*fov/numRays;
    let range = zMax, endpoint = [x+zMax*Math.cos(angle), y+zMax*Math.sin(angle)];
    for (let length = .25; length <= zMax+.11; length += .1) {
      const px = x+length*Math.cos(angle), py = y+length*Math.sin(angle);
      if (px < 0 || px >= map.rows || py < 0 || py >= map.cols) break;
      if (length >= zMax) { endpoint = [px, py]; break; }
      if (map.cells[Math.trunc(px)][Math.trunc(py)] === 1) { range = Math.hypot(px-x, py-y); endpoint = [px, py]; break; }
    }
    rays.push({angle, expectedRange: range, expectedEndpoint: endpoint});
  }
  return rays;
}

function nearestObstacleDistance(point, obstacles) {
  const row = Math.trunc(point[0]), col = Math.trunc(point[1]);
  let best = Infinity;
  for (const [orow, ocol] of obstacles) best = Math.min(best, Math.hypot(orow-row, ocol-col));
  return best;
}

export function runLikelihoodField(p, map) {
  const [x, y] = p.pose;
  const fov = p.fov ?? Math.PI/4, numRays = p.numRays ?? 12, zMax = p.maxRange ?? 8;
  if (x < 0 || x >= map.rows || y < 0 || y >= map.cols || map.cells[Math.trunc(x)][Math.trunc(y)] !== 0) {
    throw new Error('Place the robot in a free map cell');
  }
  const obstacles = [];
  for (let row = 0; row < map.rows; row++) for (let col = 0; col < map.cols; col++) {
    if (map.cells[row][col] === 1) obstacles.push([row, col]);
  }
  if (!obstacles.length) throw new Error('The selected map has no occupied cells');
  const totalWeight = p.weights.reduce((a, b) => a+b, 0);
  if (totalWeight <= 0) throw new Error('Enter at least one positive mixture weight');
  const weights = p.weights.map(w => w/totalWeight), rng = makeRandom(p.seed);
  const rays = castRays(map, p.pose, numRays, fov, zMax);
  // The main example adds one shared noise draw and rare binomial terms to all ranges.
  const sharedNoise = rng.normal(.01)
    + Number(rng.uniform() < .001)+Number(rng.uniform() < .001)
    + 10*(Number(rng.uniform() < .001)+Number(rng.uniform() < .001));
  for (const ray of rays) {
    ray.range = Math.max(0, Math.min(zMax, ray.expectedRange+sharedNoise));
    ray.endpoint = [x+ray.range*Math.cos(ray.angle), y+ray.range*Math.sin(ray.angle)];
    ray.distance = nearestObstacleDistance(ray.endpoint, obstacles);
    ray.probability = weights[0]*Math.exp(-.5*(ray.distance/p.sigmaHit)**2)
      + weights[1]*(ray.range === zMax ? 1 : 0) + weights[2]/zMax;
  }
  const distanceGrid = map.cells.map((row, r) => row.map((_, c) => nearestObstacleDistance([r, c], obstacles)));
  const maxDistance = Math.max(...distanceGrid.flat());
  const field = distanceGrid.map((row, r) => row.map((d, c) => map.cells[r][c] === -1 ? null : Math.exp(-.5*(d/p.sigmaHit)**2)));
  return {parameters: p, map, rays, weights, field, maxDistance};
}

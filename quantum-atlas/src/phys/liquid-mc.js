/* liquid-mc.js — 液态的 Metropolis Monte Carlo（Phase 4 第三步）
 *
 * 把 RSA 硬球升级成"真实对势 + 重要性采样"：
 *   势：Lennard-Jones 12-6，截断 rc=2.5σ 并平移（U(rc)=0）
 *   采样：Metropolis（局部移动 + 周期边界），得到玻尔兹曼分布的平衡构型
 *
 * 单位：σ=1、ε=1，T 为约化温度 kT/ε。构型形状与 liquid.js 的 liquidConfig 同构
 * （{pos, L, N, rho}），可直接喂给 pairCorrelation/structureFactor。
 */

const { sqrt, cbrt, exp } = Math;

const wrap = (x, L) => ((x % L) + L) % L;
const minDist = (a, b, L) => {
  let dx = Math.abs(a[0] - b[0]); dx = Math.min(dx, L - dx);
  let dy = Math.abs(a[1] - b[1]); dy = Math.min(dy, L - dy);
  let dz = Math.abs(a[2] - b[2]); dz = Math.min(dz, L - dz);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
};

export function ljPair(r, { sigma = 1, epsilon = 1 } = {}) {
  const s = sigma / r;
  const s6 = s * s * s * s * s * s;
  return 4 * epsilon * (s6 * s6 - s6);
}

function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/* 单粒子能量（含截断平移） */
function particleEnergy(pos, L, i, { sigma = 1, epsilon = 1, rc = 2.5 * sigma } = {}) {
  const shift = ljPair(rc, { sigma, epsilon });
  let E = 0;
  for (let j = 0; j < pos.length; j++) {
    if (j === i) continue;
    const r = minDist(pos[i], pos[j], L);
    if (r < rc) E += ljPair(r, { sigma, epsilon }) - shift;
  }
  return E;
}

function initPos(N, L, sigma) {
  const dmin = sigma * 0.85;
  const rnd = Math.random;
  const pos = [];
  let guard = 0;
  while (pos.length < N && guard < 400000) {
    guard++;
    const p = [rnd() * L, rnd() * L, rnd() * L];
    let good = true;
    for (const q of pos) if (minDist(p, q, L) < dmin) { good = false; break; }
    if (good) pos.push(p);
  }
  return pos;
}

export function metropolisConfig({
  N = 150, rho = 0.6, sigma = 1, epsilon = 1, T = 1.2,
  seed = 7, steps = 20000, delta = 0.1, rc = 2.5 * sigma,
} = {}) {
  const L = cbrt(N / rho);
  const beta = 1 / T;                 // T = kT/ε
  const rnd = mulberry32(seed);
  const pos = initPos(N, L, sigma);

  let E = 0;
  for (let i = 0; i < N; i++) E += particleEnergy(pos, L, i, { sigma, epsilon, rc });
  E /= 2;

  let acc = 0;
  for (let m = 0; m < steps; m++) {
    const i = (rnd() * N) | 0;
    const oldE = particleEnergy(pos, L, i, { sigma, epsilon, rc });
    const cand = [
      wrap(pos[i][0] + (rnd() * 2 - 1) * delta, L),
      wrap(pos[i][1] + (rnd() * 2 - 1) * delta, L),
      wrap(pos[i][2] + (rnd() * 2 - 1) * delta, L),
    ];
    const saved = pos[i];
    pos[i] = cand;
    const newE = particleEnergy(pos, L, i, { sigma, epsilon, rc });
    const dE = newE - oldE;
    if (dE <= 0 || rnd() < exp(-beta * dE)) { E += dE; acc++; }
    else pos[i] = saved;
  }

  return { pos, L, N: pos.length, rho, sigma, epsilon, T, E, accRate: acc / steps, method: "mc" };
}

/* 初始能量（用于"平衡使能量下降"的断言） */
export function initialEnergy({ N = 150, rho = 0.6, sigma = 1, epsilon = 1, seed = 7, rc = 2.5 * sigma } = {}) {
  const L = cbrt(N / rho);
  const pos = initPos(N, L, sigma);
  let E = 0;
  for (let i = 0; i < N; i++) E += particleEnergy(pos, L, i, { sigma, epsilon, rc });
  return { E: E / 2, pos, L };
}

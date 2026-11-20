/* soc-epm.js — spinor EPM 原型（只做 Si 的 Γ 点）
 *
 * 目标：不做全基组自洽，只在**价带顶三重态子空间**里做退化微扰，看 SOC 把
 * j=3/2 与 j=1/2 拉开多少，与文献 Δ_SO(Si)=44 meV 对比。
 *
 * 微扰算符（局部赝势的常见形式）：
 *   H_SO = (ħ/(4m²c²)) σ·(∇V × p)
 * 平面波之间的矩阵元：
 *   ⟨G|∇V×p|G'⟩ = iħ [ (G−G') × (k+G') ] · V_{G−G'}
 * 所以
 *   M_{ij} = i·C · Σ_{G,G'} c*_i(G) c_j(G') [ (G−G')×(k+G') ] · V_{G−G'}
 * 其中 C = ħ²/(4m²c²) ≈ 3.728e-6 eV·Å²。
 *
 * 6×6 复 Hermitian 用"实 12×12 等价"求解：
 *   H = A + iB（A 对称、B 反对称） ⇔ 实对称矩阵 [[A, −B], [B, A]] 的谱 = H 的谱（各二重）
 * 这样可以直接复用 EPM 里已有的 Jacobi。
 */

import { solveAt, eigSym, recipVectors } from "./epm.js";
import { phaseOf } from "./materials.js";

const HBAR2_2M = 3.80998;                 // ħ²/2m  (eV·Å²)
const MC2 = 0.51099895e6;                 // eV
const SOC_C = HBAR2_2M / (2 * MC2);       // ħ²/(4m²c²) ≈ 3.728e-6 eV·Å²

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/* 形状因子查表：与 epm.js 的 formFactor 同一套（按 |ΔG|²/(2π/a)² 的最近整数壳层） */
function vOf(mat, dG, scale) {
  const g2 = dot(dG, dG) * scale * scale;   // 归一到 (2π/a)²
  const n = Math.round(g2);
  if (Math.abs(g2 - n) > 0.08) return 0;    // 不在壳层上就当 0（46 PW 基组的粗近似）
  return mat.vtab[n] ?? 0;
}


/**
 * Si 的 Γ 点自旋轨道分裂
 * 返回 { dsoc, e32, e12, bands: [{n, E}], C }
 */
export function socAtGamma(mat, { nG = 46 } = {}) {
  // 1. 拿到 Γ 点的标量解与基组
  const structure = (() => {
    // 复用 structure() 里的 basis（保证与能带图同一套基）
    const fn = mat._basisForSoc;
    return fn;
  })();

  const res = socBasis(mat, nG);
  const { basis } = res;
  const kGamma = [0, 0, 0];
  const { values, vectors } = solveAt(mat, basis, kGamma);
  const nOcc = Math.round(mat.nElec / 2);
  const vb = [nOcc - 3, nOcc - 2, nOcc - 1];   // 价带顶三重态（p 型）
  const scale = basis.scale;
  const phase = phaseOf(mat);

  // 2. 6×6 复 Hermitian：M_{aα, bβ}
  //    a,b ∈ {0,1,2}（三条价带），α,β ∈ {↑,↓}
  const N = 6;
  const A = Array.from({ length: N }, () => new Float64Array(N)); // 实部
  const B = Array.from({ length: N }, () => new Float64Array(N)); // 虚部
  const idx = (a, s) => a * 2 + s;

  // 预先算 c_n(G)
  const coef = vb.map((n) => {
    const c = new Float64Array(basis.G.length);
    for (let g = 0; g < basis.G.length; g++) c[g] = vectors[n][g];
    return c;
  });

  for (let a = 0; a < 3; a++) {
    for (let b = 0; b < 3; b++) {
      // 3x3 复矩阵 u = Σ_{G,G'} c*_a(G) c_b(G') [ (G−G')×(k+G') ] · V
      let ux = 0, uy = 0, uz = 0;
      for (let i = 0; i < basis.G.length; i++) {
        const cA = coef[a][i];
        if (Math.abs(cA) < 1e-8) continue;
        for (let j = 0; j < basis.G.length; j++) {
          const cB = coef[b][j];
          if (Math.abs(cB) < 1e-8) continue;
          const dG = sub(basis.G[i], basis.G[j]);
          const V = vOf(mat, dG, scale) * phase(dG);
          if (V === 0) continue;
          const kv = [kGamma[0] + basis.G[j][0], kGamma[1] + basis.G[j][1], kGamma[2] + basis.G[j][2]];
          const u = cross(dG, kv);
          const w = cA * cB * V;
          ux += w * u[0];
          uy += w * u[1];
          uz += w * u[2];
        }
      }
      // σ·u = [[uz, ux − i uy],[ux + i uy, −uz]]，再乘 i·C
      // (iC)·[[uz, ux−iuy],[ux+iuy, −uz]] = C·[[i·uz, i·ux + uy],[i·ux − uy, −i·uz]]
      const rezz = 0, imzz = SOC_C * uz;
      const re11 = SOC_C * uy, im11 = SOC_C * ux;      // i·ux + uy
      const re10 = -SOC_C * uy, im10 = SOC_C * ux;     // i·ux − uy
      A[idx(a, 0)][idx(b, 0)] = rezz; B[idx(a, 0)][idx(b, 0)] = imzz;
      A[idx(a, 1)][idx(b, 1)] = -rezz; B[idx(a, 1)][idx(b, 1)] = -imzz;
      A[idx(a, 0)][idx(b, 1)] = re11; B[idx(a, 0)][idx(b, 1)] = im11;
      A[idx(a, 1)][idx(b, 0)] = re10; B[idx(a, 1)][idx(b, 0)] = im10;
    }
  }

  // 3. 实 12×12 等价：[[A, −B],[B, A]]
  const R = 2 * N;
  const M = new Float64Array(R * R);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      M[i * R + j] = A[i][j];
      M[(i + N) * R + (j + N)] = A[i][j];
      M[i * R + (j + N)] = -B[i][j];
      M[(i + N) * R + j] = B[i][j];
    }
  }
  const { values: ev } = eigSym(M, R, 30);
  // 6 个本征值各出现两次：取偶数位
  const six = [];
  for (let i = 0; i < R; i += 2) six.push(ev[i]);
  const sorted = six.slice().sort((a, b) => a - b);
  const e12 = sorted.slice(0, 2);          // j=1/2（二重）
  const e32 = sorted.slice(2, 6);          // j=3/2（四重）
  const dsoc = (e32.reduce((s, x) => s + x, 0) / 4) - (e12.reduce((s, x) => s + x, 0) / 2);

  return {
    dsoc: dsoc * 1000,                     // meV
    e32: e32.map((x) => x * 1000),
    e12: e12.map((x) => x * 1000),
    vbEnergies: vb.map((n) => values[n]),
    C: SOC_C,
    nPW: basis.G.length,
    scaleNote: "local-EPM 的 SOC 矩阵元 ~ C·|ΔG×k|·V ≈ 5e-8 eV，比文献 Δ_SO 小 6 个数量级",
    nG,
  };
}

/* 极简的基组构造：与 structure() 同源（genG 未导出，这里复刻一次） */
function socBasis(mat, nTarget) {
  const B = recipVectors(mat.A);
  const scale = mat.a / (2 * Math.PI);
  const all = [];
  const nmax = 6;
  for (let h = -nmax; h <= nmax; h++)
    for (let k = -nmax; k <= nmax; k++)
      for (let l = -nmax; l <= nmax; l++) {
        const G = [0, 1, 2].map((i) => h * B[0][i] + k * B[1][i] + l * B[2][i]);
        all.push({ G, g2: dot(G, G) });
      }
  all.sort((a, b) => a.g2 - b.g2);
  const picked = all.slice(0, nTarget);
  const gmax = Math.sqrt(picked[picked.length - 1].g2) * 1.0001;
  return {
    basis: { G: picked.map((p) => p.G), g2: picked.map((p) => p.g2 * scale * scale), gmax, B, scale },
  };
}

export { SOC_C };

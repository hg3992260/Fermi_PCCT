/* ─────────────────────────────────────────────────────────────
   bloch.js — Bloch 三重身份
   ① 定理：ψ_nk = e^{ikx} u_nk（周期性晶格上的波函数）
   ② 球：二能级纯态的 CP¹ 表示
   ③ 方程：dM/dt = γ M×B − (M−M₀)/T₁,₂（含 Rabi 驱动）
   ───────────────────────────────────────────────────────────── */

const { sin, cos, exp, sqrt, PI, hypot } = Math;

/* ── ① Bloch 定理：一维周期势中的波函数 ───────────────── */
export function blochWave({ N = 8, k = 0.6, n = 220, V = 0.8 } = {}) {
  // 近自由电子：ψ ∝ e^{ikx}，|ψ|² 受周期势调制（示意）
  const xs = [], re = [], mod = [];
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * N;
    const u = 1 + 0.28 * cos(2 * PI * x); // 周期性包络 |u|²
    const psi = cos(k * x) * Math.sqrt(u);
    xs.push(x);
    re.push(psi);
    mod.push(u);
  }
  return { xs, re, mod, k, N };
}

/* ── ② Bloch 球：球面坐标 <-> 密度矩阵 ─────────────────── */
export function sphereToState(theta, phi) {
  // |ψ> = cos(θ/2)|0> + e^{iφ} sin(θ/2)|1>
  return [cos(theta / 2), sin(theta / 2) * cos(phi), sin(theta / 2) * sin(phi)];
}

export function stateToBloch(a, b) {
  // a,b 复振幅
  const rho = [a[0], a[1], b[0], b[1]];
  const x = 2 * (rho[0] * rho[2] + rho[1] * rho[3]);
  const y = 2 * (rho[0] * rho[3] - rho[1] * rho[2]);
  const z = rho[0] * rho[0] + rho[1] * rho[1] - rho[2] * rho[2] - rho[3] * rho[3];
  return [x, y, z];
}

/* ── ③ Bloch 方程（RK4） ───────────────────────────────── */
/**
 * M = [Mx, My, Mz]（以 M0=1 归一化）
 * B(t) = [B1 cos(ωt), 0, B0]
 * dM/dt = γ M×B − (Mx/T2, My/T2, (Mz−1)/T1)
 */
export function integrateBloch({
  B0 = 0.4, B1 = 0.25, omega = 1.2, gamma = 2 * PI, T1 = 6, T2 = 2.2,
  M0 = [0, 0, 1], tMax = 12, dt = 0.012,
} = {}) {
  const f = (M, t) => {
    const Bx = B1 * cos(omega * t), Bz = B0;
    const cx = M[1] * Bz - M[2] * 0;
    const cy = M[2] * Bx - M[0] * Bz;
    const cz = M[0] * 0 - M[1] * Bx;
    return [
      gamma * cx - M[0] / T2,
      gamma * cy - M[1] / T2,
      gamma * cz - (M[2] - 1) / T1,
    ];
  };
  let M = [...M0];
  const path = [{ t: 0, M: [...M] }];
  const n = Math.floor(tMax / dt);
  for (let i = 0; i < n; i++) {
    const t = i * dt;
    const k1 = f(M, t);
    const k2 = f(M.map((v, j) => v + 0.5 * dt * k1[j]), t + dt / 2);
    const k3 = f(M.map((v, j) => v + 0.5 * dt * k2[j]), t + dt / 2);
    const k4 = f(M.map((v, j) => v + dt * k3[j]), t + dt);
    M = M.map((v, j) => v + (dt / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
    if (i % 2 === 0) path.push({ t: t + dt, M: [...M] });
  }
  return path;
}

/** 旋转坐标系下的有效场：Δ = ω_drive − ω_0 */
export function resonance({ B0 = 0.4, gamma = 2 * PI, omega = 1.2 }) {
  const w0 = gamma * B0;
  return { w0, detuning: omega - w0, rabi: gamma * 0.25 };
}

/** 光学 Bloch 方程的稳态激发概率（二能级，含失谐） */
export function steadyExcitation({ Omega = 1, detuning = 0, T1 = 1, T2 = 0.5 }) {
  const s = (Omega * Omega * T1 * T2) / (1 + detuning * detuning * T2 * T2 + Omega * Omega * T1 * T2);
  return s / (1 + s);
}

/** 拉比振荡的解析包络（共振） */
export function rabiOscillation(t, Omega, T2) {
  return 0.5 * (1 - cos(Omega * t)) * exp(-t / T2);
}

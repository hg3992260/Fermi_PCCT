// 物理内核自检：特征分解、EPM 能带、布里渊区几何
import { eigSym, structure, brillouinZone, recipVectors } from "../src/phys/epm.js";
import { materialOf, MATERIALS } from "../src/phys/materials.js";

let fail = 0;
const ok = (cond, msg) => { console.log(`${cond ? "  ok " : "FAIL "} ${msg}`); if (!cond) fail++; };
const near = (a, b, tol, msg) => ok(Math.abs(a - b) <= tol, `${msg}  (${a.toFixed(4)} vs ${b.toFixed(4)} ±${tol})`);

/* 1. 实对称特征分解：与解析解对照 */
{
  const t = 1.3;
  const A = [2, t, 0, t, 2, t, 0, t, 2];
  const { values } = eigSym(Float64Array.from(A), 3, 12);
  near(values[0], 2 - t * Math.SQRT2, 1e-8, "Jacobi 最小本征值");
  near(values[1], 2, 1e-8, "Jacobi 中间本征值");
  near(values[2], 2 + t * Math.SQRT2, 1e-8, "Jacobi 最大本征值");
}

/* 2. 倒格矢：fcc 实空间 → bcc 倒空间，|b| = √3·(2π/a) */
{
  const Si = materialOf("Si");
  const B = recipVectors(Si.A);
  near(Math.hypot(...B[0]) * Si.a / (2 * Math.PI), Math.sqrt(3), 1e-6, "fcc 倒格矢 |b| = √3·2π/a");
}

/* 3. 各材料能带 */
const expect = { Si: [1.0, 1.3], Ge: [0.5, 0.8], C: [5.0, 5.9] };
for (const id of ["Si", "Ge", "C", "Na", "graphene"]) {
  const mat = materialOf(id);
  const t0 = performance.now();
  const res = structure(mat, { nG: 46, mesh: 6 });
  const ms = (performance.now() - t0).toFixed(0);
  console.log(`\n[${id}] ${res.nBands} 带 · ${res.pts.length} k · ${ms} ms · nOcc=${res.nOcc} · metal=${res.isMetal}`);
  console.log(`      gap=${res.gap.toFixed(3)} (${res.gapKind})  direct=${res.directGap.toFixed(3)}  E_F=${(res.EF - res.zero).toFixed(3)} eV`);
  ok(res.nBands >= (id === "graphene" ? 2 : 12), `${id}: 带数足够`);
  ok(Number.isFinite(res.gap), `${id}: 带隙有限`);
  if (expect[id]) {
    const [lo, hi] = expect[id];
    ok(res.gap > lo && res.gap < hi, `${id}: 带隙在模型合理区间（${res.gap.toFixed(2)} eV，实验 ${mat.gapExp}）`);
    ok(res.directGap > res.gap, `${id}: 直接带隙 > 间接带隙`);
  }
  if (id === "Na") {
    ok(res.isMetal, "Na 判为金属");
    const b0 = res.E.map((r) => r[0] - res.zero);
    ok(Math.min(...b0) < 0 && Math.max(...b0) > 0, "Na 费米能横穿 s 带");
  }
  if (id === "graphene") {
    ok(Math.abs(res.gap) < 1e-9, "石墨烯零带隙（狄拉克点）");
    ok(res.nBands === 2, "石墨烯 2 条 π 带");
  }
}

/* 4. 布里渊区几何 */
for (const id of ["Si", "Na"]) {
  const mat = materialOf(id);
  const bz = brillouinZone(mat);
  const deg = bz.faces.map((f) => f.verts.length).join(",");
  console.log(`\n[${id}] BZ: ${bz.faces.length} 面 / ${bz.verts.length} 顶点 (${deg})`);
  ok(bz.faces.length >= 6, `${id}: BZ 至少 6 个面`);
  const r = Math.max(...bz.verts.map((v) => Math.hypot(...v)));
  if (id === "Si") near(r / (Math.PI / mat.a), Math.sqrt(5), 0.02, "Si BZ 顶点半径 = √5·π/a");
}

/* 5. 能量零点 */
{
  const res = structure(materialOf("Si"), { nG: 46, mesh: 6 });
  const maxOcc = Math.max(...res.E.map((row) => row[res.nOcc - 1]));
  near(maxOcc - res.zero, 0, 1e-9, "Si 价带顶对齐到 0");
}

/* ── 温度：刚性离子 vs Varshni ── */
const th = await import("../src/phys/thermal.js");
const matOf = (id) => MATERIALS[id] || materialOf(id);
for (const id of ["Si", "Ge", "C"]) {
  const mat = matOf(id);
  if (!mat) continue;
  const g300 = th.gapRigid(mat, 300);
  const g500 = th.gapRigid(mat, 500);
  const g800 = th.gapRigid(mat, 800);
  const d500 = g500 - g300;
  const v300 = th.varshniGap(mat, 300);
  const v500 = th.varshniGap(mat, 500);
  ok(g800 < g500 && g500 < g300, `${id}: 刚性带隙随温度单调下降 (${g300.toFixed(3)}→${g500.toFixed(3)}→${g800.toFixed(3)})`);
  ok(d500 < -0.003 && d500 > -0.06, `${id}: 300→500 K 刚性漂移 ${(d500 * 1000).toFixed(0)} meV 落在 (−60, −3)`);
  if (v500 != null) {
    const ratio = d500 / (v500 - v300);
    ok(ratio > 0.1 && ratio < 0.9, `${id}: 刚性部分占 Varshni 的 ${(ratio * 100).toFixed(0)}%（0–90%，说明还缺电声自能）`);
  }
}

/* ── 激子：Wannier–Mott ── */
const ex = await import("../src/phys/exciton.js");
const EPS = { Si: 11.7, Ge: 16.0, C: 5.7 };
for (const id of ["Si", "Ge", "C"]) {
  const w = ex.wannier(matOf(id), { eps: EPS[id] });
  if (!w) { ok(false, `${id}: 激子求解返回空`); continue; }
  ok(w.me > 0.05 && w.me < 1.5, `${id}: 电子有效质量 ${w.me.toFixed(3)} m0 落在 0.05–1.5`);
  ok(w.mh > 0.05 && w.mh < 2.5, `${id}: 空穴有效质量 ${w.mh.toFixed(3)} m0 落在 0.05–2.5`);
  ok(w.eb > 0.002 && w.eb < 0.2, `${id}: 束缚能 ${(w.eb * 1000).toFixed(1)} meV 落在 2–200`);
  ok(w.aB > 5 && w.aB < 200, `${id}: 玻尔半径 ${w.aB.toFixed(0)} A 落在 5–200`);
}
const wSi = ex.wannier(matOf("Si"), { eps: 11.7 });
ok(Math.abs(wSi.eb * 1000 - 14.7) < 5, `Si 激子束缚能 ${(wSi.eb * 1000).toFixed(1)} meV 与文献 14.7 差 <5`);
const wBig = ex.wannier(matOf("Si"), { eps: 23.4 });
ok(wBig.eb < wSi.eb, `介电常数加倍 → 束缚能下降 (${(wBig.eb * 1000).toFixed(1)} < ${(wSi.eb * 1000).toFixed(1)} meV)`);

/* ── 量子噪声 ── */
const qn = await import("../src/phys/quantum-noise.js");
{
  const ops = [{ gate: "H", qubit: 0, step: 0 }, { gate: "CNOT", control: 0, qubit: 1, step: 1 }];
  const ideal = qn.simulateNoisy(2, ops, 3, { T1: 1e9, T2: 1e9, gateTime: 1e-4 });
  ok(ideal[3].F > 0.9999, `无噪声保真度 ${ideal[3].F.toFixed(5)} ≈ 1`);
  ok(Math.abs(ideal[3].concurrence - 1) < 1e-3, `无噪声 Bell 态 concurrence ${ideal[3].concurrence.toFixed(3)} ≈ 1`);

  const fast = qn.simulateNoisy(2, ops, 3, { T1: 6, T2: 2.2, gateTime: 0.02 });
  const slow = qn.simulateNoisy(2, ops, 3, { T1: 6, T2: 2.2, gateTime: 0.4 });
  ok(fast[3].F < 1 && fast[3].F > 0.9, `20 ns 门：F=${fast[3].F.toFixed(4)} 落在 (0.9, 1)`);
  ok(slow[3].F < fast[3].F, `400 ns 门保真度更低 (${slow[3].F.toFixed(3)} < ${fast[3].F.toFixed(3)})`);
  ok(slow[3].purity < fast[3].purity && slow[3].purity > 0, `纯度随之下降 (${slow[3].purity.toFixed(3)} < ${fast[3].purity.toFixed(3)})`);
  ok(slow[3].concurrence < fast[3].concurrence, `纠缠度随之下降 (${slow[3].concurrence.toFixed(3)} < ${fast[3].concurrence.toFixed(3)})`);
  ok(slow[3].F >= 0.25 - 1e-9, `混态极限下 F 不低于 0.25（完全混合态下限）`);
}

/* ── 磁振子：动力学矩阵 vs 解析式 ── */
const mg = await import("../src/phys/magnon.js");
const spOld = await import("../src/phys/spin.js");
const magnon = spOld.magnon;
{
  const oldFM = magnon({ kind: "FM", J: 1, S: 1, D: 0.12, n: 60 });
  const newFM = mg.branches({ n: 1, kind: "FM", S: 1, J1: 1, J2: 0, D: 0.12 }, 60);
  let maxd = 0;
  newFM.w[0].forEach((w, i) => { maxd = Math.max(maxd, Math.abs(w - oldFM.w[i])); });
  ok(maxd < 1e-6, `FM 单子晶格：对角化与解析式逐点一致（最大差 ${maxd.toExponential(1)}）`);

  const oldAF = magnon({ kind: "AF", J: 1, S: 1, D: 0, n: 60 });
  const newAF = mg.branches({ n: 2, kind: "AF", S: 1, J1: 1, J2: 0, D: 0 }, 60);
  const dAF = Math.abs(Math.max(...newAF.w[0]) - Math.max(...oldAF.w));
  ok(dAF < 1e-6, `AF 双子晶格：$\sqrt{(A-B)(A+B)}$ 复现解析带宽（差 ${dAF.toExponential(1)}）`);

  const xOf = (J2) => {
    const b = mg.branches({ n: 1, kind: "FM", S: 1, J1: 1, J2, D: 0 }, 60);
    return b.w[0][Math.round(b.w[0].length / 3)];
  };
  ok(xOf(0.4) > xOf(0) && xOf(-0.4) < xOf(0), `J₂ 改变 ω(X)：${xOf(-0.4).toFixed(2)} < ${xOf(0).toFixed(2)} < ${xOf(0.4).toFixed(2)} meV`);

  const gap0 = mg.branches({ n: 1, kind: "FM", S: 1, J1: 1, J2: 0, D: 0 }, 20).w[0][0];
  const gap2 = mg.branches({ n: 1, kind: "FM", S: 1, J1: 1, J2: 0, D: 0.3 }, 20).w[0][0];
  ok(Math.abs(gap0) < 1e-9, `无各向异性时 Γ 点无能隙（${gap0.toFixed(6)}）`);
  ok(Math.abs(gap2 - 2 * 0.3) < 1e-9, `各向异性打开能隙 ω(Γ)=2DS=${(2 * 0.3).toFixed(2)}（算得 ${gap2.toFixed(3)}）`);
}

/* ── 自旋分裂：Zeeman 与 SOC ── */
const sb = await import("../src/phys/spinbands.js");
{
  const z1 = sb.zeemanAt(1);
  ok(Math.abs(z1 - 0.11577) < 1e-3, `1 T 的 Zeeman 分裂 ${z1.toFixed(4)} meV ≈ gμ_B·B`);
  const z3 = sb.splittings({ B: 3, Dex: 0, soc: 0 });
  ok(Math.abs(z3.zeeman - 3 * z1) < 1e-9, `Zeeman 与 B 线性（3 T → ${z3.zeeman.toFixed(4)} meV）`);
  const withEx = sb.splittings({ B: 3, Dex: 20, soc: 44 });
  ok(withEx.total > withEx.zeeman, `交换劈裂叠加在 Zeeman 上 (${withEx.total.toFixed(3)} > ${withEx.zeeman.toFixed(3)})`);
  ok(Math.abs(withEx.socSplitOff - (44 * 2) / 3) < 1e-9, `SOC split-off = 2/3·Δ_SO = ${withEx.socSplitOff.toFixed(1)} meV`);
  const p0 = sb.spinPolarization({ B: 0 });
  const p3 = sb.spinPolarization({ B: 3 });
  const pBig = sb.spinPolarization({ B: 3, Dex: 20 });
  ok(p0 === 0 && p3 > 0 && p3 < 0.05, `300 K 下 B=3 T 极化率很小 (${p3.toFixed(4)})`);
  ok(pBig > p3 && pBig < 0.6, `交换劈裂把极化率抬到 ${pBig.toFixed(3)}（< 0.6，未饱和）`);
  ok(sb.SOC_LIT.Si > sb.SOC_LIT.C && sb.SOC_LIT.Ge > sb.SOC_LIT.Si, `Δ_SO 输入按文献排序：C ${sb.SOC_LIT.C} < Si ${sb.SOC_LIT.Si} < Ge ${sb.SOC_LIT.Ge} meV`);
}

/* ── spinor EPM 原型：负结果也要钉住 ── */
const socE = await import("../src/phys/soc-epm.js");
{
  const r = socE.socAtGamma(matOf("Si"), { nG: 46 });
  ok(r.C > 3e-6 && r.C < 5e-6, `SOC 前置系数 ħ²/(4m²c²) = ${r.C.toExponential(2)} eV·Å²`);
  ok(Number.isFinite(r.dsoc), `Γ 点 Δ_SO 数值有限（${(r.dsoc * 1e3).toExponential(2)} µeV）`);
  ok(
    r.dsoc < 1e-3,
    `局部 EPM 算出 Δ_SO = ${(r.dsoc * 1e3).toExponential(1)} µeV，比文献 44 meV 小 5 个数量级` +
      `（负结果：局部弱赝势没有芯 d 态，做不出 SOC；若哪天这条失败，说明换了非局域势，请更新文档）`
  );
}

/* ── 不可约 k 网格：同网格下必须与全网格一致 ── */
const kg = await import("../src/phys/kgrid.js");
const sp = await import("../src/phys/spectra.js");
{
  const Si = matOf("Si");
  const im8 = kg.irreducibleMesh(Si, 8);
  ok(im8.reduced > 0 && im8.inBZ > 100, `mesh=8：BZ 内 ${im8.inBZ} 点，不可约 ${im8.reduced} 点`);
  ok(im8.inBZ / im8.reduced > 15, `压缩比 ${(im8.inBZ / im8.reduced).toFixed(1)}×（>15，可用对称性）`);

  const full = sp.dielectric(Si, { nG: 26, mesh: 8, sym: false });
  const red = sp.dielectric(Si, { nG: 26, mesh: 8, sym: true });
  const pk = (d) => Math.max(...Array.from(d.eps2));
  const rel = Math.abs(pk(red) - pk(full)) / pk(full);
  ok(rel < 1e-6, `同网格下不可约网格与全网格一致（ε₂ 峰 ${pk(red).toFixed(2)} vs ${pk(full).toFixed(2)}，差 ${(rel * 100).toExponential(1)}%）`);

  const p16 = pk(sp.dielectric(Si, { nG: 26, mesh: 16, sym: true }));
  const p24 = pk(sp.dielectric(Si, { nG: 26, mesh: 24, sym: true }));
  ok(Math.abs(p24 - p16) / p24 < 0.05, `不可约网格 mesh 16→24 峰高变化 ${((Math.abs(p24 - p16) / p24) * 100).toFixed(2)}%（<5%）`);
  ok(p24 < p16 && p16 < pk(full), `峰高随网格加密单调收敛：${pk(full).toFixed(1)} → ${p16.toFixed(1)} → ${p24.toFixed(1)}`);
}

/* ── 本征求解：当前规模下 Jacobi 够不够用（P2-10 的实测结论） ── */
{
  const mk = (n) => {
    const A = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        A[i * n + j] = i === j ? 10 + 0.3 * i : 0.05 * Math.sin(i * j * 0.7);
        if (i !== j) A[j * n + i] = A[i * n + j];
      }
    return A;
  };
  for (const n of [46, 89]) {
    const A = mk(n);
    const t0 = performance.now();
    const r = eigSym(A, n, 8);
    const ms = performance.now() - t0;
    ok(ms < 200, `n=${n}：单次稠密对角化 ${ms.toFixed(0)} ms（<200，当前规模不是瓶颈）`);
    ok(Number.isFinite(r.values[0]), `n=${n}：最低本征值有限（${r.values[0].toFixed(3)}）`);
  }
}

/* ── 收敛带：ε₂ 的误差估计随网格收紧 ── */
{
  const Si_G = matOf("Si");
  const d9 = sp.dielectric(Si_G, { nG: 26, mesh: 9 });
  const d15 = sp.dielectric(Si_G, { nG: 26, mesh: 15 });
  const d5 = sp.dielectric(Si_G, { nG: 26, mesh: 5 });
  const d8 = sp.dielectric(Si_G, { nG: 26, mesh: 8 });
  const peakRel = (fine, coarse) => {
    let iMax = 0;
    for (let i = 1; i < fine.eps2.length; i++) if (fine.eps2[i] > fine.eps2[iMax]) iMax = i;
    return Math.abs(fine.eps2[iMax] - coarse.eps2[iMax]) / Math.max(1e-9, fine.eps2[iMax]);
  };
  const r9 = peakRel(d9, d5);
  const r15 = peakRel(d15, d8);
  ok(r9 > 0 && r15 > 0, `收敛带非零：9³ 对 5³ 差 ${(r9 * 100).toFixed(1)}%，15³ 对 8³ 差 ${(r15 * 100).toFixed(1)}%`);
  // 误差带是"局部估计"（n 对 n/2），不是全局单调量：真正该断言的是**相邻加密的差在变小**
  const d24 = sp.dielectric(Si_G, { nG: 26, mesh: 24 });
  const step1 = peakRel(d15, d8);
  const step2 = peakRel(d24, d15);
  ok(step2 < step1, `逐级加密的差在收敛：15→24 的 ${(step2 * 100).toFixed(1)}% < 8→15 的 ${(step1 * 100).toFixed(1)}%`);
  ok(step2 < 0.05, `最细一档的局部误差带 ${(step2 * 100).toFixed(1)}%（<5%）`);
}

console.log(fail === 0 ? "\n全部通过 ✔" : `\n${fail} 项失败 ✘`);
process.exit(fail === 0 ? 0 : 1);

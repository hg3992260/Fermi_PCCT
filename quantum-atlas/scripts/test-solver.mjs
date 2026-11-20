/* test-solver.mjs — 统一信封 + 求解器注册表的验收
 *
 * 断言：
 *   1. bands/spin 两个域迁移后，结果与直接调 phys 内核一致（迁移 = 包壳，不丢精度）
 *   2. 信封校验能拦非法问题
 *   3. 求解器内部错误通过 Solution.error 返回，不抛（可被批量流程捕获）
 *   4. registry 与 solve() 派发正确
 */

import { structure, bandsShifted } from "../src/phys/epm.js";
import { materialOf, MATERIALS } from "../src/phys/materials.js";
import { branches } from "../src/phys/magnon.js";
import { solve, makeProblem, list, get, KINDS, validateProblem } from "../src/solver/solve.js";

let fail = 0;
const ok = (cond, msg) => { console.log(`  ${cond ? "ok  " : "FAIL"}  ${msg}`); if (!cond) fail++; };

/* 1. bands 迁移：与直接调 structure 一致 */
{
  const mat = materialOf("Si");
  const direct = structure(mat, { nG: 46, mesh: 6 });
  const directE = bandsShifted(direct);

  const sol = solve({ kind: "bands", spec: { material: "Si" }, params: { nG: 46, mesh: 6, nBands: 12 } });
  ok(sol.error === null, `bands 求解无错误（solver=${sol.solver}）`);
  ok(Math.abs(sol.data.gap - direct.gap) < 1e-9, `gap 与直接内核一致（${sol.data.gap.toFixed(4)} vs ${direct.gap.toFixed(4)}）`);
  ok(Math.abs(sol.data.E[0][0] - directE[0][0]) < 1e-9, `E(0,0) 与直接内核一致（${sol.data.E[0][0].toFixed(4)}）`);
  ok(sol.data.E[0].length === 12, `导出带数 = 12（实际 ${sol.data.E[0].length}）`);
  ok(sol.data.E.length === directE.length, `k 点数一致（${sol.data.E.length}）`);
  ok(sol.provenance.calib.includes("gap"), `provenance 标定项含 gap`);
  ok(sol.meta.nPW >= 19, `meta 记录平面波数（${sol.meta.nPW}）`);
  ok(sol.data.raw && sol.data.raw.gap === sol.data.gap && typeof sol.data.raw.zero === "number", `bands data.raw 完整（供 3D 场景单一数据源）`);
}

/* 2. spin 迁移：与直接调 branches 一致 */
{
  const spec = { kind: "AF", S: 1, J1: 1, J2: 0.3, D: 0.1, n: 60 };
  const direct = branches({ n: 2, kind: "AF", S: 1, J1: 1, J2: 0.3, D: 0.1 }, 60);
  const sol = solve({ kind: "spin", spec });
  ok(sol.error === null, `spin 求解无错误`);
  ok(sol.data.w.length === direct.w.length, `支数一致（${sol.data.w.length}）`);
  let maxd = 0;
  sol.data.w.forEach((row, b) => row.forEach((w, i) => { maxd = Math.max(maxd, Math.abs(w - direct.w[b][i])); }));
  ok(maxd < 1e-9, `色散逐点一致（最大差 ${maxd.toExponential(1)}）`);
  ok(sol.data.gap === direct.gap, `能隙一致（${sol.data.gap}）`);
}

/* 3. 信封校验 */
{
  ok(!validateProblem({ kind: "nope", spec: {} }).ok, `非法 kind 被拦截`);
  ok(!validateProblem({ kind: "bands", spec: 42 }).ok, `非法 spec 类型被拦截`);
  const thrown = (() => { try { makeProblem({ kind: "nope", spec: {} }); return false; } catch { return true; } })();
  ok(thrown, `makeProblem 对非法问题抛错`);
  ok(KINDS.length === 8, `KINDS 覆盖八域（${KINDS.length}）`);
}

/* 4. 求解器内部错误走 error 字段，不抛 */
{
  const sol = solve({ kind: "bands", spec: { material: "不存在" } });
  ok(sol.error !== null && sol.data === null, `未知材料 → error 字段返回（"${sol.error}"）`);
  const sol2 = solve({ kind: "spin", spec: { kind: "XXZ" } });
  ok(sol2.error !== null, `spin 非法 kind → error 字段返回`);
}

/* 5. registry 与派发：八域全覆盖 */
{
  const reg = list();
  const names = reg.map((s) => s.kind).sort().join(",");
  ok(reg.length === 8, `registry 覆盖八域（${reg.length}：${names}）`);
  ok(get("nope") === null, `registry 对未注册 kind 返回 null`);
  const threw = (() => { try { solve({ kind: "nope" }); return false; } catch { return true; } })();
  ok(threw, `solve 对非法/未注册 kind 抛错`);
}

/* 6. 其余六域：每个都能无错求解 + 至少一个与直接内核一致 */
{
  // crystal
  const cr = solve({ kind: "crystal", spec: { material: "Si" } });
  ok(cr.error === null && cr.data.bzFaces === 14, `crystal：无错，BZ 面数 = ${cr.data.bzFaces}`);

  // spectra：与直接 dielectric 一致
  const sp = await import("../src/phys/spectra.js");
  const spDirect = sp.dielectric(materialOf("Si"), { nG: 26, mesh: 8, gamma: 0.1, wMax: 8, dW: 0.02, T: 300, sym: true });
  const spSol = solve({ kind: "spectra", spec: { material: "Si" }, params: { nG: 26, mesh: 8, gamma: 0.1 } });
  ok(spSol.error === null, `spectra 无错`);
  ok(spSol.data.raw && spSol.data.raw.w.length === spSol.data.w.length, `spectra data.raw 完整（单一数据源）`);
  ok(Math.abs(spSol.data.eps2[0] - spDirect.eps2[0]) < 1e-9, `spectra ε₂(0) 与直接内核一致（${spSol.data.eps2[0].toFixed(4)}）`);

  // liquid：与直接 pairCorrelation 一致
  const liq = await import("../src/phys/liquid.js");
  const cfgDirect = liq.liquidConfig({ N: 60, seed: 7 });
  const pcDirect = liq.pairCorrelation(cfgDirect);
  const liqSol = solve({ kind: "liquid", spec: { N: 60, seed: 7, sigma: 3.3, rho: 0.0164 } });
  ok(liqSol.error === null && Math.abs(liqSol.data.g[0] - pcDirect.g[0]) < 1e-9, `liquid g(r) 与直接内核一致`);

  // quantum：Bell 态 concurrence = 1
  const qSol = solve({ kind: "quantum", spec: { n: 2, state: [[1 / Math.SQRT2, 0], [0, 0], [0, 0], [1 / Math.SQRT2, 0]] } });
  ok(qSol.error === null && Math.abs(qSol.data.concurrence - 1) < 1e-9, `quantum Bell 态 concurrence = ${qSol.data.concurrence.toFixed(3)}`);

  // circuit：无噪声与直接 simulate 一致
  const qm = await import("../src/phys/quantum.js");
  const ops = [{ gate: "H", qubit: 0, step: 0 }, { gate: "CNOT", control: 0, qubit: 1, step: 1 }];
  const dirTimeline = qm.simulate(2, ops, 3);
  const cirSol = solve({ kind: "circuit", spec: { n: 2, ops, steps: 3 } });
  ok(cirSol.error === null, `circuit 无错`);
  ok(Math.abs(cirSol.data.final[0] - (dirTimeline[3][0][0] ** 2 + dirTimeline[3][0][1] ** 2)) < 1e-9, `circuit 末态概率与直接 simulate 一致`);
  const cirNoisy = solve({ kind: "circuit", spec: { n: 2, ops, steps: 3, noise: { T1: 6, T2: 2.2, gateTime: 0.4 } } });
  ok(cirNoisy.error === null && cirNoisy.data.frames.length === 4 && cirNoisy.data.frames[3].F < 1, `circuit 含噪声：F=${cirNoisy.data.frames[3].F.toFixed(3)}`);

  // device：无错 + 关键标量在场
  const devSol = solve({ kind: "device", spec: { material: "CZT", d: 1, V: 300, E: 60, T: 300 } });
  ok(devSol.error === null && "cceAvg" in devSol.data.scalars, `device 无错，含 cceAvg=${devSol.data.scalars.cceAvg?.toFixed(3) ?? "?"}`);
}

/* 7. 通用输入：自定义材料（Phase 2 第一步） */
{
  const { defineMaterial } = await import("../src/solver/material.js");

  // 自定义金刚石：改晶格常数 → 带隙应变化且有限（≠ Si、≠ NaN）
  const bigSi = defineMaterial({
    id: "test:diamond-a58",
    kind: "diamond", a: 5.8,
    basis: [[0, 0, 0], [0.25, 0.25, 0.25]],
    nElec: 8,
    vtab: { 3: -0.22, 8: 0.04, 11: 0.08 },
  });
  ok(bigSi.A && bigSi.A.length === 3 && bigSi.kind === "diamond", `defineMaterial 推导出 A 与 kind`);
  const bigSol = solve({ kind: "bands", spec: { material: bigSi }, params: { nG: 46, mesh: 6 } });
  const siSol = solve({ kind: "bands", spec: { material: "Si" }, params: { nG: 46, mesh: 6 } });
  ok(bigSol.error === null, `自定义材料能带求解无错`);
  ok(Number.isFinite(bigSol.data.gap) && bigSol.data.gap > 0, `自定义材料带隙有限（${bigSol.data.gap.toFixed(4)} eV）`);
  ok(Math.abs(bigSol.data.gap - siSol.data.gap) > 0.3, `改晶格常数改变带隙（a=5.8 → ${bigSol.data.gap.toFixed(4)} vs Si ${siSol.data.gap.toFixed(4)}）`);

  // 诚实记录：a=5.0 时这个形状因子集会把带隙压到 0（模型预测 gap 闭合，是"拟合形状因子只在一个 a 处有效"的直接体现）
  const closed = solve({ kind: "bands", spec: { material: defineMaterial({ id: "test:diamond-a50", kind: "diamond", a: 5.0, basis: [[0, 0, 0], [0.25, 0.25, 0.25]], nElec: 8, vtab: { 3: -0.22, 8: 0.04, 11: 0.08 } }) }, params: { nG: 46, mesh: 6 } });
  ok(closed.error === null && closed.data.gap === 0, `a=5.0 时 gap 闭合（${closed.data.gap}）——形状因子只在其拟合的 a 处有效`);

  // formFactors 支持查表名：借用 Si 的形状因子，换晶格常数
  const namedFF = defineMaterial({
    id: "test:diamond-a58b", kind: "diamond", a: 5.8,
    basis: [[0, 0, 0], [0.25, 0.25, 0.25]], nElec: 8, formFactors: "Si",
  });
  const namedSol = solve({ kind: "bands", spec: { material: namedFF }, params: { nG: 46, mesh: 6 } });
  ok(namedSol.error === null && Math.abs(namedSol.data.gap - 1.9276) < 1e-3, `formFactors:"Si" 借表成功（gap=${namedSol.data.gap.toFixed(4)}，等于自带 vtab 的 1.9276）`);

  // 缺字段 → 显式报错（不静默）
  const bad = solve({ kind: "bands", spec: { material: { id: "test:bad", kind: "diamond", a: 5.0, basis: [[0, 0, 0]] } } });
  ok(bad.error !== null && /vtab|nElec/.test(bad.error), `缺失字段显式报错（"${bad.error}"）`);

  // bcc 自定义（Na 类）也能跑
  const bcc = defineMaterial({ id: "test:bcc", kind: "bcc", a: 4.2, basis: [[0, 0, 0]], nElec: 1, vtab: { 2: 0.1 } });
  const bccSol = solve({ kind: "bands", spec: { material: bcc }, params: { nG: 27, mesh: 6 } });
  ok(bccSol.error === null && Number.isFinite(bccSol.data.gap), `bcc 自定义材料能带求解（gap=${bccSol.data.gap.toFixed(4)}，金属则 0 也是有限值）`);

  // 自定义材料走 crystal 域：BZ 面数应 > 0
  const crSol = solve({ kind: "crystal", spec: { material: bcc } });
  ok(crSol.error === null && crSol.data.bzFaces >= 6, `自定义材料 crystal 域 BZ 面数 = ${crSol.data.bzFaces}`);
}

/* 8. 任意晶格（Phase 2 第二步）：共线 FM 的键表输入 */
{
  const PI = Math.PI;
  const { branchesGeneric, j0Of } = await import("../src/phys/magnon-generic.js");

  // 方格子键表 = 4 近邻 + 4 次近邻，应与 legacy branches() 逐点一致
  const sqBonds = [
    { r: [1, 0], J: 1 }, { r: [-1, 0], J: 1 }, { r: [0, 1], J: 1 }, { r: [0, -1], J: 1 },
    { r: [1, 1], J: 0.3 }, { r: [-1, 1], J: 0.3 }, { r: [1, -1], J: 0.3 }, { r: [-1, -1], J: 0.3 },
  ];
  const sqPath = [{ label: "Γ", q: [0, 0] }, { label: "X", q: [PI, 0] }, { label: "M", q: [PI, PI] }, { label: "Γ", q: [0, 0] }];
  const genSq = branchesGeneric({ S: 1, D: 0, bonds: sqBonds, path: sqPath }, 60);
  const legSq = branches({ n: 1, kind: "FM", S: 1, J1: 1, J2: 0.3, D: 0 }, 60);
  let maxd = 0;
  genSq.w[0].forEach((w, i) => { maxd = Math.max(maxd, Math.abs(w - legSq.w[0][i])); });
  ok(maxd < 1e-9, `键表方格子 = legacy 方格子（最大差 ${maxd.toExponential(1)}）`);

  // 一维链：ω = 4SJ(1−cos q)，q=π → 8SJ = 8
  const chainBonds = [{ r: [1, 0], J: 1 }, { r: [-1, 0], J: 1 }];
  const chainPath = [{ label: "Γ", q: [0, 0] }, { label: "X", q: [PI, 0] }];
  const chain = branchesGeneric({ S: 1, D: 0, bonds: chainBonds, path: chainPath }, 60);
  ok(Math.abs(j0Of(chainBonds) - 2) < 1e-12, `链的 J(0)=2（实际 ${j0Of(chainBonds)}）`);
  const wX = chain.w[0][chain.w[0].length - 1];
  ok(Math.abs(wX - 8) < 1e-9, `一维链 ω(X)=8SJ=8（实际 ${wX.toFixed(4)}）`);

  // spin 求解器走任意晶格通路
  const sol = solve({ kind: "spin", spec: { S: 1, D: 0, bonds: chainBonds, path: chainPath, n: 60 } });
  ok(sol.error === null && sol.meta.lattice === "custom", `spin 求解器任意晶格通路（meta.lattice=${sol.meta.lattice}）`);
  ok(Math.abs(sol.data.w[0][sol.data.w[0].length - 1] - 8) < 1e-9, `求解器返回的 ω(X)=${sol.data.w[0][sol.data.w[0].length - 1].toFixed(4)}`);
  const noPath = solve({ kind: "spin", spec: { S: 1, bonds: chainBonds } });
  ok(noPath.error !== null, `任意晶格缺 path → 显式报错`);
}

/* 9. 门列表抽 spec（Phase 2 第三步）：参数化旋转 + 任意矩阵 */
{
  const { rotationMatrix } = await import("../src/solver/circuit.js");

  // Rx(π) ≈ X（至全局相位）：|0⟩ → 概率 1 落在 |1⟩
  const rxPi = solve({ kind: "circuit", spec: { n: 1, ops: [{ gate: "Rx", theta: Math.PI, qubit: 0, step: 0 }], steps: 1 } });
  ok(rxPi.error === null, `Rx(θ) 参数化旋转无错`);
  ok(Math.abs(rxPi.data.final[1] - 1) < 1e-9, `Rx(π)|0⟩ → |1⟩ 概率 ${rxPi.data.final[1].toFixed(6)} ≈ 1`);

  // 任意 2×2 矩阵 = H 矩阵，与命名 H 等价
  const hU = [[[1 / Math.SQRT2, 0], [1 / Math.SQRT2, 0]], [[1 / Math.SQRT2, 0], [-1 / Math.SQRT2, 0]]];
  const byMatrix = solve({ kind: "circuit", spec: { n: 1, ops: [{ gate: hU, qubit: 0, step: 0 }], steps: 1 } });
  const byName = solve({ kind: "circuit", spec: { n: 1, ops: [{ gate: "H", qubit: 0, step: 0 }], steps: 1 } });
  ok(byMatrix.error === null, `任意 2×2 矩阵门无错`);
  ok(Math.abs(byMatrix.data.final[0] - byName.data.final[0]) < 1e-9, `任意矩阵 = 命名 H（|0⟩ 概率 ${byMatrix.data.final[0].toFixed(6)}）`);

  // 未知门名 → 显式报错
  const bad = solve({ kind: "circuit", spec: { n: 1, ops: [{ gate: "NOTAGATE", qubit: 0, step: 0 }], steps: 1 } });
  ok(bad.error !== null && /未知门/.test(bad.error), `未知门显式报错（"${bad.error}"）`);

  // Ry(π) ≈ Y（至全局相位）
  const ry = rotationMatrix("Ry", Math.PI);
  const byRy = solve({ kind: "circuit", spec: { n: 1, ops: [{ gate: ry, qubit: 0, step: 0 }], steps: 1 } });
  ok(Math.abs(byRy.data.final[1] - 1) < 1e-9, `Ry(π)|0⟩ → |1⟩ 概率 ${byRy.data.final[1].toFixed(6)} ≈ 1`);
}

/* 10. 器件材料表抽 spec（Phase 2 第四步）：自定义探测器 + 参数真正生效 */
{
  const { defineDetector } = await import("../src/solver/detector.js");

  // 自定义探测器：与 CZT 相同的几何/迁移率，但寿命差 10×
  const fastDet = defineDetector({
    id: "test:fastdet", Eg: 1.57, mu_e: 1000, mu_h: 80,
    tau_e: 5e-5, tau_h: 5e-6, W: 4.6, rho_m: 5.78, comp: { Cd: 0.45, Zn: 0.05, Te: 0.5 },
  });
  const base = solve({ kind: "device", spec: { material: "CZT", d: 1, V: 300 } });
  const fast = solve({ kind: "device", spec: { material: fastDet, d: 1, V: 300 } });
  ok(fast.error === null, `自定义探测器无错`);
  ok(fast.data.scalars.cceAvg > base.data.scalars.cceAvg, `更长寿命 → CCE 更高（${fast.data.scalars.cceAvg.toFixed(3)} vs ${base.data.scalars.cceAvg.toFixed(3)}）`);

  // 参数真正生效（此前 deviceModel 的 opts 映射有 bug，d/V 被忽略）
  const thick = solve({ kind: "device", spec: { material: "CZT", d: 3, V: 300 } });
  ok(thick.data.scalars.cceAvg < base.data.scalars.cceAvg, `更厚(d=3) → 同偏压下 CCE 更低（${thick.data.scalars.cceAvg.toFixed(3)} vs ${base.data.scalars.cceAvg.toFixed(3)}）`);

  // 未知材料名 → 显式报错（不静默回退 CZT）
  const unknown = solve({ kind: "device", spec: { material: "不存在" } });
  ok(unknown.error !== null && /未知探测器/.test(unknown.error), `未知探测器显式报错（"${unknown.error}"）`);

  // 缺字段 → 显式报错
  const bad = solve({ kind: "device", spec: { material: { id: "test:bad", Eg: 1.5 } } });
  ok(bad.error !== null && /无效/.test(bad.error), `探测器缺字段显式报错`);
}

/* 11. 库出口 + 批量 sweep（Phase 3） */
{
  const { sweep } = await import("../src/solver/index.js");

  // 库出口可用
  const lib = await import("../src/solver/index.js");
  ok(typeof lib.solve === "function" && typeof lib.sweep === "function" && typeof lib.defineMaterial === "function",
    `index.js 库出口完整（solve/sweep/defineMaterial）`);

  // sweep spin.spec.J1：带宽随 J1 单调上升
  const sp = sweep({ kind: "spin", spec: { kind: "FM", S: 1, J1: 1, D: 0 } }, { path: "spec.J1", values: [0.5, 1, 2] });
  ok(sp.solutions.length === 3 && sp.table.every((t) => t.error === null), `spin sweep 无错（${sp.table.length} 个点）`);
  ok(sp.table[0].value < sp.table[1].value && sp.table[1].value < sp.table[2].value,
    `带宽随 J1 单调上升（${sp.table.map((t) => t.value.toFixed(2)).join(" < ")}）`);

  // sweep device.spec.d：CCE 随厚度下降
  const dv = sweep({ kind: "device", spec: { material: "CZT", V: 300 } }, { path: "spec.d", values: [1, 2, 3] });
  ok(dv.table.every((t) => t.error === null), `device sweep 无错`);
  ok(dv.table[0].value > dv.table[1].value && dv.table[1].value > dv.table[2].value,
    `CCE 随厚度单调下降（${dv.table.map((t) => t.value.toFixed(3)).join(" > ")}）`);

  // sweep bands.params.nG：两个截断都出有限带隙
  const bd = sweep({ kind: "bands", spec: { material: "Si" }, params: { nG: 46 } }, { path: "params.nG", values: [46, 65] });
  ok(bd.table.every((t) => t.error === null && Number.isFinite(t.value) && t.value > 0), `bands sweep nG 两个截断带隙有限`);

  // 缺 path → 报错
  const threw = (() => { try { sweep({ kind: "spin", spec: {} }, { values: [1] }); return false; } catch { return true; } })();
  ok(threw, `sweep 缺 path 报错`);

  // sweepAsync：结果与 sync 一致 + 逐个回报 + 确实让出事件循环
  const { sweepAsync } = await import("../src/solver/index.js");
  const synced = sweep({ kind: "spin", spec: { kind: "FM", S: 1, J1: 1, D: 0 } }, { path: "spec.J1", values: [0.5, 1, 2] });
  const progress = [];
  const asynced = await sweepAsync({ kind: "spin", spec: { kind: "FM", S: 1, J1: 1, D: 0 } }, { path: "spec.J1", values: [0.5, 1, 2], onProgress: (i, sol) => progress.push(i) });
  ok(asynced.table.map((t) => t.value).join(",") === synced.table.map((t) => t.value).join(","), `sweepAsync 结果与 sync 一致`);
  ok(progress.join(",") === "1,2,3", `sweepAsync 逐个回报（${progress.join(",")}）`);

  // 让出事件循环：调度一个 0ms 定时器，它应在 sweepAsync 完成**之前**先跑一次（说明中间 yield 过）
  let timerRan = false;
  const timer = new Promise((r) => setTimeout(() => { timerRan = true; r(); }, 0));
  await sweepAsync({ kind: "spin", spec: { kind: "FM", S: 1, J1: 1, D: 0 } }, { path: "spec.J1", values: [0.5, 1, 2, 3, 4, 5, 6, 7, 8] });
  await timer;
  ok(timerRan, `sweepAsync 中间让出事件循环（0ms 定时器在批处理期间触发）`);
}

/* 12. 有限温度磁振子（Phase 4 第一步）：Bose 占据 + 平均场磁化 */
{
  const th = await import("../src/phys/magnon-thermal.js");

  // Bose 因子精确值
  const kT = 0.086173 * 300;             // 25.85 meV
  const exact = 1 / (Math.exp(1 / kT) - 1);
  ok(Math.abs(th.bose(1, 300) - exact) < 1e-9, `bose(1 meV, 300 K) = ${th.bose(1, 300).toFixed(4)}（精确 ${exact.toFixed(4)}）`);

  // 经典极限：ω→0 时 n → kT/ħω − 1/2
  const classic = kT / 0.001 - 0.5;
  ok(Math.abs(th.bose(0.001, 300) - classic) < classic * 0.01, `经典极限 n(ω→0) = ${th.bose(0.001, 300).toFixed(1)} ≈ kT/ħω − 1/2`);

  // 平均场磁化：0 → 1，Tc → ≈0，单调
  const m0 = th.magnetizationT(0, { S: 1, Tc: 300 });
  const mHalf = th.magnetizationT(150, { S: 1, Tc: 300 });
  const mTc = th.magnetizationT(300, { S: 1, Tc: 300 });
  ok(m0 === 1 && mHalf > mTc && mHalf < 1, `磁化 m(0)=${m0}, m(150)=${mHalf.toFixed(3)}, m(Tc)=${mTc.toFixed(3)} 单调`);

  // spin 求解器带温度：零温 occupation=null，有限温 occupation 有值
  const cold = solve({ kind: "spin", spec: { kind: "FM", S: 1, J1: 1, D: 0 } });
  const hot = solve({ kind: "spin", spec: { kind: "FM", S: 1, J1: 1, D: 0, T: 300 } });
  ok(cold.data.occupation === null && cold.data.magnetization === 1, `零温：occupation=null、magnetization=1`);
  ok(hot.data.occupation && Number.isFinite(hot.data.occupation.avg) && hot.data.occupation.avg > 0 && hot.data.magnetization < 1, `有限温：平均占据 ${hot.data.occupation.avg.toFixed(2)}、磁化 ${hot.data.magnetization.toFixed(3)}（跳过 Γ 零点模式）`);
}

/* 13. 任意 Kraus 噪声通道 + 任意门×噪声（Phase 4 第二步） */
{
  const X = [[[0, 0], [1, 0]], [[1, 0], [0, 0]]];
  const H = [[[1 / Math.SQRT2, 0], [1 / Math.SQRT2, 0]], [[1 / Math.SQRT2, 0], [-1 / Math.SQRT2, 0]]];

  // depolarizing p=0 → 无噪声，F=1
  const p0 = solve({ kind: "circuit", spec: { n: 1, ops: [], steps: 1, noise: { channels: [{ q: 0, type: "depolarizing", p: 0 }] } } });
  ok(p0.error === null && Math.abs(p0.data.frames[1].F - 1) < 1e-9, `depolarizing p=0 → F=1（无噪声）`);

  // bitflip p=1 → |0⟩→|1⟩：纯态纯度 1、F=0
  const bf = solve({ kind: "circuit", spec: { n: 1, ops: [], steps: 1, noise: { channels: [{ q: 0, type: "bitflip", p: 1 }] } } });
  ok(bf.error === null && Math.abs(bf.data.frames[1].F) < 1e-9 && Math.abs(bf.data.frames[1].purity - 1) < 1e-9, `bitflip p=1 → F=0、纯度=1`);

  // 自定义 kraus = 确定性 X → 同 bitflip
  const ck = solve({ kind: "circuit", spec: { n: 1, ops: [], steps: 1, noise: { channels: [{ q: 0, kraus: [X] }] } } });
  ok(ck.error === null && Math.abs(ck.data.frames[1].F) < 1e-9, `自定义 Kraus=[X] → F=0`);

  // 任意矩阵门 × 噪声：H + depolarizing p=0.1 → F < 1、纯度 < 1
  const mixed = solve({ kind: "circuit", spec: { n: 1, ops: [{ gate: H, qubit: 0, step: 0 }], steps: 1, noise: { channels: [{ q: 0, type: "depolarizing", p: 0.1 }] } } });
  ok(mixed.error === null && mixed.data.frames[1].F < 1 && mixed.data.frames[1].F > 0.5 && mixed.data.frames[1].purity < 1,
    `任意矩阵门 H + depolarizing(0.1) → F=${mixed.data.frames[1].F.toFixed(3)}、纯度=${mixed.data.frames[1].purity.toFixed(3)}`);

  // 旧接口 T1/T2 仍可用（F < 1）
  const legacy = solve({ kind: "circuit", spec: { n: 2, ops: [{ gate: "H", qubit: 0, step: 0 }, { gate: "CNOT", control: 0, qubit: 1, step: 1 }], steps: 3, noise: { T1: 6, T2: 2.2, gateTime: 0.4 } } });
  ok(legacy.error === null && legacy.data.frames[3].F < 1, `旧接口 T1/T2 仍可用（F=${legacy.data.frames[3].F.toFixed(3)}）`);

  // 噪声路径不支持 CZ/SWAP → 显式报错
  const cz = solve({ kind: "circuit", spec: { n: 2, ops: [{ gate: "CZ", control: 0, qubit: 1, step: 0 }], steps: 1, noise: { channels: [] } } });
  ok(cz.error !== null && /CNOT/.test(cz.error), `噪声路径 CZ → 显式报错`);
}

/* 14. 液态 Monte Carlo（Phase 4 第三步）：LJ 势 + Metropolis */
{
  const mc = await import("../src/phys/liquid-mc.js");

  // 平衡使能量下降：MC 后能量 ≤ 初始能量
  const init = mc.initialEnergy({ N: 100, rho: 0.5, seed: 3 });
  const cfg = mc.metropolisConfig({ N: 100, rho: 0.5, sigma: 1, epsilon: 1, T: 1.0, seed: 3, steps: 30000 });
  ok(cfg.E <= init.E, `Metropolis 平衡使能量下降（${cfg.E.toFixed(2)} ≤ 初始 ${init.E.toFixed(2)}）`);
  ok(cfg.accRate > 0.05 && cfg.accRate < 0.95, `接受率在合理区间（${(cfg.accRate * 100).toFixed(0)}%）`);

  // g(r)：硬核排斥 → 小 r 处 g≈0，第一峰 >1，远场 →1
  // 注意：box 边长 L=∛(N/ρ)，rMax 必须 < L/2（周期边界下最小镜像距离上限）
  const { pairCorrelation } = await import("../src/phys/liquid.js");
  const rMax = 2.8;                       // < L/2 = 2.93
  const pc = pairCorrelation(cfg, { bins: 90, rMax });
  const g = pc.g;
  ok(g[0] < 0.2, `g(0)≈0（硬核排斥，实际 ${g[0].toFixed(3)}）`);
  const peak = Math.max(...g);
  ok(peak > 1.5, `第一峰 >1.5（实际 ${peak.toFixed(2)}）`);
  const tail = g.slice(-10).reduce((a, b) => a + b, 0) / 10;
  ok(Math.abs(tail - 1) < 0.2, `远场 g→1（实际 ${tail.toFixed(2)}）`);

  // 升温 → 结构变弱（第一峰降低）
  const hotCfg = mc.metropolisConfig({ N: 100, rho: 0.5, sigma: 1, epsilon: 1, T: 3.0, seed: 3, steps: 20000 });
  const hotPc = pairCorrelation(hotCfg, { bins: 90, rMax });
  const hotPeak = Math.max(...hotPc.g);
  ok(hotPeak < peak, `升温降低第一峰（${hotPeak.toFixed(2)} < ${peak.toFixed(2)}）`);

  // liquid 求解器走 mc 通路
  const sol = solve({ kind: "liquid", spec: { method: "mc", N: 80, rho: 0.5, T: 1.0, seed: 3, steps: 5000 } });
  ok(sol.error === null && sol.data.g.length > 0 && sol.data.energy !== undefined, `liquid 求解器 mc 通路（E=${sol.data.energy.toFixed(1)}，acc=${(sol.data.accRate * 100).toFixed(0)}%）`);
  const rsa = solve({ kind: "liquid", spec: { N: 60, seed: 7 } });
  ok(rsa.error === null && rsa.meta.method === "rsa", `liquid 求解器 rsa 通路仍可用（旧接口）`);
}

/* 15. 多子晶格自旋波（Phase 4 第四步）：共线 FM 的 m 子晶格 */
{
  const mg = await import("../src/phys/magnon-generic.js");
  const PI = Math.PI;

  // m=1 单子晶格 = frequenciesGeneric（退化正确）
  const sqBonds = [
    { a: 0, b: 0, r: [1, 0], J: 1 }, { a: 0, b: 0, r: [-1, 0], J: 1 },
    { a: 0, b: 0, r: [0, 1], J: 1 }, { a: 0, b: 0, r: [0, -1], J: 1 },
    { a: 0, b: 0, r: [1, 1], J: 0.3 }, { a: 0, b: 0, r: [-1, 1], J: 0.3 },
    { a: 0, b: 0, r: [1, -1], J: 0.3 }, { a: 0, b: 0, r: [-1, -1], J: 0.3 },
  ];
  for (const q of [[0, 0], [0.5, 0], [PI, 0], [PI, PI]]) {
    const multi = mg.frequenciesMulti([{ S: 1, D: 0 }], sqBonds, q);
    const gen = mg.frequenciesGeneric({ S: 1, D: 0, bonds: sqBonds.map(({ r, J }) => ({ r, J })) }, q);
    ok(Math.abs(multi[0] - gen[0]) < 1e-9, `m=1 退化：q=${q.join(",")} ω=${multi[0].toFixed(4)} = generic ${gen[0].toFixed(4)}`);
  }

  // 2 子晶格 FM（方格子，仅次近邻间交换）：ω = 2S[zJ+D±J(q)]，Γ 处 {0, 2S·2zJ}={0,16}，X 处 {8,8}
  const biBonds = [
    { a: 0, b: 1, r: [1, 0], J: 1 }, { a: 0, b: 1, r: [-1, 0], J: 1 },
    { a: 0, b: 1, r: [0, 1], J: 1 }, { a: 0, b: 1, r: [0, -1], J: 1 },
  ];
  const sub = [{ S: 1, D: 0 }, { S: 1, D: 0 }];
  const atGamma = mg.frequenciesMulti(sub, biBonds, [0, 0]);      // J(q)=4 → 2[4±4]={0,16}
  const atX = mg.frequenciesMulti(sub, biBonds, [PI, 0]);          // J(q)=0 → 2[4]={8,8}
  ok(Math.abs(atGamma[0]) < 1e-9 && Math.abs(atGamma[1] - 16) < 1e-9, `2 子晶格 Γ 处两支 {0, 16}（实际 {${atGamma.map((x) => x.toFixed(3)).join(",")}}）`);
  ok(Math.abs(atX[0] - 8) < 1e-9 && Math.abs(atX[1] - 8) < 1e-9, `2 子晶格 X 处两支 {8, 8}（实际 {${atX.map((x) => x.toFixed(3)).join(",")}}）`);

  // spin 求解器多子晶格通路
  const sol = solve({ kind: "spin", spec: { sublattices: sub, bonds: biBonds, path: [{ label: "Γ", q: [0, 0] }, { label: "X", q: [PI, 0] }], n: 40 } });
  ok(sol.error === null && sol.meta.lattice === "multi" && sol.data.w.length === 2, `spin 求解器多子晶格（${sol.meta.sublattices} 子晶格、${sol.data.w.length} 支）`);
  ok(Math.abs(sol.data.w[1][0] - 16) < 1e-9, `最高支 Γ 处 ω=16（实际 ${sol.data.w[1][0].toFixed(3)}）`);
}

/* 16. 可复现（Phase 5 第一步）：快照回放 + 指纹 */
{
  const { fingerprint, dataFingerprint, replay } = await import("../src/solver/index.js");

  // 确定性求解器：bands 同输入 → 同指纹
  const a = solve({ kind: "bands", spec: { material: "Si" }, params: { nG: 46, mesh: 6 } });
  const b = solve({ kind: "bands", spec: { material: "Si" }, params: { nG: 46, mesh: 6 } });
  ok(a.problem && a.problem.spec.material === "Si", `Solution 带 problem 快照（spec.material=${a.problem.spec.material}）`);
  ok(fingerprint(a) === fingerprint(b), `同输入 → 同指纹（${fingerprint(a)}）`);
  ok(dataFingerprint(a) === dataFingerprint(b), `同输入 → 同数据指纹`);

  // 改参数 → 指纹变
  const c = solve({ kind: "bands", spec: { material: "Si" }, params: { nG: 65, mesh: 6 } });
  ok(fingerprint(a) !== fingerprint(c), `改 nG → 指纹变（${fingerprint(a)} vs ${fingerprint(c)}）`);

  // 回放：从快照重算，数据逐位一致
  const r = replay(a);
  ok(r.ok === true && r.same === true, `replay 从快照重算并逐位一致`);

  // 多域回放：spin / device / circuit / liquid 都能回放
  for (const prob of [
    { kind: "spin", spec: { kind: "FM", S: 1, J1: 1, D: 0.1 } },
    { kind: "device", spec: { material: "CZT", d: 1, V: 300 } },
    { kind: "circuit", spec: { n: 2, ops: [{ gate: "H", qubit: 0, step: 0 }, { gate: "CNOT", control: 0, qubit: 1, step: 1 }], steps: 3 } },
    { kind: "liquid", spec: { N: 60, seed: 7 } },
  ]) {
    const s0 = solve(prob);
    const rr = replay(s0);
    ok(rr.ok && rr.same, `${prob.kind} 回放一致`);
  }
}

/* 17. OpenQASM 导出（Phase 5 第二步）：qiskit 对拍 */
{
  const { toQASM, u3Matrix, decomposeU } = await import("../src/solver/index.js");

  // Bell 电路 → 含 h / cx
  const bell = toQASM(2, [{ gate: "H", qubit: 0, step: 0 }, { gate: "CNOT", control: 0, qubit: 1, step: 1 }]);
  ok(/OPENQASM 2\.0/.test(bell) && /qreg q\[2\]/.test(bell) && /h q\[0\]/.test(bell) && /cx q\[0\],q\[1\]/.test(bell), `Bell → 合法 OpenQASM（含 h/cx/measure）`);

  // Rx(π) → rx(3.14159…)
  const rx = toQASM(1, [{ gate: "Rx", theta: Math.PI, qubit: 0, step: 0 }]);
  ok(/rx\(3\.141593\)/.test(rx), `Rx(π) → rx(3.141593)`);

  // 任意矩阵 H → u3(θ,φ,λ)，且 round-trip 重建后与 H 相等（至全局相位）
  const H = [[[1 / Math.SQRT2, 0], [1 / Math.SQRT2, 0]], [[1 / Math.SQRT2, 0], [-1 / Math.SQRT2, 0]]];
  const { theta, phi, lambda } = decomposeU(H);
  const recon = u3Matrix(theta, phi, lambda);
  // 与 H 的逐项重叠（忽略全局相位）：|⟨recon, H⟩| 用 Frobenius 内积归一化
  let ov = 0, nA = 0, nB = 0;
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    ov += recon[i][j][0] * H[i][j][0] + recon[i][j][1] * H[i][j][1];
    nA += recon[i][j][0] ** 2 + recon[i][j][1] ** 2;
    nB += H[i][j][0] ** 2 + H[i][j][1] ** 2;
  }
  const f = ov / Math.sqrt(nA * nB);
  ok(Math.abs(f - 1) < 1e-9, `任意矩阵 H 分解 u3 后重建，重叠 |⟨u3,H⟩|=1（实际 ${f.toFixed(9)}）`);

  // 命名门全量导出
  const all = toQASM(2, [
    { gate: "X", qubit: 0, step: 0 }, { gate: "Y", qubit: 1, step: 0 },
    { gate: "Z", qubit: 0, step: 1 }, { gate: "S", qubit: 1, step: 1 }, { gate: "T", qubit: 0, step: 2 },
  ]);
  ok(/x q\[0\]/.test(all) && /y q\[1\]/.test(all) && /z q\[0\]/.test(all) && /s q\[1\]/.test(all) && /t q\[0\]/.test(all), `命名门 X/Y/Z/S/T 全部导出`);
}

console.log(fail === 0 ? "\n求解器信封：全部通过 ✔" : `\n${fail} 项失败 ✘`);
process.exit(fail === 0 ? 0 : 1);

/* qasm.js — OpenQASM 2.0 导出（Phase 5 第二步：外部对拍）
 *
 * 把 circuit 的 ops 导成 qiskit 能直接 `QuantumCircuit.from_qasm_str` 的文本。
 * 支持：命名门 H/X/Y/Z/S/T、参数化旋转 Rx/Ry/Rz(θ)、CNOT/CZ/SWAP、
 *       以及任意 2×2 复矩阵（分解成 u3(θ,φ,λ)，至全局相位）。
 */

const { cos, sin, atan2, hypot, acos, PI } = Math;
const C = (re, im = 0) => [re, im];

export function u3Matrix(theta, phi, lambda) {
  const c = cos(theta / 2), s = sin(theta / 2);
  return [
    [[c, 0], [-s * cos(lambda), -s * sin(lambda)]],
    [[s * cos(phi), s * sin(phi)], [c * cos(phi + lambda), c * sin(phi + lambda)]],
  ];
}

/* 任意 2×2 复矩阵 → u3(θ,φ,λ)，至全局相位 */
export function decomposeU(U) {
  const [a, b, c] = [U[0][0], U[0][1], U[1][0]];
  const d = U[1][1];
  const mag = (z) => hypot(z[0], z[1]);
  const arg = (z) => atan2(z[1], z[0]);
  const ma = mag(a), mb = mag(b);
  if (mb < 1e-12) {
    // 对角（|b|≈0）：θ=0，只剩相位
    return { theta: 0, phi: 0, lambda: arg(d) - arg(a) };
  }
  const theta = 2 * atan2(mb, ma);
  const phi = arg(c);
  const lambda = arg(b) + PI;      // b = −e^{iλ} sin(θ/2)
  return { theta, phi, lambda };
}

/* 复数乘法（仅用于 round-trip 校验） */
export function mul(a, b) {
  return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
}

/* 单比特门 → QASM 片段（无尾随分号） */
function qasm1(gate, theta) {
  if (gate === "Rx") return `rx(${Number(theta).toFixed(6)})`;
  if (gate === "Ry") return `ry(${Number(theta).toFixed(6)})`;
  if (gate === "Rz") return `rz(${Number(theta).toFixed(6)})`;
  if (Array.isArray(gate)) {
    const { theta: t, phi, lambda } = decomposeU(gate);
    return `u3(${t.toFixed(6)},${phi.toFixed(6)},${lambda.toFixed(6)})`;
  }
  return gate.toLowerCase();       // H/X/Y/Z/S/T
}

export function toQASM(n, ops, { steps = null } = {}) {
  const lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', `qreg q[${n}];`, `creg c[${n}];`];
  const maxStep = steps ?? Math.max(-1, ...ops.map((o) => o.step ?? 0));
  for (let s = 0; s <= maxStep; s++) {
    for (const o of ops.filter((x) => (x.step ?? 0) === s)) {
      if (o.gate === "M") continue;
      if (o.gate === "CNOT") { lines.push(`cx q[${o.control ?? 0}],q[${o.qubit}];`); continue; }
      if (o.gate === "CZ") { lines.push(`cz q[${o.control ?? 0}],q[${o.qubit}];`); continue; }
      if (o.gate === "SWAP") { lines.push(`swap q[${o.control ?? 0}],q[${o.qubit}];`); continue; }
      const g = qasm1(o.gate, o.theta ?? 0);
      lines.push(`${g} q[${o.qubit ?? 0}];`);
    }
  }
  lines.push(`measure q -> c;`);
  return lines.join("\n") + "\n";
}

/* detector.js — 通用探测器材料定义（Phase 2 第四步）
 *
 * 把 DETECTORS 那 9 个硬编码材料变成"用户可定义"。一个探测器对象至少需要：
 *   id, Eg(eV), mu_e, mu_h(cm²/Vs), tau_e, tau_h(s), W(eV), rho_m(g/cm³), comp({元素:原子占比})
 * 可选：pol, tau_pol, name。
 */

export function defineDetector(spec = {}) {
  const { id, name, Eg, mu_e, mu_h, tau_e, tau_h, W, rho_m, comp, pol = 0.1, tau_pol = 10 } = spec;
  const errors = [];
  if (!id) errors.push("id 缺失");
  if (!(Number(Eg) > 0)) errors.push("Eg（带隙 eV）缺失");
  if (!(Number(mu_e) > 0) || !(Number(mu_h) >= 0)) errors.push("mu_e/mu_h（迁移率）缺失");
  if (!(Number(tau_e) > 0) || !(Number(tau_h) > 0)) errors.push("tau_e/tau_h（寿命）缺失");
  if (!(Number(W) > 0)) errors.push("W（电离能 eV）缺失");
  if (!(Number(rho_m) > 0)) errors.push("rho_m（密度 g/cm³）缺失");
  if (!comp || typeof comp !== "object" || Object.keys(comp).length === 0) errors.push("comp（元素成分）缺失");
  if (errors.length) throw new Error("探测器定义无效：" + errors.join("；"));

  return {
    id, name: name ?? id, Eg: Number(Eg),
    mu_e: Number(mu_e), mu_h: Number(mu_h),
    tau_e: Number(tau_e), tau_h: Number(tau_h),
    W: Number(W), rho_m: Number(rho_m),
    comp, pol: Number(pol), tau_pol: Number(tau_pol),
  };
}

/* 解析 spec.material：字符串 → 内置表（未知抛错，不静默回退 CZT）；对象 → 规范化 */
export function resolveDetector(material, DETECTORS) {
  if (typeof material === "string") {
    const m = DETECTORS[material];
    if (!m) throw new Error(`未知探测器材料："${material}"（可用 ${Object.keys(DETECTORS).join("/")}）`);
    return m;
  }
  if (material && typeof material === "object") {
    const complete = material.Eg !== undefined && material.mu_e !== undefined && material.comp !== undefined;
    return complete ? material : defineDetector(material);
  }
  throw new Error("device material 需要 id 字符串或探测器定义对象");
}

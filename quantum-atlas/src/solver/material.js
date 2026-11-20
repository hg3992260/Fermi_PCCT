/* material.js — 通用材料定义（Phase 2 第一步）
 *
 * 把「写死的 5 个材料」变成「用户可定义的材料对象」。
 * defineMaterial(spec) 把一个不完整的用户 spec 规范成 EPM 能直接吃的完整对象，
 * 并对缺失字段显式报错（而不是像旧 materialOf() 那样静默回退）。
 *
 * 一个材料对象至少需要：id, kind, a, basis, nElec, vtab。
 * A（正格子矢量）可由 kind + a 自动推导，species/name/proj 有默认。
 *
 * kind ∈ diamond | fcc | bcc | rocksalt —— 决定 k 路径、结构因子、BZ。
 */

import { MATERIALS } from "../phys/materials.js";

export const KINDS = ["diamond", "fcc", "bcc", "rocksalt"];

/* 各晶格的正格子矢量（原胞基矢，Å） */
const A_FOR = {
  fcc: (a) => [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]],
  diamond: (a) => [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]],
  rocksalt: (a) => [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]],
  bcc: (a) => [[-a / 2, a / 2, a / 2], [a / 2, -a / 2, a / 2], [a / 2, a / 2, -a / 2]],
};

/* 形状因子可来自：vtab（显式）或 formFactors（对象 / 内置材料名，如 "Si"） */
function resolveFormFactors(spec) {
  if (spec.vtab) return spec.vtab;
  const ff = spec.formFactors;
  if (typeof ff === "string") {
    const src = MATERIALS[ff];
    if (!src) throw new Error(`formFactors 引用了未知材料 "${ff}"`);
    return src.vtab;
  }
  if (ff && typeof ff === "object") return ff;
  return null;
}

export function defineMaterial(spec = {}) {
  const { id, kind = "diamond", a, A, basis, nElec, species, name } = spec;
  const vtab = resolveFormFactors(spec);
  const errors = [];
  if (!id) errors.push("id 缺失");
  if (!KINDS.includes(kind)) errors.push(`kind 只支持 ${KINDS.join("/")}（收到 "${kind}"）`);
  if (!(Number(a) > 0)) errors.push("a 必须 > 0（Å）");
  if (!Array.isArray(basis) || basis.length === 0) errors.push("basis 缺失（分数坐标数组）");
  if (!Number.isInteger(nElec) || nElec <= 0) errors.push("nElec 必须为正整数");
  if (!vtab || typeof vtab !== "object" || Object.keys(vtab).length === 0) errors.push("vtab/formFactors（形状因子）缺失");
  if (errors.length) throw new Error("材料定义无效：" + errors.join("；"));

  return {
    id,
    name: name ?? id,
    kind,
    proj: "3D",
    a: Number(a),
    A: A ?? A_FOR[kind](Number(a)),
    basis,
    species: species ?? basis.map(() => id),
    vtab,
    nElec,
  };
}

/* 解析 spec.material：字符串 → 内置表；对象 → 定义/规范化自定义材料 */
export function resolveMaterial(specMaterial, MATERIALS) {
  if (typeof specMaterial === "string") {
    const m = MATERIALS[specMaterial];
    if (!m) throw new Error(`未知材料："${specMaterial}"（可用 ${Object.keys(MATERIALS).join("/")}）`);
    return m;
  }
  if (specMaterial && typeof specMaterial === "object") {
    // 自带 A/basis/vtab/nElec/kind/a 的完整对象直接放行；否则当作 spec 去规范化
    const complete = specMaterial.A && specMaterial.basis && specMaterial.vtab && specMaterial.nElec && specMaterial.kind && specMaterial.a;
    return complete ? specMaterial : defineMaterial(specMaterial);
  }
  throw new Error("material 需要 id 字符串或材料定义对象");
}

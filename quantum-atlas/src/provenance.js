/* provenance.js — 读数行的来源标注与"能力边界"提示
 *
 * 三类来源：
 *   calc  真算出来的（EPM 对角化、跃迁求和、RK4、态矢演化…）
 *   calib 标定量：参数按实验拟合，数值等于"拟合结果"而不是"预测"
 *   input 输入常数：查表/文献值，进模型前就定死了
 *
 * 这件事不写清楚，界面就是在暗示"所有数字都是算出来的"。
 */

export const KINDS = {
  calc: { label: "计算", color: "#009999", mark: "●" },
  calib: { label: "标定", color: "#ec6602", mark: "◆" },
  input: { label: "输入", color: "#8a8a8a", mark: "■" },
};

/* 每个图层里那些"不是纯计算"的键 */
const OVERRIDES = {
  crystal: {
    晶格常数: "input",
    晶格常数a: "input",
    "a(Å)": "input",
    原子数: "input",
    配位数: "input",
  },
  bands: {
    带隙: "calib",
    直接带隙: "calib",
    间接带隙: "calib",
    实验带隙: "input",
    平面波数: "input",
    带数: "input",
    温度: "input",
    "带隙(T)": "calib",
    "ΔEg 刚性(300→T)": "calc",
    "ΔEg Varshni": "calib",
    "ΔEg 刚性": "calc",
    "Zeeman 分裂": "calc",
    "交换劈裂": "input",
    "Δ_SO（文献输入）": "input",
    "价带顶总分裂": "calc",
    "自旋极化率 P": "calc",
    类型: "calib",
  },
  spectra: {
    "ε∞": "calib",
    "ε₁(0)": "calib",
    "ε₂ 峰值": "calib",
    直接带隙: "calib",
    "带隙 Eg": "calib",
    "峰位": "calib",
    "m*_e / m*_h": "calc",
    "约化质量 μ": "calc",
    "激子束缚能": "calc",
    "激子玻尔半径": "calc",
    "吸收边 Eg−Eb": "calc",
  },
  device: {
    "µτe": "input",
    "µτh": "input",
    "μτe": "input",
    "μτh": "input",
    "衰减系数": "input",
    "密度": "input",
    "禁带宽度": "input",
    "FWHM 极限": "input",
  },
  liquid: {
    密度: "input",
    σ: "input",
    "σ(Å)": "input",
    原子数: "input",
  },
  circuit: {
    门数: "input",
    比特数: "input",
    "保真度 F": "calc",
    "纯度 Tr(ρ²)": "calc",
    concurrence: "calc",
    "T₁ / T₂": "input",
    单门耗时: "input",
  },
  bloch: {
    B0: "input",
    T1: "input",
    T2: "input",
    拉莫尔频率: "calc",
  },
};

const DEFAULT_KIND = "calc";

export function kindOf(sceneId, key) {
  const o = OVERRIDES[sceneId];
  if (!o) return DEFAULT_KIND;
  const hit = Object.keys(o).find((k) => key === k || key.replace(/\s+/g, "") === k.replace(/\s+/g, ""));
  return hit ? o[hit] : DEFAULT_KIND;
}

/* 图层级的能力边界提示（静态但诚实） */
const NOTICES = {
  bands: [
    "band",
    "带隙取 46 个平面波、形状因子按实验标定；截断 46→89 的漂移是 7–16%（见 docs/convergence.md）。",
  ],
  spectra: [
    "band",
    "ε₁(0) 被锚定到查表 ε∞，ε₂ 的绝对幅度随这把尺子走；k 网格 8→12 峰高变化约 6%。",
  ],
  device: [
    "band",
    "器件是 1D 模型：无像素间电荷共享、无读出电子学噪声；µτ 与衰减系数来自查表。",
  ],
};

export function noticeOf(sceneId) {
  const n = NOTICES[sceneId];
  return n ? { tone: n[0], text: n[1] } : null;
}

/* ═══════════════════════════════════════════════════════════════
   glyphs.js — 左栏图层图示（16×16 视觉框，内容留 1.2 边距）
   形状取自论文配图的惯例，不是装饰：
   L1 球棍+晶胞 / L2 E(k) 色散 / L3 自旋箭头 / L4 ε₁ ε₂ 双曲线
   L5 无序点 + g(r) / L6 Bloch 球 + 进动矢量 / L7 Q-sphere 纬度环
   L8 导线 + 门 / L9 探测器平板 + 光子 / L0 层叠刻度塔
   ═══════════════════════════════════════════════════════════════ */

const S = 'fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"';

export const GLYPHS = {
  /* L0 全景：三层堆叠 + 顶层一个点 */
  atlas: `<g ${S}>
    <path d="M2.4 11.6 L8 14 L13.6 11.6" opacity=".5"/>
    <path d="M2.4 8.4 L8 10.8 L13.6 8.4" opacity=".75"/>
    <path d="M2.4 5.2 L8 2.8 L13.6 5.2 L8 7.6 Z"/>
    <circle cx="8" cy="5.2" r="1" fill="currentColor" stroke="none"/>
  </g>`,

  /* L1 晶格：立方晶胞 + 中心原子（球棍模型） */
  crystal: `<g ${S}>
    <path d="M3 12.2 L8 14.4 L13 12.2 L13 5.8 L8 3.6 L3 5.8 Z"/>
    <path d="M3 5.8 L8 8 L13 5.8" opacity=".45"/>
    <path d="M8 8 L8 14.4" opacity=".45"/>
    <circle cx="8" cy="8" r="1.5" fill="currentColor" stroke="none"/>
  </g>`,

  /* L2 能带：导带/价带两条色散 + 中间带隙 + 高对称刻线 */
  bands: `<g ${S}>
    <path d="M2.2 6.2 C4.4 8.6 5.8 5.6 7.4 7.2 C9 8.8 10.6 6.4 13.8 7.4" opacity=".55"/>
    <path d="M2.2 12.4 C4.4 10.2 5.8 12.8 7.4 11.4 C9 10 10.6 12.2 13.8 11.2"/>
    <path d="M2.6 13.8 L13.4 13.8" opacity=".3"/>
    <path d="M5.2 13.8 L5.2 13 M10.8 13.8 L10.8 13" opacity=".55"/>
  </g>`,

  /* L3 自旋：三根磁矩箭头（自旋场） */
  spin: `<g ${S}>
    <path d="M2.4 13.4 L13.6 13.4" opacity=".3"/>
    <path d="M4.6 12.6 L4.6 5.6 M4.6 5.6 L3.5 6.9 M4.6 5.6 L5.7 6.9"/>
    <path d="M8 12.6 L8 3.4 M8 3.4 L6.9 4.7 M8 3.4 L9.1 4.7"/>
  </g>`,

  /* L4 光谱：ε₂ 吸收峰 + ε₁ 色散（过零点） */
  spectra: `<g ${S}>
    <path d="M2.4 9.4 L13.6 9.4" opacity=".28"/>
    <path d="M2.4 12 C3.6 12 4.4 4.2 6.4 4.2 C8.4 4.2 9.2 12 10.6 12 C11.6 12 12.4 11 13.6 10.6" opacity=".5"/>
    <path d="M2.4 6 C3.4 6 3.9 11 5.2 11 C6.6 11 6.9 3.4 8.4 3.4 C9.8 3.4 10.2 9 11.6 9 C12.4 9 12.9 8.2 13.6 8"/>
  </g>`,

  /* L5 液态：无序点 + g(r) 起伏 */
  liquid: `<g ${S}>
    <circle cx="3.6" cy="4.6" r="1.1" fill="currentColor" stroke="none" opacity=".65"/>
    <circle cx="7.6" cy="3.4" r="1.1" fill="currentColor" stroke="none" opacity=".9"/>
    <circle cx="11.6" cy="5.2" r="1.1" fill="currentColor" stroke="none" opacity=".5"/>
    <circle cx="5.6" cy="7.6" r="1.1" fill="currentColor" stroke="none" opacity=".75"/>
    <path d="M2.2 13.6 C3 10.6 4 10.6 4.8 13.6 C5.6 10.6 6.4 10.6 7.2 13.6 C8 10.6 8.8 10.6 9.6 13.6 C10.6 10.4 12 11.6 13.8 13.2"/>
  </g>`,

  /* L6 Bloch：球 + 进动矢量 */
  bloch: `<g ${S}>
    <circle cx="8" cy="8" r="5.6" opacity=".5"/>
    <path d="M8 8 L11.9 4.2" stroke-width="1.9"/>
    <circle cx="11.9" cy="4.2" r="1.2" fill="currentColor" stroke="none"/>
    <circle cx="8" cy="8" r="0.8" fill="currentColor" stroke="none" opacity=".7"/>
  </g>`,

  /* L7 量子态：Q-sphere（纬度环 + 大小不同的节点） */
  qubit: `<g ${S}>
    <circle cx="8" cy="8" r="5.6" opacity=".45"/>
    <ellipse cx="8" cy="5.4" rx="4.8" ry="1.7" opacity=".35"/>
    <ellipse cx="8" cy="10.6" rx="4.8" ry="1.7" opacity=".35"/>
    <circle cx="8" cy="2.4" r="1.35" fill="currentColor" stroke="none"/>
    <circle cx="4.6" cy="10.9" r="0.95" fill="currentColor" stroke="none" opacity=".75"/>
    <circle cx="11.4" cy="10.9" r="0.95" fill="currentColor" stroke="none" opacity=".75"/>
  </g>`,

  /* L8 电路：两根导线 + 两个门 + CNOT */
  circuit: `<g ${S}>
    <path d="M2.2 5.4 L13.8 5.4" opacity=".4"/>
    <path d="M2.2 11.6 L13.8 11.6" opacity=".4"/>
    <rect x="3.1" y="3.4" width="3.2" height="4" rx="0.7"/>
    <rect x="7.4" y="9.6" width="3.2" height="4" rx="0.7"/>
    <circle cx="12.4" cy="5.4" r="1.05" fill="currentColor" stroke="none"/>
    <circle cx="12.4" cy="11.6" r="1.5"/>
    <path d="M12.4 10.1 L12.4 13.1 M10.9 11.6 L13.9 11.6"/>
  </g>`,

  /* L9 器件：探测器平板 + 入射光子 + 电场 */
  device: `<g ${S}>
    <path d="M2.6 9.6 L8 11.8 L13.4 9.6 L13.4 6.6 L8 4.4 L2.6 6.6 Z"/>
    <path d="M2.6 9.6 L2.6 11.8 L8 14 L13.4 11.8 L13.4 9.6" opacity=".4"/>
    <path d="M8 1.6 L8 5.6 M8 5.6 L7 4.7 M8 5.6 L9 4.7"/>
    <path d="M5 10 L5 8.4 M8 11 L8 9.4 M11 10 L11 8.4" opacity=".7"/>
  </g>`,
};

export function glyphSVG(id, size = 20) {
  const body = GLYPHS[id] || GLYPHS.atlas;
  return `<svg viewBox="0 0 16 16" width="${size}" height="${size}" aria-hidden="true">${body}</svg>`;
}

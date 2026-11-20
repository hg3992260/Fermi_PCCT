/* ═══════════════════════════════════════════════════════════════
   engine.js — WebGL2 批渲染器（v3：浅色主题 + 更强的三维细节）
   新增：
   · 背景渐变（画布内绘制，随主题切换）
   · 深度雾（depth fog）：远处的线/点/面向背景色收敛，产生纵深
   · 球体 impostor：点不再是圆片，而是带漫反射/高光/边缘光的球
   · thick()：把折线挤出成带宽度的带状面，用于主曲线
   · setPalette()：明暗两套调色板（西门子医疗 brand palette）
   ═══════════════════════════════════════════════════════════════ */

export const V3 = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  mul: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
  lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
};

export function hex(h) {
  h = h.replace("#", "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255];
}

/** 单色渐变：t∈[0,1] → 从 c0 到 c1 再到 c2 */
export function ramp(t, stops) {
  t = Math.max(0, Math.min(1, t));
  const n = stops.length - 1;
  const x = t * n;
  const i = Math.min(n - 1, Math.floor(x));
  return V3.lerp(stops[i], stops[i + 1], x - i);
}

/* ── 两套调色板（西门子医疗 brand：Healthy Orange + Siemens Petrol + 中性灰） ── */
const PALETTES = {
  light: {
    ice: hex("#009999"),        // Siemens Petrol
    petrolDeep: hex("#02787d"),
    cyan: hex("#3abfed"),       // SH Cyan
    violet: hex("#2b2483"),     // SH Blue
    pink: hex("#7a162d"),       // SH Berry
    amber: hex("#ec6602"),      // Healthy Orange
    lime: hex("#009a38"),       // SH Green
    red: hex("#e7001d"),        // SH Red
    slate: hex("#6f6f6f"),      // 中性灰（辅助线）
    white: hex("#1a1a1a"),      // "高亮"在浅底上应当变深
    bgTop: hex("#ffffff"),
    bgBottom: hex("#eef2f2"),
    fog: hex("#f4f7f7"),
    alphaScale: 1.45,
    glow: 0.35,
  },
  dark: {
    ice: hex("#3fc7c7"),
    petrolDeep: hex("#87d2d2"),
    cyan: hex("#3abfed"),
    violet: hex("#9a95d6"),
    pink: hex("#ff9d7a"),
    amber: hex("#ff9a3c"),
    lime: hex("#57c777"),
    red: hex("#ff6b6b"),
    slate: hex("#7f8a95"),
    white: hex("#eef2f6"),
    bgTop: hex("#0e1618"),
    bgBottom: hex("#060a0b"),
    fog: hex("#0a1113"),
    alphaScale: 1.0,
    glow: 0.9,
  },
};

/* C 必须是"自己的数组"：{...PALETTES.light} 是浅拷贝，
   就地改 C 的元素会把母版 PALETTES.light 也一起改掉，
   于是切回浅色时读到的还是深色值——三维永远回不到浅色。 */
export const C = {};
for (const k of Object.keys(PALETTES.light)) {
  C[k] = Array.isArray(PALETTES.light[k]) ? PALETTES.light[k].slice() : PALETTES.light[k];
}
let PALETTE_MODE = "light";

/** 就地改写颜色，保证场景里 `const C` 的引用依然有效 */
export function setPalette(mode) {
  const p = PALETTES[mode] || PALETTES.light;
  PALETTE_MODE = mode;
  for (const k of Object.keys(p)) {
    if (Array.isArray(C[k]) && Array.isArray(p[k])) {
      C[k][0] = p[k][0]; C[k][1] = p[k][1]; C[k][2] = p[k][2];
    } else if (Array.isArray(p[k])) {
      C[k] = p[k].slice();
    } else {
      C[k] = p[k];
    }
  }
}
export const paletteMode = () => PALETTE_MODE;

/* ── 4×4 矩阵 ───────────────────────────────────────────── */
function perspective(fovyDeg, aspect, near, far) {
  const f = 1 / Math.tan((fovyDeg * Math.PI) / 360);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function lookAt(eye, at, up) {
  const z = V3.norm(V3.sub(eye, at));
  const x = V3.norm(V3.cross(up, z));
  const y = V3.cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -V3.dot(x, eye), -V3.dot(y, eye), -V3.dot(z, eye), 1,
  ]);
}

function mul(a, b) {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
      o[i * 4 + j] = s;
    }
  }
  return o;
}

/* ── 着色器 ─────────────────────────────────────────────── */
const VS_BG = `#version 300 es
in vec2 p; out float t;
void main(){ gl_Position = vec4(p, 0.0, 1.0); t = p.y * 0.5 + 0.5; }`;
const FS_BG = `#version 300 es
precision mediump float;
in float t; uniform vec3 cTop; uniform vec3 cBottom; out vec4 o;
void main(){ o = vec4(mix(cBottom, cTop, smoothstep(0.0, 1.0, t)), 1.0); }`;

const VS_LINE = `#version 300 es
in vec3 p; in vec4 c;
uniform mat4 mvp;
out vec4 vc; out float vd;
void main(){ vec4 cp = mvp * vec4(p, 1.0); gl_Position = cp; vc = c; vd = cp.w; }`;
const FS_LINE = `#version 300 es
precision mediump float;
in vec4 vc; in float vd;
uniform vec3 fogColor; uniform vec2 fogRange; uniform float alphaScale;
out vec4 o;
void main(){
  float f = smoothstep(fogRange.x, fogRange.y, vd);
  o = vec4(mix(vc.rgb, fogColor, f * 0.55), vc.a * alphaScale);
}`;

const VS_POINT = `#version 300 es
in vec3 p; in vec4 c; in float s;
uniform mat4 mvp;
out vec4 vc; out float vd;
void main(){
  vec4 cp = mvp * vec4(p, 1.0);
  gl_Position = cp;
  vc = c; vd = cp.w;
  gl_PointSize = s * clamp(11.0 / cp.w, 0.55, 2.4);
}`;
/* 球体 impostor：法线由点内的圆盘坐标重建，做漫反射 + 高光 + 边缘光 */
const FS_POINT = `#version 300 es
precision mediump float;
in vec4 vc; in float vd;
uniform vec3 fogColor; uniform vec2 fogRange; uniform float alphaScale;
out vec4 o;
void main(){
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(q, q);
  if (r2 > 1.0) discard;
  vec3 n = vec3(q.x, -q.y, sqrt(max(0.0, 1.0 - r2)));
  vec3 L = normalize(vec3(-0.42, 0.62, 0.66));
  float diff = max(0.0, dot(n, L));
  vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
  float spec = pow(max(0.0, dot(n, H)), 26.0);
  float rim = pow(1.0 - n.z, 2.2);
  vec3 col = vc.rgb * (0.52 + 0.62 * diff) + vec3(0.55) * spec * 0.55 + vc.rgb * rim * 0.30;
  float f = smoothstep(fogRange.x, fogRange.y, vd);
  o = vec4(mix(col, fogColor, f * 0.55), vc.a * alphaScale * (0.82 + 0.18 * n.z));
}`;

const VS_TRI = `#version 300 es
in vec3 p; in vec3 n; in vec4 c;
uniform mat4 mvp;
out vec4 vc; out vec3 vn; out float vd;
void main(){ vec4 cp = mvp * vec4(p, 1.0); gl_Position = cp; vc = c; vn = n; vd = cp.w; }`;
const FS_TRI = `#version 300 es
precision mediump float;
in vec4 vc; in vec3 vn; in float vd;
uniform vec3 fogColor; uniform vec2 fogRange; uniform float alphaScale;
out vec4 o;
void main(){
  vec3 n = normalize(vn);
  vec3 L1 = normalize(vec3(0.42, 0.86, 0.30));
  vec3 L2 = normalize(vec3(-0.55, 0.15, -0.82));
  float d = 0.52 + 0.40 * abs(dot(n, L1)) + 0.16 * abs(dot(n, L2));
  vec3 col = vc.rgb * d;
  float f = smoothstep(fogRange.x, fogRange.y, vd);
  o = vec4(mix(col, fogColor, f * 0.55), vc.a * alphaScale);
}`;

export class View3D {
  constructor(canvas, labelLayer) {
    this.canvas = canvas;
    this.labelLayer = labelLayer;
    const gl = canvas.getContext("webgl2", { antialias: true, alpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error("需要 WebGL2");
    this.gl = gl;

    this.cam = { yaw: 0.72, pitch: 0.42, dist: 13, target: [0, 0, 0], fov: 42 };
    this.camHome = { ...this.cam, target: [0, 0, 0] };

    this._line = []; this._point = []; this._tri = [];
    this._labels = []; this._picks = [];
    this.highlight = null;
    this.onPick = null;
    this.showGrid = true;

    this._progs = {
      bg: this._program(VS_BG, FS_BG),
      line: this._program(VS_LINE, FS_LINE),
      point: this._program(VS_POINT, FS_POINT),
      tri: this._program(VS_TRI, FS_TRI),
    };
    this._bgBuf = gl.createBuffer();
    this._bgVao = gl.createVertexArray();
    this._buf = { line: gl.createBuffer(), point: gl.createBuffer(), tri: gl.createBuffer() };
    this._vao = { line: gl.createVertexArray(), point: gl.createVertexArray(), tri: gl.createVertexArray() };
    this._cap = { line: 0, point: 0, tri: 0 };

    this._fps = { t: performance.now(), n: 0, val: 0 };
    this._drag = null;
    this._bindPointer();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  _program(vsSrc, fsSrc) {
    const gl = this.gl;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || "shader");
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) || "link");
    return p;
  }

  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, this.canvas.clientWidth);
    const h = Math.max(1, this.canvas.clientHeight);
    if (this._w === w && this._h === h && this._dpr === dpr) return;
    this._w = w; this._h = h; this._dpr = dpr;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    // 改尺寸会清空后备缓冲：必须补一帧，否则布局变化后画布会留黑边
    this.request();
  }

  /* ── 相机 ─────────────────────────────────────────── */
  eye() {
    const { yaw, pitch, dist, target } = this.cam;
    return [
      target[0] + dist * Math.cos(pitch) * Math.sin(yaw),
      target[1] + dist * Math.sin(pitch),
      target[2] + dist * Math.cos(pitch) * Math.cos(yaw),
    ];
  }
  mvp() {
    const aspect = this._w / this._h;
    const proj = perspective(this.cam.fov, aspect, 0.05, 400);
    const view = lookAt(this.eye(), this.cam.target, [0, 1, 0]);
    return mul(proj, view);
  }
  setPreset(name) {
    const c = this.cam;
    if (name === "iso") { c.yaw = 0.72; c.pitch = 0.42; }
    else if (name === "top") { c.yaw = 0.0; c.pitch = 1.45; }
    else if (name === "front") { c.yaw = 0.0; c.pitch = 0.06; }
    else if (name === "reset") { Object.assign(c, this.camHome); }
  }
  fit(r = 4, margin = 1.5) {
    this.cam.dist = (r * margin) / Math.tan((this.cam.fov * Math.PI) / 360);
  }

  /* ── 图元 ─────────────────────────────────────────── */
  frame() {
    this._line.length = 0; this._point.length = 0; this._tri.length = 0;
    this._labels.length = 0; this._picks.length = 0;
  }
  _c(color, a = 1) {
    const c = typeof color === "string" ? hex(color) : color;
    return [c[0], c[1], c[2], a];
  }
  line(a, b, color, a2 = 1) {
    const c = this._c(color, a2);
    this._line.push(a[0], a[1], a[2], ...c, b[0], b[1], b[2], ...c);
  }
  /** 带宽度的折线：沿视线方向挤出成带状面（主曲线用它，1px 线在浅色主题下太弱） */
  thick(pts, color, width = 0.018, a = 1) {
    if (!pts || pts.length < 2) return;
    const eye = this.eye();
    for (let i = 0; i < pts.length - 1; i++) {
      const a0 = pts[i], b0 = pts[i + 1];
      const seg = V3.sub(b0, a0);
      const view = V3.sub(eye, a0);
      let side = V3.cross(seg, view);
      const L = V3.len(side);
      if (L < 1e-6) continue;
      side = V3.mul(side, (width * 0.5) / L);
      this.quad(V3.add(a0, side), V3.add(b0, side), V3.sub(b0, side), V3.sub(a0, side), color, a);
    }
  }
  poly(pts, color, { close = false, a = 1, width = 0 } = {}) {
    if (width > 0) {
      const list = close ? [...pts, pts[0]] : pts;
      this.thick(list, color, width, a);
      return;
    }
    for (let i = 0; i < pts.length - 1; i++) this.line(pts[i], pts[i + 1], color, a);
    if (close && pts.length > 2) this.line(pts[pts.length - 1], pts[0], color, a);
  }
  point(p, color, size = 6, a = 1) {
    this._point.push(p[0], p[1], p[2], ...this._c(color, a), size);
  }
  tri(a, b, c, color, a2 = 1) {
    const n = V3.norm(V3.cross(V3.sub(b, a), V3.sub(c, a)));
    const col = this._c(color, a2);
    this._tri.push(a[0], a[1], a[2], ...n, ...col, b[0], b[1], b[2], ...n, ...col, c[0], c[1], c[2], ...n, ...col);
  }
  quad(a, b, c, d, color, alpha = 1) {
    this.tri(a, b, c, color, alpha); this.tri(a, c, d, color, alpha);
  }
  label(p, text, { cls = "", offset = [0, -12] } = {}) {
    this._labels.push({ p, text, cls, offset });
  }
  pick(p, id, r = 0.25, data = null) {
    this._picks.push({ p, id, r, data });
  }

  /* ── 常用道具 ─────────────────────────────────────── */
  grid(size = 6, step = 1, color, a = 0.10) {
    const col = color || C.slate;
    // 主/次网格两级 + 径向淡出
    for (let x = -size; x <= size; x += step) {
      if (Math.abs(x) < 1e-6) continue;
      const major = Math.abs(Math.round(x / step) % 5) === 0;
      const k = major ? 1.5 : 0.7;
      const fade = (xx, zz) => a * k * Math.max(0.06, 1 - 0.8 * Math.min(1, Math.hypot(xx, zz) / size) ** 2);
      this.line([x, 0, -size], [x, 0, size], col, fade(x, size * 0.55));
      this.line([-size, 0, x], [size, 0, x], col, fade(size * 0.55, x));
    }
  }
  axes(len = 3.2, labels = ["x", "y", "z"]) {
    const cols = [C.amber, C.lime, C.ice];
    const dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (let i = 0; i < 3; i++) {
      const d = V3.mul(dirs[i], len);
      this.line([0, 0, 0], d, cols[i], 0.75);
      this.label(V3.mul(d, 1.06), labels[i], { cls: "dim" });
    }
  }
  sphereWire(c, r, color, { seg = 12, a = 0.35 } = {}) {
    for (let i = 1; i < seg; i++) {
      const phi = (i / seg) * Math.PI;
      const pts = [];
      for (let j = 0; j <= 48; j++) {
        const th = (j / 48) * Math.PI * 2;
        pts.push([c[0] + r * Math.sin(phi) * Math.cos(th), c[1] + r * Math.cos(phi), c[2] + r * Math.sin(phi) * Math.sin(th)]);
      }
      this.poly(pts, color, { a });
    }
    for (let i = 0; i < seg; i++) {
      const th = (i / seg) * Math.PI * 2;
      const pts = [];
      for (let j = 0; j <= 24; j++) {
        const phi = (j / 24) * Math.PI;
        pts.push([c[0] + r * Math.sin(phi) * Math.cos(th), c[1] + r * Math.cos(phi), c[2] + r * Math.sin(phi) * Math.sin(th)]);
      }
      this.poly(pts, color, { a: a * 0.6 });
    }
  }
  arrow(a, b, color, { head = 0.22, a: alpha = 1, width = 1 } = {}) {
    this.line(a, b, color, alpha);
    const d = V3.sub(b, a);
    const L = V3.len(d);
    if (L < 1e-6) return;
    const u = V3.mul(d, 1 / L);
    const up = Math.abs(u[1]) > 0.92 ? [1, 0, 0] : [0, 1, 0];
    const s = V3.mul(V3.norm(V3.cross(u, up)), head * (0.55 + 0.45 * width));
    const base = V3.sub(b, V3.mul(u, head));
    this.line(b, V3.add(base, s), color, alpha);
    this.line(b, V3.sub(base, s), color, alpha);
  }

  /* ── 选中环 ───────────────────────────────────────── */
  _appendSelection() {
    if (!this.highlight) return;
    const pk = this._picks.find((p) => p.id === this.highlight);
    if (!pk) return;
    const d = V3.norm(V3.sub(this.eye(), pk.p));
    const ref = Math.abs(d[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
    const right = V3.norm(V3.cross(d, ref));
    const up = V3.cross(right, d);
    const R = Math.max(0.09, (pk.r || 0.2) * 1.5);
    const pts = [];
    for (let i = 0; i <= 44; i++) {
      const t = (i / 44) * Math.PI * 2;
      pts.push([
        pk.p[0] + (right[0] * Math.cos(t) + up[0] * Math.sin(t)) * R,
        pk.p[1] + (right[1] * Math.cos(t) + up[1] * Math.sin(t)) * R,
        pk.p[2] + (right[2] * Math.cos(t) + up[2] * Math.sin(t)) * R,
      ]);
    }
    this.thick(pts, C.amber, 0.014, 0.95);
  }

  /* ── 渲染 ─────────────────────────────────────────── */
  draw() {
    const gl = this.gl;
    const W = this.canvas.width, H = this.canvas.height;
    gl.viewport(0, 0, W, H);
    gl.clearColor(C.bgBottom[0], C.bgBottom[1], C.bgBottom[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.pointSize = 1;

    this._drawBackground();
    this._appendSelection();
    const mvp = this.mvp();
    const fogRange = [this.cam.dist * 1.05, this.cam.dist * 3.2];

    // 三角面：开深度
    if (this._tri.length) {
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.depthMask(true);
      this._draw(this._progs.tri, this._buf.tri, this._tri, "tri", mvp, 10, fogRange);
      gl.disable(gl.DEPTH_TEST);
    } else {
      gl.disable(gl.DEPTH_TEST);
    }

    if (this._line.length) this._draw(this._progs.line, this._buf.line, this._line, "line", mvp, 7, fogRange);
    if (this._point.length) this._draw(this._progs.point, this._buf.point, this._point, "point", mvp, 8, fogRange);

    this._syncLabels(mvp);
    this._syncGizmo();
    this._fps.n++;
    const now = performance.now();
    if (now - this._fps.t > 500) {
      this._fps.val = Math.round((this._fps.n * 1000) / (now - this._fps.t));
      this._fps.n = 0; this._fps.t = now;
      const el = document.getElementById("hud-fps");
      if (el) el.textContent = this._fps.val >= 5 ? `${this._fps.val} fps` : "待机";
      const nobj = document.getElementById("hud-nobj");
      if (nobj) nobj.textContent = `${(this._line.length / 14 + this._point.length / 8 + this._tri.length / 30) | 0} prim`;
    }
    return this._fps.val;
  }

  _drawBackground() {
    const gl = this.gl;
    gl.disable(gl.DEPTH_TEST);
    const prog = this._progs.bg;
    gl.useProgram(prog);
    gl.bindVertexArray(this._bgVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this._bgBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.DYNAMIC_DRAW);
    const loc = gl.getAttribLocation(prog, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.uniform3fv(gl.getUniformLocation(prog, "cTop"), C.bgTop);
    gl.uniform3fv(gl.getUniformLocation(prog, "cBottom"), C.bgBottom);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  _draw(prog, buf, data, kind, mvp, stride, fogRange) {
    const gl = this.gl;
    gl.useProgram(prog);
    gl.bindVertexArray(this._vao[kind]);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    const need = data.length;
    if (!this._cap[kind] || this._cap[kind] < need) {
      this._cap[kind] = Math.ceil(need * 1.5) + 1024;
      gl.bufferData(gl.ARRAY_BUFFER, this._cap[kind] * 4, gl.DYNAMIC_DRAW);
    }
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(data));

    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "mvp"), false, mvp);
    gl.uniform3fv(gl.getUniformLocation(prog, "fogColor"), C.fog);
    gl.uniform2f(gl.getUniformLocation(prog, "fogRange"), fogRange[0], fogRange[1]);
    gl.uniform1f(gl.getUniformLocation(prog, "alphaScale"), C.alphaScale);

    const strideBytes = stride * 4;
    const attr = (name, size, off) => {
      const loc = gl.getAttribLocation(prog, name);
      if (loc < 0) return;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, strideBytes, off * 4);
    };
    if (kind === "tri") {
      attr("p", 3, 0); attr("n", 3, 3); attr("c", 4, 6);
      gl.drawArrays(gl.TRIANGLES, 0, need / stride);
    } else if (kind === "line") {
      attr("p", 3, 0); attr("c", 4, 3);
      gl.drawArrays(gl.LINES, 0, need / stride);
    } else {
      attr("p", 3, 0); attr("c", 4, 3); attr("s", 1, 7);
      gl.drawArrays(gl.POINTS, 0, need / stride);
    }
    gl.bindVertexArray(null);
  }

  _project(mvp, p) {
    const x = p[0], y = p[1], z = p[2];
    const cx = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12];
    const cy = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13];
    const cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
    if (cw <= 0.001) return null;
    return {
      x: ((cx / cw) * 0.5 + 0.5) * this._w,
      y: (1 - ((cy / cw) * 0.5 + 0.5)) * this._h,
      w: cw,
    };
  }

  /** 迷你坐标轴：把世界三轴投影到屏幕，随相机旋转（画在 DOM 的 SVG 里） */
  _syncGizmo() {
    const host = this._gizmoHost || (this._gizmoHost = document.getElementById("gizmo-axes"));
    if (!host) return;
    const NS = "http://www.w3.org/2000/svg";
    const up = [0, 1, 0];
    const z = V3.norm(V3.sub(this.eye(), this.cam.target));
    const x = V3.norm(V3.cross(up, z));
    const y = V3.cross(z, x);
    const defs = [
      { e: [1, 0, 0], n: "x", cls: "ax-x" },
      { e: [0, 1, 0], n: "y", cls: "ax-y" },
      { e: [0, 0, 1], n: "z", cls: "ax-z" },
    ];
    const items = defs.map((d) => {
      const depth = V3.dot(d.e, z);
      return { ...d, sx: V3.dot(d.e, x), sy: -V3.dot(d.e, y), depth };
    }).sort((a, b) => a.depth - b.depth);
    const R = 19;
    const frag = document.createDocumentFragment();
    for (const it of items) {
      const px = it.sx * R, py = it.sy * R;
      const g = document.createElementNS(NS, "g");
      g.setAttribute("class", it.cls);
      g.setAttribute("opacity", String(0.45 + 0.55 * Math.min(1, Math.max(0, (it.depth + 1) / 2))));
      const ln = document.createElementNS(NS, "line");
      ln.setAttribute("x1", "0"); ln.setAttribute("y1", "0");
      ln.setAttribute("x2", px.toFixed(2)); ln.setAttribute("y2", py.toFixed(2));
      const dot = document.createElementNS(NS, "circle");
      dot.setAttribute("cx", px.toFixed(2)); dot.setAttribute("cy", py.toFixed(2));
      dot.setAttribute("r", "1.5"); dot.setAttribute("fill", "currentColor");
      dot.setAttribute("stroke", "none");
      const tx = document.createElementNS(NS, "text");
      tx.setAttribute("x", (px * 1.3).toFixed(2));
      tx.setAttribute("y", (py * 1.3 + 2.4).toFixed(2));
      tx.setAttribute("stroke", "none");
      tx.textContent = it.n;
      g.append(ln, dot, tx);
      frag.append(g);
    }
    host.replaceChildren(frag);
  }

  _syncLabels(mvp) {
    const layer = this.labelLayer;
    if (!layer) return;
    const W = this._w, H = this._h;
    const cand = [];
    for (const lb of this._labels) {
      const s = this._project(mvp, lb.p);
      if (!s) continue;
      if (s.x < -100 || s.y < -60 || s.x > W + 100 || s.y > H + 60) continue;
      const cls = lb.cls || "";
      const prio = cls.includes("hi") ? 3 : cls.includes("tag") ? 2 : cls.includes("dim") ? 0 : 1;
      cand.push({
        lb, s, prio,
        x: s.x + (lb.offset?.[0] || 0),
        y: s.y + (lb.offset?.[1] || 0),
        w: Math.max(18, lb.text.length * 5.9),
      });
    }
    cand.sort((a, b) => b.prio - a.prio || a.y - b.y);
    const placed = [];
    const frag = document.createDocumentFragment();
    for (const c of cand) {
      const clash = placed.some((p) => Math.abs(p.y - c.y) < 13 && Math.abs(p.x - c.x) < (p.w + c.w) * 0.5 + 6);
      if (clash && c.prio < 2) continue;
      placed.push(c);
      const d = document.createElement("div");
      d.className = "lab " + (c.lb.cls || "");
      d.textContent = c.lb.text;
      d.style.left = `${c.x}px`;
      d.style.top = `${c.y}px`;
      frag.append(d);
      if (placed.length > 90) break;
    }
    layer.replaceChildren(frag);
  }

  /* ── 交互 ─────────────────────────────────────────── */
  _bindPointer() {
    const el = this.canvas;
    let last = null, moved = 0, button = 0;
    el.addEventListener("pointerdown", (e) => {
      el.setPointerCapture(e.pointerId);
      last = { x: e.clientX, y: e.clientY };
      moved = 0; button = e.button;
      el.style.cursor = "grabbing";
    });
    el.addEventListener("pointermove", (e) => {
      if (!last) return;
      const dx = e.clientX - last.x, dy = e.clientY - last.y;
      last = { x: e.clientX, y: e.clientY };
      moved += Math.abs(dx) + Math.abs(dy);
      const pan = button === 2 || e.shiftKey;
      if (pan) {
        const s = this.cam.dist / 900;
        const cy = Math.cos(this.cam.yaw), sy = Math.sin(this.cam.yaw);
        this.cam.target[0] -= (dx * cy + dy * sy) * s;
        this.cam.target[2] -= (-dx * sy + dy * cy) * s;
      } else {
        this.cam.yaw -= dx * 0.0075;
        this.cam.pitch = Math.max(-1.5, Math.min(1.5, this.cam.pitch + dy * 0.0065));
      }
      this.request();
    });
    const up = (e) => {
      if (!last) return;
      last = null;
      el.style.cursor = "grab";
      if (moved < 6) {
        if (e.button === 2 && this.onContext) {
          const rect = el.getBoundingClientRect();
          this.onContext(e.clientX - rect.left, e.clientY - rect.top);
        } else {
          this._click(e);
        }
      }
    };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.cam.dist = Math.max(1.2, Math.min(120, this.cam.dist * Math.exp(e.deltaY * 0.0011)));
      this.request();
    }, { passive: false });
    el.style.cursor = "grab";
  }

  _click(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const mvp = this.mvp();
    let best = null;
    for (const pk of this._picks) {
      const s = this._project(mvp, pk.p);
      if (!s) continue;
      const dx = s.x - mx, dy = s.y - my;
      const d2 = dx * dx + dy * dy;
      const tol = Math.max(10, Math.min(26, (pk.r * 900) / s.w));
      if (d2 > tol * tol) continue;
      const score = d2 - s.w * 2;
      if (!best || score < best.score) best = { score, pk, s };
    }
    if (best && this.onPick) {
      this.highlight = best.pk.id;
      this.onPick(best.pk);
    } else if (this.onPick) {
      this.highlight = null;
      this.onPick(null);
    }
    this.request();
  }

  request() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = 0;
      if (this.onRender) this.onRender();
    });
  }
}

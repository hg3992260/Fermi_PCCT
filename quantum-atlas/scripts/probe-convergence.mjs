/* probe-convergence.mjs — 量化 EPM 的平面波截断与 k 网格敏感性
   目的：给"这个程序算出的数字有多可靠"一个可复现的量化依据。 */
import { structure } from "../src/phys/epm.js";
import { MATERIALS } from "../src/phys/materials.js";

const mats = ["Si", "Ge", "C"];
const nGs = [11, 19, 27, 46, 65, 89];
const meshes = [4, 6, 8, 12];

const mat = (id) => MATERIALS[id] || MATERIALS.find?.((m) => m.id === id);

console.log("材料,nG,mesh,gap_eV,带数,耗时ms");
for (const id of mats) {
  const m = mat(id);
  if (!m) { console.log(`${id},（未找到）`); continue; }
  for (const nG of nGs) {
    const t0 = performance.now();
    let res;
    try {
      res = structure(m, { nG, mesh: 8 });
    } catch (e) {
      console.log(`${id},${nG},8,ERR:${e.message}`);
      continue;
    }
    const ms = Math.round(performance.now() - t0);
    console.log(`${id},${nG},8,${res.gap.toFixed(4)},${res.nBands},${ms}`);
  }
}

console.log("\n--- k 网格敏感性（nG=46）---");
for (const id of ["Si"]) {
  const m = mat(id);
  for (const mesh of meshes) {
    const t0 = performance.now();
    const res = structure(m, { nG: 46, mesh });
    const ms = Math.round(performance.now() - t0);
    console.log(`${id},46,${mesh},${res.gap.toFixed(4)},${res.nBands},${ms}`);
  }
}

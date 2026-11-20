console.log("PROBE type", process.type);
const tries = ["electron", "electron/main", "electron/common", "node:electron"];
for (const id of tries) {
  try {
    const m = require(id);
    const t = typeof m;
    console.log(`RESULT ${id} -> ${t}`, t === "object" ? Object.keys(m).slice(0, 6).join(",") : String(m).slice(0, 60));
  } catch (e) {
    console.log(`RESULT ${id} -> THROW ${e.code || e.message}`);
  }
}
process.exit(0);

// onehop parity test — gates all figure work.
//
// Runs the hand-written JS forward pass (js/model.mjs) on the inputs in
// golden.json and compares every stored PyTorch intermediate — embeddings,
// per-head q/k/v, pre-softmax scores, post-softmax weights, residuals,
// LayerNorm outputs, FFN activations, logits — to within the tolerance
// recorded in golden.json (1e-4). Also checks the greedy decode string.
//
// Usage: node test/parity.mjs

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { forward, greedyDecode, loadModel } from "../js/model.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const config = JSON.parse(readFileSync(join(root, "config.json"), "utf8"));
const golden = JSON.parse(readFileSync(join(root, "golden.json"), "utf8"));
const bin = readFileSync(join(root, "model.bin"));
const model = loadModel(
  config,
  bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength),
);

const TOL = golden.tolerance;
let failures = 0;

/** Walk golden (reference) against ours, comparing every number. */
function compare(ref, ours, path, stats) {
  if (typeof ref === "number") {
    if (typeof ours !== "number" || Number.isNaN(ours)) {
      fail(`${path}: got ${ours}`);
      return;
    }
    const d = Math.abs(ref - ours);
    if (d > stats.max) {
      stats.max = d;
      stats.at = path;
    }
    if (d > TOL) stats.over++;
    return;
  }
  if (Array.isArray(ref)) {
    if (!Array.isArray(ours) || ours.length !== ref.length) {
      fail(`${path}: shape mismatch (ref ${ref.length}, got ${ours?.length})`);
      return;
    }
    for (let i = 0; i < ref.length; i++) compare(ref[i], ours[i], `${path}[${i}]`, stats);
    return;
  }
  for (const key of Object.keys(ref)) {
    if (!(key in ours)) {
      fail(`${path}.${key}: missing from JS trace`);
      continue;
    }
    compare(ref[key], ours[key], `${path}.${key}`, stats);
  }
}

function fail(msg) {
  failures++;
  console.error(`  FAIL ${msg}`);
}

for (const c of golden.cases) {
  console.log(`case: ${JSON.stringify(c.src)}`);
  const trace = forward(model, c.src_ids, c.tgt_in_ids);
  for (const section of ["encoder", "decoder", "logits"]) {
    const stats = { max: 0, at: "", over: 0 };
    compare(c[section], trace[section], section, stats);
    const ok = stats.over === 0;
    if (!ok) failures++;
    console.log(
      `  ${ok ? "ok  " : "FAIL"} ${section.padEnd(7)} max |diff| ${stats.max.toExponential(2)}` +
        (ok ? "" : ` (${stats.over} values over ${TOL}, worst at ${stats.at})`),
    );
  }
  const g = greedyDecode(model, c.src_ids);
  if (g.text === c.greedy) {
    console.log(`  ok   greedy  ${JSON.stringify(g.text)}`);
  } else {
    fail(`greedy: got ${JSON.stringify(g.text)}, golden ${JSON.stringify(c.greedy)}`);
  }
}

if (failures) {
  console.error(`\nparity: FAILED (${failures} failure${failures === 1 ? "" : "s"})`);
  process.exit(1);
}
console.log(`\nparity: PASS (${golden.cases.length} cases, tolerance ${TOL})`);

// onehop — the tensor HUD.
//
// A persistent instrument showing the shape of whatever the reader is
// currently looking at. Figures publish to it; it never captures the
// pointer and never scrolls the page.
//
//   hud.set(stages, activeIndex)   stages: [{label, dims, hue?}]
//                                  dims: [rows, cols] or [heads, rows, cols]
//   hud.active(i)                  move the highlight without relayout
//   hud.value(name, v)             live readout line; hud.value(null) clears
//   hud.idle()                     model card (no figure in view)
//
// Layout is proportional: a 13×48 tensor is drawn as a rect 48 units wide
// and 13 tall; a per-head [4,13,12] stage is four 12-wide rects side by
// side. The reshape from one stage to the next is the content, so rects
// transition geometry (150ms linear) except under prefers-reduced-motion.

import { motion } from "./runtime.mjs";

const SVG = "http://www.w3.org/2000/svg";
const RAIL_W = 172;
const SCALE = 1.15; // px per element; d_ff 128 -> 147px, fits the rail
const GAP = 26; // vertical gap between stages (room for labels)
const HEAD_GAP = 3;

let root, svg, valueEl, cardEl, stageEl;
let stages = null;
let activeIdx = -1;
let model = null;

export function initHUD(el, loadedModel) {
  root = el;
  model = loadedModel ?? null;
  root.innerHTML =
    `<div class="hud-card"></div>` +
    `<svg class="hud-pipe" width="${RAIL_W}" aria-hidden="true"></svg>` +
    `<div class="hud-stage"></div>` +
    `<div class="hud-value" aria-live="polite"></div>`;
  cardEl = root.querySelector(".hud-card");
  svg = root.querySelector(".hud-pipe");
  stageEl = root.querySelector(".hud-stage");
  valueEl = root.querySelector(".hud-value");
  idle();
}

/** Called once weights arrive so the idle card can show the model. */
export function setModel(m) {
  model = m;
  if (!stages) idle();
}

function dimsText(dims) {
  return dims.length === 3
    ? `${dims[0]}×(${dims[1]}×${dims[2]})`
    : dims.join("×");
}

function stageRects(stage, y) {
  // Returns [{x, y, w, h}] for one stage, plus its total height.
  const dims = stage.dims;
  const [rows, cols] = dims.length === 3 ? [dims[1], dims[2]] : dims;
  const h = Math.max(8, Math.min(46, rows * SCALE));
  const w = Math.max(8, cols * SCALE);
  if (dims.length === 3) {
    const n = dims[0];
    const rects = [];
    const total = n * w + (n - 1) * HEAD_GAP;
    const x0 = (RAIL_W - Math.min(total, RAIL_W)) / 2;
    const wEach = total > RAIL_W ? (RAIL_W - (n - 1) * HEAD_GAP) / n : w;
    for (let i = 0; i < n; i++) {
      rects.push({ x: x0 + i * (wEach + HEAD_GAP), y, w: wEach, h });
    }
    return { rects, h };
  }
  return { rects: [{ x: (RAIL_W - Math.min(w, RAIL_W)) / 2, y, w: Math.min(w, RAIL_W), h }], h };
}

function render() {
  if (!root) return;
  cardEl.hidden = !!stages;
  svg.hidden = !stages;
  if (!stages) {
    if (stageEl) stageEl.textContent = "";
    return;
  }

  // Build target geometry for every rect + label.
  const wants = [];
  let y = 6;
  stages.forEach((stage, si) => {
    const { rects, h } = stageRects(stage, y);
    rects.forEach((r) => wants.push({ ...r, stage: si }));
    wants[wants.length - 1].label = {
      text: `${stage.label} ${dimsText(stage.dims)}`,
      y: y + h + 13,
      stage: si,
    };
    y += h + GAP;
  });
  svg.setAttribute("height", y);

  // Reuse rect elements so geometry changes transition; rebuild labels.
  let rects = [...svg.querySelectorAll("rect")];
  if (rects.length !== wants.length) {
    svg.innerHTML = "";
    rects = wants.map(() => svg.appendChild(document.createElementNS(SVG, "rect")));
  } else {
    svg.querySelectorAll("text").forEach((t) => t.remove());
  }
  wants.forEach((wnt, i) => {
    const r = rects[i];
    r.style.transition = motion.reduced ? "none" : "x .15s linear, y .15s linear, width .15s linear, height .15s linear";
    r.style.x = `${wnt.x}px`;
    r.style.y = `${wnt.y}px`;
    r.style.width = `${wnt.w}px`;
    r.style.height = `${wnt.h}px`;
    r.setAttribute("rx", 1.5);
    r.dataset.stage = wnt.stage;
    if (wnt.label) {
      const t = document.createElementNS(SVG, "text");
      t.setAttribute("x", RAIL_W / 2);
      t.setAttribute("y", wnt.label.y);
      t.setAttribute("text-anchor", "middle");
      t.dataset.stage = wnt.label.stage;
      t.textContent = wnt.label.text;
      svg.appendChild(t);
    }
  });
  paint();
}

function paint() {
  const hue = stages?.[activeIdx]?.hue ?? "azure";
  // Text mirror of the active stage — this is what the bottom-strip
  // (narrow viewport) mode shows in place of the pipeline SVG.
  const a = stages?.[activeIdx];
  if (stageEl) stageEl.textContent = a ? `${a.label} ${dimsText(a.dims)}` : "";
  for (const el of svg.querySelectorAll("rect, text")) {
    const on = +el.dataset.stage === activeIdx;
    el.classList.toggle("hud-on", on);
    if (el.tagName === "rect") {
      el.style.fill = on ? `var(--${hue})` : "var(--ink-25)";
    } else {
      el.style.fill = on ? "var(--ink)" : "var(--ink-55)";
    }
  }
}

export function set(newStages, active = 0) {
  stages = newStages;
  activeIdx = active;
  render();
}

export function active(i) {
  if (!stages) return;
  activeIdx = i;
  paint();
}

export function value(name, v) {
  if (!valueEl) return;
  if (name == null) {
    valueEl.textContent = "";
    return;
  }
  valueEl.innerHTML = `<span class="hud-name">${name}</span> = ${typeof v === "number" ? formatV(v) : v}`;
}

function formatV(v) {
  if (!Number.isFinite(v)) return v > 0 ? "+∞" : "−∞";
  return Math.abs(v) >= 1000 ? v.toPrecision(6) : v.toFixed(4).replace("-", "−");
}

export function idle() {
  stages = null;
  activeIdx = -1;
  if (!root) return;
  cardEl.innerHTML = model
    ? `<span class="hud-title">toy model</span>
       ${model.config.n_enc} enc + ${model.config.n_dec} dec ·
       d<sub>model</sub> ${model.config.d_model} · h ${model.config.n_heads}<br>
       ${model.config.n_params.toLocaleString("en-US")} params · fp16<br>
       <span class="hud-note">not the paper's model — trained on date
       normalization so every number here is real</span>`
    : `<span class="hud-title">onehop</span><span class="hud-note">loading weights…</span>`;
  render();
}

export const hud = { set, active, value, idle, setModel };

// Module 3 — Why sqrt(d_k) (paper S3.2.1, footnote 4).
//
// Footnote 4: if the components of q and k are independent with mean 0 and
// variance 1, then q.k = sum q_i k_i has mean 0 and variance d_k. This
// figure samples exactly that — synthetic unit-variance vectors, NOT the
// model's — and shows the consequence and the cure:
//
//   histogram   q.k for 2000 pairs, on a FIXED axis; variance grows with
//               d_k, the +-sqrt(d_k) band widens with it
//   softmax     one query against 16 keys; large scores saturate it
//   gradient    sum_i p_i(1-p_i), the trace of the softmax Jacobian —
//               the gradient magnitude that vanishes as p goes one-hot
//
// The toggle divides by sqrt(d_k). Same vectors, same axis; everything
// snaps back. The dot products are accumulated dimension by dimension, so
// dragging d_k widens the SAME sample rather than redrawing a new one.

import { registerFigure, makeSlider } from "../runtime.mjs";
import { hud } from "../hud.mjs";

const DKS = [4, 8, 16, 32, 64, 128, 256, 512];
const NPAIRS = 2000;
const NKEYS = 16;
const AXIS = 80; // fixed histogram half-range

const css = getComputedStyle(document.documentElement);
const C = {
  paper: css.getPropertyValue("--paper").trim(),
  ink: css.getPropertyValue("--ink").trim(),
  ink55: css.getPropertyValue("--ink-55").trim(),
  ink25: css.getPropertyValue("--ink-25").trim(),
  ink08: css.getPropertyValue("--ink-08").trim(),
  azure: css.getPropertyValue("--azure").trim(),
  vermilion: css.getPropertyValue("--vermilion").trim(),
};

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function initSqrtDk(figEl) {
  figEl.innerHTML = `
    <div class="badge badge-live">live · synthetic unit-variance vectors — footnote 4's assumption, not the model</div>
    <div class="sq-panels">
      <div class="sq-panel"><canvas data-p="hist"></canvas></div>
      <div class="sq-panel"><canvas data-p="soft"></canvas></div>
      <div class="sq-panel"><canvas data-p="grad"></canvas></div>
    </div>
    <div class="fig-controls"></div>
    <figcaption>Dot products of random unit-variance vectors, before the
    softmax. The axis never changes; only d<sub>k</sub> and the scaling do.
    The gauge is Σ pᵢ(1−pᵢ) — the trace of the softmax Jacobian, which is
    what the gradient flows through.</figcaption>`;

  const canvases = {};
  for (const cv of figEl.querySelectorAll("canvas")) canvases[cv.dataset.p] = cv;
  const controls = figEl.querySelector(".fig-controls");

  const st = { dkIdx: 4, scaled: false, seed: 12 };

  // ------------------------------------------------------ sampling
  // Incremental accumulation: dots[d][i] uses the same component stream
  // for every d_k, so the slider widens one sample instead of resampling.
  let dots, scores;
  function sample() {
    const rnd = mulberry32(st.seed);
    const gauss = () => {
      const u = 1 - rnd(), v = rnd();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    dots = new Map(DKS.map((d) => [d, new Float64Array(NPAIRS)]));
    for (let i = 0; i < NPAIRS; i++) {
      let acc = 0, di = 0;
      for (let j = 0; j < DKS[DKS.length - 1]; j++) {
        acc += gauss() * gauss();
        if (j + 1 === DKS[di]) {
          dots.get(DKS[di])[i] = acc;
          di++;
        }
      }
    }
    scores = new Map(DKS.map((d) => [d, new Float64Array(NKEYS)]));
    const q = Array.from({ length: 512 }, gauss);
    for (let k = 0; k < NKEYS; k++) {
      let acc = 0, di = 0;
      for (let j = 0; j < 512; j++) {
        acc += q[j] * gauss();
        if (j + 1 === DKS[di]) {
          scores.get(DKS[di])[k] = acc;
          di++;
        }
      }
    }
  }

  const dk = () => DKS[st.dkIdx];
  const scale = () => (st.scaled ? 1 / Math.sqrt(dk()) : 1);

  function softmax(xs) {
    const m = Math.max(...xs);
    const e = xs.map((x) => Math.exp(x - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  }

  // ------------------------------------------------------ drawing
  function prep(cv, w, h) {
    const dpr = devicePixelRatio || 1;
    cv.width = w * dpr;
    cv.height = h * dpr;
    cv.style.width = `${w}px`;
    cv.style.height = `${h}px`;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return ctx;
  }

  function title(ctx, text, sub) {
    ctx.fillStyle = C.ink;
    ctx.font = "500 12px 'IBM Plex Mono', monospace";
    ctx.fillText(text, 0, 14);
    if (sub) {
      ctx.fillStyle = C.ink55;
      ctx.font = "400 11px 'IBM Plex Mono', monospace";
      ctx.fillText(sub, 0, 30);
    }
  }

  function drawHist() {
    const w = Math.min(400, canvases.hist.parentElement.clientWidth);
    const h = 240;
    const ctx = prep(canvases.hist, w, h);
    const data = dots.get(dk());
    const s = scale();
    const NB = 121;
    const bins = new Float64Array(NB);
    let clipped = 0;
    for (const v0 of data) {
      const v = v0 * s;
      if (Math.abs(v) > AXIS) { clipped++; continue; }
      bins[Math.round(((v + AXIS) / (2 * AXIS)) * (NB - 1))]++;
    }
    const bmax = Math.max(...bins);
    const y0 = h - 34, ph = h - 88;
    const x = (v) => ((v + AXIS) / (2 * AXIS)) * w;

    // predicted-spread band: +-sqrt(d_k), or +-1 once scaled
    const band = st.scaled ? 1 : Math.sqrt(dk());
    ctx.fillStyle = "rgba(0,114,178,0.07)";
    ctx.fillRect(x(-band), y0 - ph, x(band) - x(-band), ph);

    for (let b = 0; b < NB; b++) {
      if (!bins[b]) continue;
      const bh = (bins[b] / bmax) * ph;
      ctx.fillStyle = C.azure;
      ctx.fillRect((b / NB) * w, y0 - bh, w / NB - 0.5, bh);
    }
    // axis
    ctx.fillStyle = C.ink25;
    ctx.fillRect(0, y0, w, 1);
    ctx.fillStyle = C.ink55;
    ctx.font = "400 10px 'IBM Plex Mono', monospace";
    for (const v of [-64, -32, 0, 32, 64]) {
      ctx.fillText(String(v), x(v) - 6, y0 + 14);
    }
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const varr = data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length;
    const sd = Math.sqrt(varr) * s;
    title(ctx, "q·k across 2000 pairs",
      `measured σ ${sd.toFixed(2)} · predicted √d_k${st.scaled ? "/√d_k = 1" : ` = ${Math.sqrt(dk()).toFixed(2)}`}`);
    if (clipped) {
      ctx.fillStyle = C.ink55;
      ctx.fillText(`${clipped} beyond axis`, w - 96, 30);
    }
    return sd;
  }

  function drawSoft() {
    const w = Math.min(300, canvases.soft.parentElement.clientWidth);
    const h = 240;
    const ctx = prep(canvases.soft, w, h);
    const p = softmax([...scores.get(dk())].map((v) => v * scale()));
    const pmax = Math.max(...p);
    const y0 = h - 34, ph = h - 88;
    const bw = w / NKEYS;
    p.forEach((pi, i) => {
      const bh = Math.max(1, pi * ph);
      ctx.fillStyle = pi === pmax ? C.azure : "rgba(0,114,178,0.45)";
      ctx.fillRect(i * bw + 2, y0 - bh, bw - 4, bh);
    });
    ctx.fillStyle = C.ink25;
    ctx.fillRect(0, y0, w, 1);
    ctx.fillStyle = C.ink55;
    ctx.font = "400 10px 'IBM Plex Mono', monospace";
    ctx.fillText("16 keys, one query", 0, y0 + 14);
    title(ctx, "softmax of those scores", `max weight ${pmax.toFixed(3)}`);
    return p;
  }

  function drawGrad(p) {
    const w = Math.min(170, canvases.grad.parentElement.clientWidth);
    const h = 240;
    const ctx = prep(canvases.grad, w, h);
    const g = p.reduce((a, pi) => a + pi * (1 - pi), 0);
    const gmax = 1 - 1 / NKEYS; // uniform distribution maximizes it
    const frac = g / gmax;
    const y0 = h - 34, ph = h - 88;
    const bx = w / 2 - 17;
    ctx.fillStyle = C.ink08;
    ctx.fillRect(bx, y0 - ph, 34, ph);
    ctx.fillStyle = frac < 0.12 ? C.vermilion : C.azure;
    ctx.fillRect(bx, y0 - ph * frac, 34, ph * frac);
    ctx.fillStyle = C.ink25;
    ctx.fillRect(0, y0, w, 1);
    title(ctx, "gradient", `Σ pᵢ(1−pᵢ) = ${g.toFixed(3)}`);
    ctx.fillStyle = C.ink55;
    ctx.font = "400 10px 'IBM Plex Mono', monospace";
    ctx.fillText(frac < 0.12 ? "vanishing" : `${(frac * 100).toFixed(0)}% of max`, bx - 8, y0 + 14);
  }

  function render() {
    const sd = drawHist();
    const p = drawSoft();
    drawGrad(p);
    hud.value(`σ(q·k)${st.scaled ? " · scaled" : ""} at d_k=${dk()}`, +sd.toFixed(3));
  }

  // ------------------------------------------------------ controls
  const dkSlider = makeSlider({
    label: "d_k", min: 0, max: DKS.length - 1, value: st.dkIdx,
    format: (i) => String(DKS[i]),
    onInput: (i) => { st.dkIdx = i; render(); },
  });
  const toggle = document.createElement("button");
  toggle.className = "ctl-btn";
  const setToggle = () => {
    toggle.textContent = st.scaled ? "scaling on · ÷√d_k" : "scaling off";
    toggle.setAttribute("aria-pressed", st.scaled);
    toggle.classList.toggle("on-azure", st.scaled);
  };
  toggle.addEventListener("click", () => { st.scaled = !st.scaled; render(); setToggle(); });
  setToggle();
  const resample = document.createElement("button");
  resample.className = "ctl-btn";
  resample.textContent = "resample";
  resample.addEventListener("click", () => { st.seed++; sample(); render(); });
  controls.append(dkSlider.el, toggle, resample);

  // Static figure — no animation loop; viewport entry points the HUD here.
  registerFigure(figEl, {
    start() {
      hud.set([
        { label: "q·k", dims: [1, NKEYS] },
        { label: "softmax", dims: [1, NKEYS] },
      ], 0);
    },
    stop() { hud.idle(); },
  });

  sample();
  render();
  new ResizeObserver(render).observe(figEl.querySelector(".sq-panels"));
}

// Module 4 — Positional encoding (paper S3.5).
//
// PE(pos, 2i)   = sin(pos / 10000^(2i/d))
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
//
// Each dimension PAIR is one clock hand spinning at angular rate
// omega_i = 1/10000^(2i/d) — rates geometrically spaced from 1 down to
// ~1/10000. The paper's claim that PE(pos+k) is a linear function of
// PE(pos) is the statement that adding k rotates every hand by its own
// FIXED angle k*omega_i, wherever it started. Drag pos and the hands
// spin at their different speeds; drag the offset and every hand
// advances rigidly. The heatmap is the same matrix as rows; the
// similarity matrix PE·PEᵀ/(d/2) shows the banded near-diagonal
// structure this produces.
//
// Computed by the same sinusoidalPE the model itself uses.

import { sinusoidalPE } from "../model.mjs";
import { hud } from "../hud.mjs";
import { registerFigure, makeSlider, probe, probeVerb } from "../runtime.mjs";

const D = 48, MAXPOS = 64, NCLK = D / 2;

const css = getComputedStyle(document.documentElement);
const C = {
  paper: css.getPropertyValue("--paper").trim(),
  ink: css.getPropertyValue("--ink").trim(),
  ink55: css.getPropertyValue("--ink-55").trim(),
  ink25: css.getPropertyValue("--ink-25").trim(),
  azure: css.getPropertyValue("--azure").trim(),
  vermilion: css.getPropertyValue("--vermilion").trim(),
};

export function initPE(figEl) {
  figEl.innerHTML = `
    <div class="badge badge-live">live · the §3.5 formula, the same code the model runs</div>
    <div class="pe-clocks"><canvas data-p="clocks"
      aria-label="one clock hand per dimension pair, rates geometrically spaced"></canvas></div>
    <div class="fig-controls"></div>
    <div class="pe-maps">
      <div class="pe-map"><canvas data-p="pe" aria-label="positional encoding heatmap"></canvas></div>
      <div class="pe-map"><canvas data-p="sim" aria-label="PE similarity matrix"></canvas></div>
    </div>
    <div class="sdpa-readout" aria-live="polite">${probeVerb()} either heatmap to read exact values</div>
    <figcaption>Top: the 24 dimension-pair clocks at your position
    (dark hand) and at position + offset (blue hand). The blue advance is
    the same rotation for a given clock wherever you start — that is the
    linear-map claim, demonstrated. Bottom left: the same encoding as a
    matrix, one row per position. Bottom right: PE·PEᵀ, normalized —
    nearby positions stay similar along the band.</figcaption>`;

  const cv = {};
  for (const c of figEl.querySelectorAll("canvas")) cv[c.dataset.p] = c;
  const readout = figEl.querySelector(".sdpa-readout");
  const controls = figEl.querySelector(".fig-controls");

  const PE = sinusoidalPE(MAXPOS, D);
  // similarity, normalized so the diagonal is exactly 1
  const SIM = PE.map((a) =>
    PE.map((b) => a.reduce((s, v, i) => s + v * b[i], 0) / (D / 2)));

  const st = { pos: 8, k: 4 };
  const omega = (i) => 1 / 10000 ** ((2 * i) / D);

  function prep(c, w, h) {
    const dpr = devicePixelRatio || 1;
    c.width = w * dpr;
    c.height = h * dpr;
    c.style.width = `${w}px`;
    c.style.height = `${h}px`;
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return ctx;
  }

  // ---------------------------------------------------------- clocks
  function drawClocks() {
    const W = cv.clocks.parentElement.clientWidth;
    const perRow = W < 640 ? 8 : 12;
    const rows = Math.ceil(NCLK / perRow);
    const cell = Math.min(72, W / perRow);
    const r = cell * 0.36;
    const H = rows * cell + 18;
    const ctx = prep(cv.clocks, W, H);
    for (let i = 0; i < NCLK; i++) {
      const cx = (i % perRow) * cell + cell / 2 + (W - perRow * cell) / 2;
      const cy = Math.floor(i / perRow) * cell + cell / 2 + 4;
      ctx.strokeStyle = C.ink25;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, 7);
      ctx.stroke();
      const a0 = st.pos * omega(i);
      const a1 = (st.pos + st.k) * omega(i);
      // the advance arc — the fixed rotation k*omega_i
      ctx.strokeStyle = "rgba(0,114,178,0.30)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 0.62, a0 - Math.PI / 2, a1 - Math.PI / 2);
      ctx.stroke();
      const hand = (a, color, len, wd) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = wd;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + len * Math.cos(a - Math.PI / 2), cy + len * Math.sin(a - Math.PI / 2));
        ctx.stroke();
      };
      hand(a0, C.ink55, r * 0.9, 1.5);
      hand(a1, C.azure, r * 0.9, 2);
    }
    ctx.fillStyle = C.ink55;
    ctx.font = "400 10px 'IBM Plex Mono', monospace";
    ctx.fillText("period 2π", 4, H - 2);
    const t = "period 2π·10000";
    ctx.fillText(t, W - ctx.measureText(t).width - 4, H - 2);
  }

  // ---------------------------------------------------------- heatmaps
  function signedColor(v, vmax = 1) {
    const a = Math.min(1, Math.abs(v) / vmax);
    return v >= 0
      ? `rgba(0,114,178,${(0.05 + 0.95 * a).toFixed(3)})`
      : `rgba(213,94,0,${(0.05 + 0.95 * a).toFixed(3)})`;
  }

  function drawPEMap() {
    const W = cv.pe.parentElement.clientWidth;
    const H = Math.round(W * 0.72) + 34;
    const ctx = prep(cv.pe, W, H);
    const cw = W / D, ch = (H - 34) / MAXPOS;
    for (let p = 0; p < MAXPOS; p++) {
      for (let i = 0; i < D; i++) {
        ctx.fillStyle = signedColor(PE[p][i]);
        ctx.fillRect(i * cw, 18 + p * ch, cw + 0.5, ch + 0.5);
      }
    }
    // mark the current row
    ctx.strokeStyle = C.ink;
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 18 + Math.round(st.pos) * ch, W - 0.5, ch);
    ctx.fillStyle = C.ink;
    ctx.font = "500 12px 'IBM Plex Mono', monospace";
    ctx.fillText("PE · 64 positions × 48 dims", 0, 12);
    ctx.fillStyle = C.ink55;
    ctx.font = "400 10px 'IBM Plex Mono', monospace";
    ctx.fillText("fast dims", 0, H - 4);
    const t = "slow dims →";
    ctx.fillText(t, W - ctx.measureText(t).width, H - 4);
  }

  function drawSim() {
    const W = cv.sim.parentElement.clientWidth;
    const H = Math.round(W * 0.72) + 34;
    const ctx = prep(cv.sim, W, H);
    const cw = W / MAXPOS, ch = (H - 34) / MAXPOS;
    for (let a = 0; a < MAXPOS; a++) {
      for (let b = 0; b < MAXPOS; b++) {
        ctx.fillStyle = signedColor(SIM[a][b]);
        ctx.fillRect(b * cw, 18 + a * ch, cw + 0.5, ch + 0.5);
      }
    }
    // mark (pos, pos+k)
    const pa = Math.round(st.pos), pb = Math.min(MAXPOS - 1, Math.round(st.pos + st.k));
    ctx.strokeStyle = C.ink;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(pb * cw - 1, 18 + pa * ch - 1, cw + 2, ch + 2);
    ctx.fillStyle = C.ink;
    ctx.font = "500 12px 'IBM Plex Mono', monospace";
    ctx.fillText("PE·PEᵀ / 24 · 64 × 64", 0, 12);
    ctx.fillStyle = C.ink55;
    ctx.font = "400 10px 'IBM Plex Mono', monospace";
    ctx.fillText(`similarity of pos ${pa} to pos ${pb}: ${SIM[pa][pb].toFixed(3)}`, 0, H - 4);
  }

  function render() {
    drawClocks();
    drawPEMap();
    drawSim();
  }

  // ---------------------------------------------------------- probe
  function mapHover(c, fn) {
    probe(c, (x, y, r) => fn(x / r.width, (y - 18) / (r.height - 34)));
  }
  mapHover(cv.pe, (fx, fy) => {
    const i = Math.floor(fx * D), p = Math.floor(fy * MAXPOS);
    if (i < 0 || i >= D || p < 0 || p >= MAXPOS) return;
    readout.textContent = `PE[${p}][${i}] = ${PE[p][i].toFixed(6)}`;
    hud.value(`PE[${p}][${i}]`, PE[p][i]);
    hud.active(0);
  });
  mapHover(cv.sim, (fx, fy) => {
    const b = Math.floor(fx * MAXPOS), a = Math.floor(fy * MAXPOS);
    if (a < 0 || a >= MAXPOS || b < 0 || b >= MAXPOS) return;
    readout.textContent = `PE·PEᵀ[${a}][${b}] / 24 = ${SIM[a][b].toFixed(6)}`;
    hud.value(`(PE·PEᵀ)[${a}][${b}]/24`, SIM[a][b]);
    hud.active(1);
  });

  // ---------------------------------------------------------- controls
  controls.append(
    makeSlider({
      label: "pos", min: 0, max: MAXPOS - 1, step: 0.25, value: st.pos,
      format: (v) => v.toFixed(2),
      onInput: (v) => { st.pos = v; render(); },
    }).el,
    makeSlider({
      label: "offset k", min: 0, max: 32, step: 0.25, value: st.k,
      format: (v) => v.toFixed(2),
      onInput: (v) => { st.k = v; render(); },
    }).el,
  );

  registerFigure(figEl, {
    start() {
      hud.set([
        { label: "PE", dims: [MAXPOS, D] },
        { label: "PE·PEᵀ", dims: [MAXPOS, MAXPOS] },
      ], 0);
    },
    stop() { hud.idle(); },
  });

  render();
  new ResizeObserver(render).observe(figEl.querySelector(".pe-clocks"));
}

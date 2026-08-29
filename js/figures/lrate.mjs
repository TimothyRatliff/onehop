// Module 9 — The learning-rate schedule (paper S5.3, eq. 3).
//
// lrate = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
//
// The two arms of the min meet exactly at step = warmup_steps: a linear
// ramp up, then inverse-square-root decay. The warmup handle moves the
// kink; the paper's warmup_steps = 4000 with d_model = 512 stays marked.

import { registerFigure, makeSlider, probe, probeVerb } from "../runtime.mjs";
import { hud } from "../hud.mjs";

const css = getComputedStyle(document.documentElement);
const C = {
  ink: css.getPropertyValue("--ink").trim(),
  ink55: css.getPropertyValue("--ink-55").trim(),
  ink25: css.getPropertyValue("--ink-25").trim(),
  azure: css.getPropertyValue("--azure").trim(),
};

const D = 512, SMAX = 100000;
const lrate = (step, warmup) =>
  D ** -0.5 * Math.min(step ** -0.5, step * warmup ** -1.5);

export function initLrate(figEl) {
  figEl.innerHTML = `
    <div class="badge">reported · Vaswani et al. 2017, §5.3 with d_model = 512</div>
    <div class="fig-body"><canvas aria-label="learning rate schedule with warmup handle"></canvas></div>
    <div class="fig-controls"></div>
    <div class="sdpa-readout" aria-live="polite">${probeVerb()} the curve to read the exact rate at any step</div>
    <figcaption>The two arms of the min — the linear ramp and the
    inverse-square-root decay — meet at the kink, exactly at
    warmup_steps. The paper trains with warmup_steps = 4000.</figcaption>`;

  const canvas = figEl.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const readout = figEl.querySelector(".sdpa-readout");
  const st = { warmup: 4000, hover: null };

  let W = 0, H = 360, dpr = 1;
  const PADL = 66, PADR = 16, PADT = 18, PADB = 42;
  const ymax = () => lrate(Math.min(st.warmup, SMAX), st.warmup) * 1.15;
  const xOf = (s) => PADL + (s / SMAX) * (W - PADL - PADR);
  const yOf = (v) => PADT + (1 - v / ymax()) * (H - PADT - PADB);
  const sAt = (x) => ((x - PADL) / (W - PADL - PADR)) * SMAX;

  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = C.ink25;
    ctx.strokeRect(PADL, PADT, W - PADL - PADR, H - PADT - PADB);
    ctx.font = "400 11px 'IBM Plex Mono', monospace";
    ctx.fillStyle = C.ink55;
    for (let s = 0; s <= SMAX; s += 20000) {
      ctx.fillText(s === 0 ? "0" : `${s / 1000}k`, xOf(s) - 8, H - PADB + 16);
    }
    ctx.fillText("training step →", W / 2 - 48, H - 6);
    for (const f of [0.25, 0.5, 0.75, 1]) {
      const v = ymax() * f;
      ctx.fillText(v.toExponential(1), 8, yOf(v) + 4);
    }

    // the two arms, faint, then the min, solid
    const arm = (f, dash) => {
      ctx.strokeStyle = C.ink25;
      ctx.setLineDash(dash);
      ctx.lineWidth = 1;
      ctx.beginPath();
      let started = false;
      for (let x = PADL + 1; x <= W - PADR; x += 2) {
        const s = Math.max(1, sAt(x));
        const v = f(s);
        if (v > ymax()) { started = false; continue; }
        const y = yOf(v);
        started ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        started = true;
      }
      ctx.stroke();
      ctx.setLineDash([]);
    };
    arm((s) => D ** -0.5 * s * st.warmup ** -1.5, [4, 4]);
    arm((s) => D ** -0.5 * s ** -0.5, [4, 4]);

    ctx.strokeStyle = C.azure;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let x = PADL + 1; x <= W - PADR; x += 2) {
      const s = Math.max(1, sAt(x));
      const y = yOf(lrate(s, st.warmup));
      x === PADL + 1 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // the kink
    const kx = xOf(st.warmup), ky = yOf(lrate(st.warmup, st.warmup));
    ctx.fillStyle = C.ink;
    ctx.beginPath();
    ctx.arc(kx, ky, 4, 0, 7);
    ctx.fill();
    ctx.font = "500 12px 'IBM Plex Mono', monospace";
    ctx.fillText(`kink at warmup = ${st.warmup}`, Math.min(kx + 10, W - 230), ky - 10);

    // paper's setting marker
    if (st.warmup !== 4000) {
      ctx.strokeStyle = C.ink25;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(xOf(4000), PADT);
      ctx.lineTo(xOf(4000), H - PADB);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.ink55;
      ctx.font = "400 11px 'IBM Plex Mono', monospace";
      ctx.fillText("paper: 4000", xOf(4000) + 4, PADT + 14);
    }

    if (st.hover) {
      ctx.strokeStyle = C.ink25;
      ctx.beginPath();
      ctx.moveTo(xOf(st.hover), PADT);
      ctx.lineTo(xOf(st.hover), H - PADB);
      ctx.stroke();
      ctx.fillStyle = C.azure;
      ctx.beginPath();
      ctx.arc(xOf(st.hover), yOf(lrate(st.hover, st.warmup)), 3.5, 0, 7);
      ctx.fill();
    }
  }

  function resize() {
    W = figEl.querySelector(".fig-body").clientWidth;
    H = Math.min(380, Math.max(280, Math.round(W * 0.4)));
    dpr = devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    draw();
  }

  probe(canvas, (x) => {
    const s = Math.round(Math.max(1, Math.min(SMAX, sAt(x))));
    st.hover = s;
    const v = lrate(s, st.warmup);
    readout.textContent = `step ${s}: lrate = 512^−0.5 · min(${s}^−0.5, ${s}·${st.warmup}^−1.5) = ${v.toExponential(3)}`;
    hud.value(`lrate(step=${s})`, +v.toExponential(3));
    draw();
  }, { onLeave: () => { st.hover = null; draw(); } });

  figEl.querySelector(".fig-controls").append(
    makeSlider({
      label: "warmup_steps", min: 500, max: 20000, step: 500, value: st.warmup,
      onInput: (v) => { st.warmup = v; draw(); },
    }).el,
  );

  registerFigure(figEl, {
    start() { hud.set([{ label: "lrate(step)", dims: [1, 100] }], 0); },
    stop() { hud.idle(); },
  });
  resize();
  new ResizeObserver(resize).observe(figEl.querySelector(".fig-body"));
}

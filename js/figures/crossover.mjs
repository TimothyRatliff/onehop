// Module 8 — The complexity crossover (paper S4, Table 1).
//
// Per-layer work: self-attention O(n^2 * d), recurrent O(n * d^2).
// Attention is cheaper exactly when n < d. This plots both against
// sequence length on log-log axes with a d slider; the crossover sits
// at n = d, the paper's d = 512 is marked, and the band shades where
// sentence lengths actually live. The reader drags d and watches the
// crossover walk.

import { registerFigure, makeSlider } from "../runtime.mjs";
import { hud } from "../hud.mjs";

const css = getComputedStyle(document.documentElement);
const C = {
  ink: css.getPropertyValue("--ink").trim(),
  ink55: css.getPropertyValue("--ink-55").trim(),
  ink25: css.getPropertyValue("--ink-25").trim(),
  azure: css.getPropertyValue("--azure").trim(),
  vermilion: css.getPropertyValue("--vermilion").trim(),
};

const NMIN = 1, NMAX = 1 << 14; // 1..16384 tokens

export function initCrossover(figEl) {
  figEl.innerHTML = `
    <div class="badge">reported · Vaswani et al. 2017, Table 1 complexities</div>
    <div class="fig-body"><canvas aria-label="per-layer work of self-attention and recurrence versus sequence length"></canvas></div>
    <div class="fig-controls"></div>
    <div class="sdpa-readout" aria-live="polite">hover the plot to compare the two costs at any n</div>
    <figcaption>Per-layer work on log–log axes. The curves cross exactly
    at n = d: below it the quadratic term is the cheaper one. Sentences
    are short; that is the whole bet.</figcaption>`;

  const canvas = figEl.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const readout = figEl.querySelector(".sdpa-readout");
  const st = { d: 512, hoverN: null };

  let W = 0, H = 400, dpr = 1;
  const PADL = 64, PADR = 16, PADT = 20, PADB = 44;
  const YMAX = Math.log10(NMAX * NMAX * 4096); // headroom for biggest d

  const xOf = (n) => PADL + (Math.log2(n) / Math.log2(NMAX)) * (W - PADL - PADR);
  const yOf = (v) => PADT + (1 - Math.log10(v) / YMAX) * (H - PADT - PADB);
  const nAt = (x) => 2 ** (((x - PADL) / (W - PADL - PADR)) * Math.log2(NMAX));

  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.font = "400 11px 'IBM Plex Mono', monospace";

    // sentence-length band: 10..70 tokens
    ctx.fillStyle = "rgba(34,30,25,0.05)";
    ctx.fillRect(xOf(10), PADT, xOf(70) - xOf(10), H - PADT - PADB);
    ctx.fillStyle = C.ink55;
    ctx.fillText("typical sentences", xOf(10) + 4, PADT + 14);

    // axes
    ctx.strokeStyle = C.ink25;
    ctx.lineWidth = 1;
    ctx.strokeRect(PADL, PADT, W - PADL - PADR, H - PADT - PADB);
    ctx.fillStyle = C.ink55;
    for (let e = 0; e <= 12; e += 2) {
      const n = 2 ** e;
      ctx.fillText(String(n), xOf(n) - 8, H - PADB + 16);
    }
    ctx.fillText("sequence length n →", W / 2 - 60, H - 8);
    for (let e = 0; e <= YMAX; e += 4) {
      ctx.fillText(`10^${e}`, 14, yOf(10 ** e) + 4);
    }

    // curves
    const curve = (f, color) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let x = PADL; x <= W - PADR; x += 2) {
        const n = nAt(x);
        const y = yOf(Math.max(1, f(n)));
        x === PADL ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    curve((n) => n * n * st.d, C.azure);
    curve((n) => n * st.d * st.d, C.vermilion);

    // crossover at n = d
    const cx = xOf(st.d), cy = yOf(st.d ** 3);
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = C.ink55;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx, H - PADB);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.ink;
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, 7);
    ctx.fill();
    ctx.font = "500 12px 'IBM Plex Mono', monospace";
    ctx.fillText(`crossover n = d = ${st.d}`, Math.min(cx + 8, W - 210), cy - 10);

    // labels on curves
    ctx.fillStyle = C.azure;
    ctx.fillText("n²·d  self-attention", xOf(2200), yOf(2200 * 2200 * st.d) - 10);
    ctx.fillStyle = C.vermilion;
    ctx.fillText("n·d²  recurrent", xOf(2), yOf(2 * st.d * st.d) - 10);

    // hover
    if (st.hoverN) {
      const n = st.hoverN;
      const x = xOf(n);
      ctx.strokeStyle = C.ink25;
      ctx.beginPath();
      ctx.moveTo(x, PADT);
      ctx.lineTo(x, H - PADB);
      ctx.stroke();
      for (const [f, color] of [[(m) => m * m * st.d, C.azure], [(m) => m * st.d * st.d, C.vermilion]]) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, yOf(f(n)), 3.5, 0, 7);
        ctx.fill();
      }
    }
  }

  function resize() {
    W = figEl.querySelector(".fig-body").clientWidth;
    H = Math.min(420, Math.max(320, Math.round(W * 0.44)));
    dpr = devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    draw();
  }

  canvas.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    const n = Math.round(nAt(e.clientX - r.left));
    if (n < NMIN || n > NMAX) { st.hoverN = null; draw(); return; }
    st.hoverN = n;
    const att = n * n * st.d, rec = n * st.d * st.d;
    const ratio = att / rec;
    readout.textContent =
      `n = ${n}: attention n²·d = ${att.toExponential(2)} · recurrent n·d² = ${rec.toExponential(2)}` +
      ` · attention is ${ratio < 1 ? (1 / ratio).toFixed(1) + "× cheaper" : ratio.toFixed(1) + "× dearer"}`;
    hud.value(`n²d / nd² at n=${n}, d=${st.d}`, +ratio.toFixed(4));
    draw();
  });
  canvas.addEventListener("pointerleave", () => { st.hoverN = null; draw(); });
  // keyboard probe along n
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "slider");
  canvas.setAttribute("aria-label", "probe sequence length n");
  canvas.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const cur = st.hoverN ?? 64;
    st.hoverN = Math.max(NMIN, Math.min(NMAX,
      e.key === "ArrowRight" ? cur * 2 : Math.round(cur / 2)));
    const n = st.hoverN;
    const att = n * n * st.d, rec = n * st.d * st.d;
    const ratio = att / rec;
    readout.textContent =
      `n = ${n}: attention n²·d = ${att.toExponential(2)} · recurrent n·d² = ${rec.toExponential(2)}` +
      ` · attention is ${ratio < 1 ? (1 / ratio).toFixed(1) + "× cheaper" : ratio.toFixed(1) + "× dearer"}`;
    draw();
  });

  figEl.querySelector(".fig-controls").append(
    makeSlider({
      label: "d", min: 6, max: 12, value: Math.log2(st.d),
      format: (v) => String(2 ** v),
      onInput: (v) => { st.d = 2 ** v; draw(); },
    }).el,
  );

  registerFigure(figEl, {
    start() {
      hud.set([{ label: "n²·d vs n·d²", dims: [2, 64] }], 0);
    },
    stop() { hud.idle(); },
  });
  resize();
  new ResizeObserver(resize).observe(figEl.querySelector(".fig-body"));
}

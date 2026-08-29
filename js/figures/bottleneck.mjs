// Module 7 — The bottleneck (paper S4, Table 1: sequential operations).
//
// Training-time view of the same fact module 1 showed at inference:
// a recurrent layer computes h_t from h_{t-1}, so its n positions are a
// chain — O(n) sequential operations. A self-attention layer has no such
// dependency: all n positions compute at once — O(1). The reader sets n
// and watches one layer of each kind process the same sequence.

import { motion, registerFigure, makeSlider } from "../runtime.mjs";

const css = getComputedStyle(document.documentElement);
const C = {
  ink: css.getPropertyValue("--ink").trim(),
  ink55: css.getPropertyValue("--ink-55").trim(),
  ink25: css.getPropertyValue("--ink-25").trim(),
  azure: css.getPropertyValue("--azure").trim(),
  vermilion: css.getPropertyValue("--vermilion").trim(),
};

export function initBottleneck(figEl) {
  figEl.innerHTML = `
    <div class="badge">simulation · sequential dependency only, no model</div>
    <div class="fig-body"><canvas aria-label="recurrent layer filling sequentially versus attention filling at once"></canvas></div>
    <div class="fig-controls"></div>
    <figcaption>One layer of each kind processing the same positions
    during training. Every cell in the top row waits for the cell before
    it — that chain is why there is no parallelism to find inside a
    recurrent example, and why at long sequence lengths memory limits how
    many examples can be batched to compensate.</figcaption>`;

  const canvas = figEl.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const st = { n: 24, t: 0, playing: !motion.reduced };
  const HOP = () => Math.min(260, 5200 / st.n);
  const tEnd = () => st.n * HOP();
  const period = () => tEnd() + 1400;

  let W = 0, H = 190, dpr = 1;
  function resize() {
    W = figEl.querySelector(".fig-body").clientWidth;
    dpr = devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    draw();
  }

  function draw() {
    const t = Math.min(st.t, tEnd());
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const PADL = 10, PADR = 150;
    const cw = (W - PADL - PADR) / st.n;
    const rows = [
      { y: 46, label: "recurrent", sub: "h_t needs h_t−1", color: C.vermilion,
        done: Math.min(st.n, Math.floor(t / HOP())),
        steps: `${Math.min(st.n, Math.floor(t / HOP()))}/${st.n} sequential` },
      { y: 128, label: "self-attention", sub: "no order among positions", color: C.azure,
        done: t >= HOP() ? st.n : 0,
        steps: `${t >= HOP() ? 1 : 0}/1 parallel` },
    ];
    for (const r of rows) {
      ctx.font = "500 12px 'IBM Plex Mono', monospace";
      ctx.fillStyle = C.ink;
      ctx.fillText(r.label, PADL, r.y - 26);
      ctx.font = "400 11px 'IBM Plex Mono', monospace";
      ctx.fillStyle = C.ink55;
      ctx.fillText(r.sub, PADL, r.y - 12);
      for (let i = 0; i < st.n; i++) {
        const filled = i < r.done;
        ctx.fillStyle = filled ? r.color : "rgba(34,30,25,0.08)";
        ctx.fillRect(PADL + i * cw, r.y, cw - 2, 22);
      }
      // dependency arrows on the recurrent row
      if (r.label === "recurrent" && cw > 14) {
        ctx.strokeStyle = C.ink25;
        ctx.lineWidth = 1;
        for (let i = 1; i < st.n; i++) {
          const x = PADL + i * cw - 1;
          ctx.beginPath();
          ctx.moveTo(x - 4, r.y + 33);
          ctx.lineTo(x + 3, r.y + 33);
          ctx.lineTo(x, r.y + 30);
          ctx.moveTo(x + 3, r.y + 33);
          ctx.lineTo(x, r.y + 36);
          ctx.stroke();
        }
      }
      ctx.font = "500 12px 'IBM Plex Mono', monospace";
      ctx.fillStyle = C.ink;
      ctx.fillText(r.steps, W - PADR + 12, r.y + 15);
    }
  }

  let raf = null, last = null;
  function tick(ts) {
    raf = requestAnimationFrame(tick);
    if (last != null && st.playing) {
      st.t = (st.t + (ts - last)) % period();
      draw();
    }
    last = ts;
  }

  const nS = makeSlider({
    label: "n", min: 8, max: 64, value: st.n,
    onInput: (v) => { st.n = v; st.t = motion.reduced && !st.playing ? tEnd() : 0; draw(); },
  });
  const play = document.createElement("button");
  play.className = "ctl-btn";
  const setPlay = () => { play.textContent = st.playing ? "pause" : "play"; };
  play.addEventListener("click", () => { st.playing = !st.playing; if (st.playing && st.t >= tEnd()) st.t = 0; setPlay(); });
  setPlay();
  const scrub = makeSlider({
    label: "scrub", min: 0, max: 100, value: motion.reduced ? 100 : 0, unit: "%",
    onInput: (v) => { st.playing = false; setPlay(); st.t = (v / 100) * tEnd(); draw(); },
  });
  figEl.querySelector(".fig-controls").append(nS.el, play, scrub.el);

  registerFigure(figEl, {
    start() { last = null; if (!raf) raf = requestAnimationFrame(tick); },
    stop() { if (raf) cancelAnimationFrame(raf); raf = null; },
  });
  if (motion.reduced) { st.playing = false; setPlay(); st.t = tEnd(); }
  resize();
  new ResizeObserver(resize).observe(figEl.querySelector(".fig-body"));
}

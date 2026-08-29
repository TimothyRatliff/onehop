// Module 11 — Cost against quality (paper Table 2).
//
// BLEU against training FLOPs on a log axis, EN-DE / EN-FR toggle.
// Every point is transcribed from Table 2. The transformer sits up and
// to the left; the argument makes itself, so nothing here narrates it.

import { registerFigure } from "../runtime.mjs";
import { hud } from "../hud.mjs";

const css = getComputedStyle(document.documentElement);
const C = {
  paper: css.getPropertyValue("--paper").trim(),
  ink: css.getPropertyValue("--ink").trim(),
  ink55: css.getPropertyValue("--ink-55").trim(),
  ink25: css.getPropertyValue("--ink-25").trim(),
  azure: css.getPropertyValue("--azure").trim(),
};

// [name, BLEU, FLOPs] per language pair; null = not reported
const DATA = {
  "EN-DE": [
    ["ByteNet", 23.75, null],
    ["GNMT + RL", 24.6, 2.3e19],
    ["ConvS2S", 25.16, 9.6e18],
    ["MoE", 26.03, 2.0e19],
    ["GNMT + RL ensemble", 26.30, 1.8e20],
    ["ConvS2S ensemble", 26.36, 7.7e19],
    ["Transformer (base)", 27.3, 3.3e18],
    ["Transformer (big)", 28.4, 2.3e19],
  ],
  "EN-FR": [
    ["Deep-Att + PosUnk", 39.2, 1.0e20],
    ["GNMT + RL", 39.92, 1.4e20],
    ["ConvS2S", 40.46, 1.5e20],
    ["MoE", 40.56, 1.2e20],
    ["Deep-Att + PosUnk ensemble", 40.4, 8.0e20],
    ["GNMT + RL ensemble", 41.16, 1.1e21],
    ["ConvS2S ensemble", 41.29, 1.2e21],
    ["Transformer (base)", 38.1, 3.3e18],
    ["Transformer (big)", 41.8, 2.3e19],
  ],
};
const RANGE = {
  "EN-DE": { bmin: 23, bmax: 29 },
  "EN-FR": { bmin: 37.5, bmax: 42.5 },
};
const EMIN = 18, EMAX = 21.5; // log10 FLOPs axis

export function initCost(figEl) {
  figEl.innerHTML = `
    <div class="badge">reported · Vaswani et al. 2017, Table 2 · newstest2014</div>
    <div class="head-picker cost-picker" role="radiogroup" aria-label="language pair"></div>
    <div class="fig-body"><canvas aria-label="BLEU against training FLOPs, log axis"></canvas></div>
    <div class="sdpa-readout" aria-live="polite">hover a point for its exact numbers</div>
    <figcaption>Quality against training compute, one point per model.
    Points without a reported cost are omitted from the plot.
    <span class="footmark" tabindex="0">A footnote for the close
    reader:</span> Table 2 and the abstract give the big model 41.8 on
    EN-FR; the text of §6.1 says 41.0. The tables are the ones everyone
    cites.</figcaption>`;

  const picker = figEl.querySelector(".cost-picker");
  const canvas = figEl.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const readout = figEl.querySelector(".sdpa-readout");
  const st = { pair: "EN-DE", hover: null };

  for (const p of Object.keys(DATA)) {
    const b = document.createElement("button");
    b.className = "ctl-btn head-btn";
    b.textContent = p;
    b.setAttribute("role", "radio");
    b.addEventListener("click", () => { st.pair = p; st.hover = null; draw(); paint(); });
    picker.append(b);
  }
  const paint = () => picker.querySelectorAll(".head-btn").forEach((b) => {
    const on = b.textContent === st.pair;
    b.setAttribute("aria-checked", on);
    b.classList.toggle("on", on);
  });
  paint();

  let W = 0, H = 420, dpr = 1;
  const PADL = 56, PADR = 20, PADT = 16, PADB = 46;
  const xOf = (f) => PADL + ((Math.log10(f) - EMIN) / (EMAX - EMIN)) * (W - PADL - PADR);
  const yOf = (b) => {
    const { bmin, bmax } = RANGE[st.pair];
    return PADT + (1 - (b - bmin) / (bmax - bmin)) * (H - PADT - PADB);
  };

  function pts() {
    return DATA[st.pair].filter(([, , f]) => f != null);
  }

  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = C.ink25;
    ctx.strokeRect(PADL, PADT, W - PADL - PADR, H - PADT - PADB);
    ctx.font = "400 11px 'IBM Plex Mono', monospace";
    ctx.fillStyle = C.ink55;
    for (let e = 18; e <= 21; e++) {
      ctx.fillText(`10^${e}`, xOf(10 ** e) - 12, H - PADB + 16);
    }
    ctx.fillText("training FLOPs (log) →", W / 2 - 70, H - 8);
    const { bmin, bmax } = RANGE[st.pair];
    for (let b = Math.ceil(bmin); b <= bmax; b++) {
      ctx.fillText(b.toFixed(0), 26, yOf(b) + 4);
    }
    ctx.save();
    ctx.translate(12, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("BLEU", -16, 0);
    ctx.restore();

    for (const [name, bleu, flops] of pts()) {
      const ours = name.startsWith("Transformer");
      const x = xOf(flops), y = yOf(bleu);
      ctx.fillStyle = ours ? C.azure : "rgba(34,30,25,0.45)";
      ctx.beginPath();
      ctx.arc(x, y, ours ? 6 : 4.5, 0, 7);
      ctx.fill();
      if (st.hover === name) {
        ctx.strokeStyle = C.ink;
        ctx.stroke();
      }
      ctx.font = `${ours ? "500" : "400"} 11px 'IBM Plex Mono', monospace`;
      ctx.fillStyle = ours ? C.azure : C.ink55;
      const short = name.replace(" ensemble", " ens.");
      const tw = ctx.measureText(short).width;
      const tx = Math.min(Math.max(x - tw / 2, PADL + 2), W - PADR - tw - 2);
      ctx.fillText(short, tx, y - 12);
    }
  }

  function resize() {
    W = figEl.querySelector(".fig-body").clientWidth;
    H = Math.min(440, Math.max(320, Math.round(W * 0.46)));
    dpr = devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    draw();
  }

  canvas.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, bd = 20 * 20;
    for (const [name, bleu, flops] of pts()) {
      const d = (xOf(flops) - mx) ** 2 + (yOf(bleu) - my) ** 2;
      if (d < bd) { bd = d; best = [name, bleu, flops]; }
    }
    st.hover = best?.[0] ?? null;
    if (best) {
      readout.textContent = `${best[0]} · ${st.pair}: BLEU ${best[1]} · ${best[2].toExponential(1).replace("e+", "·10^")} FLOPs`;
      hud.value(`Table2[${best[0]}].${st.pair}`, best[1]);
    }
    draw();
  });
  canvas.addEventListener("pointerleave", () => { st.hover = null; draw(); });

  registerFigure(figEl, {
    start() { hud.set([{ label: "Table 2", dims: [9, 4] }], 0); },
    stop() { hud.idle(); },
  });
  resize();
  new ResizeObserver(resize).observe(figEl.querySelector(".fig-body"));
}

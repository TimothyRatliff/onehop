// Module 10 — Ablations (paper Table 3, newstest2013 dev set).
//
// Every number here is transcribed from Table 3. The reader picks which
// knob the paper varied; the bars show dev BLEU against the base
// setting's 25.8, with per-wordpiece perplexity alongside. The stories:
// single-head is 0.9 BLEU below the best; too many heads also hurts;
// shrinking d_k hurts; bigger models help; dropout matters a great deal.

import { registerFigure } from "../runtime.mjs";
import { hud } from "../hud.mjs";

const BASE = { label: "base", bleu: 25.8, ppl: 4.92 };
// rows: [label, BLEU, PPL]
const GROUPS = {
  "heads h": {
    note: "row (A): vary h with h·d_k fixed — compute held constant",
    rows: [["h=1", 24.9, 5.29], ["h=4", 25.5, 5.00], ["h=8 (base)", 25.8, 4.92],
           ["h=16", 25.8, 4.91], ["h=32", 25.4, 5.01]],
  },
  "key size d_k": {
    note: "row (B): shrinking d_k hurts — compatibility is not easy",
    rows: [["d_k=16", 25.1, 5.16], ["d_k=32", 25.4, 5.01], ["d_k=64 (base)", 25.8, 4.92]],
  },
  "depth N": {
    note: "row (C): bigger is better",
    rows: [["N=2", 23.7, 6.11], ["N=4", 25.3, 5.19], ["N=6 (base)", 25.8, 4.92], ["N=8", 25.5, 4.88]],
  },
  "width d_model": {
    note: "row (C): bigger is better",
    rows: [["256", 24.5, 5.75], ["512 (base)", 25.8, 4.92], ["1024", 26.0, 4.66]],
  },
  "ffn d_ff": {
    note: "row (C)",
    rows: [["1024", 25.4, 5.12], ["2048 (base)", 25.8, 4.92], ["4096", 26.2, 4.75]],
  },
  "dropout": {
    note: "row (D): dropout is very helpful in avoiding over-fitting",
    rows: [["P=0.0", 24.6, 5.77], ["P=0.1 (base)", 25.8, 4.92], ["P=0.2", 25.5, 4.95]],
  },
};

export function initAblations(figEl) {
  figEl.innerHTML = `
    <div class="badge">reported · Vaswani et al. 2017, Table 3 · dev BLEU, newstest2013</div>
    <div class="head-picker abl-picker" role="radiogroup" aria-label="which knob to vary"></div>
    <p class="abl-note"></p>
    <div class="abl-bars"></div>
    <div class="sdpa-readout" aria-live="polite">pick a knob — the base model holds every other value fixed</div>
    <figcaption>The paper's own one-thing-at-a-time sweep. The hairline
    is the base model's 25.8 BLEU; perplexities are per-wordpiece.
    The learned-position variant (row E) scored 25.7 — sinusoids cost
    nothing.</figcaption>`;

  const picker = figEl.querySelector(".abl-picker");
  const note = figEl.querySelector(".abl-note");
  const barsEl = figEl.querySelector(".abl-bars");
  const readout = figEl.querySelector(".sdpa-readout");
  const st = { group: "heads h" };

  for (const g of Object.keys(GROUPS)) {
    const b = document.createElement("button");
    b.className = "ctl-btn head-btn";
    b.textContent = g;
    b.setAttribute("role", "radio");
    b.addEventListener("click", () => { st.group = g; render(); });
    picker.append(b);
  }

  const YMIN = 23, YMAX = 26.6, HMAX = 190;
  const hOf = (b) => ((b - YMIN) / (YMAX - YMIN)) * HMAX;

  function render() {
    picker.querySelectorAll(".head-btn").forEach((b) => {
      const on = b.textContent === st.group;
      b.setAttribute("aria-checked", on);
      b.classList.toggle("on", on);
    });
    const g = GROUPS[st.group];
    note.textContent = g.note;
    barsEl.innerHTML = "";
    barsEl.style.setProperty("--baseline", `${hOf(BASE.bleu)}px`);
    for (const [label, bleu, ppl] of g.rows) {
      const isBase = label.includes("base");
      const col = document.createElement("button");
      col.className = "abl-col" + (isBase ? " abl-base" : "");
      col.innerHTML = `
        <span class="abl-val">${bleu.toFixed(1)}</span>
        <i style="height:${hOf(bleu).toFixed(0)}px"></i>
        <span class="abl-lab">${label}</span>
        <span class="abl-ppl">ppl ${ppl.toFixed(2)}</span>`;
      const show = () => {
        const d = bleu - BASE.bleu;
        readout.textContent =
          `${st.group} ${label}: BLEU ${bleu.toFixed(1)} (${d === 0 ? "base" : (d > 0 ? "+" : "") + d.toFixed(1) + " vs base"}) · perplexity ${ppl.toFixed(2)}`;
        hud.value(`Table3[${st.group}][${label}].BLEU`, bleu);
      };
      col.addEventListener("pointerenter", show);
      col.addEventListener("focus", show);
      col.addEventListener("click", show);
      barsEl.append(col);
    }
  }

  registerFigure(figEl, {
    start() { hud.set([{ label: "Table 3", dims: [6, 5] }], 0); },
    stop() { hud.idle(); },
  });
  render();
}

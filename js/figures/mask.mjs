// Module 5 — The mask (paper S3.2.3).
//
// Decoder self-attention with the causal mask as a live grid: rows are
// query positions, columns key positions, upper triangle is the future.
// With the mask on, every future score is the -inf sentinel and the
// softmax gives EXACTLY zero — not a small number, zero. The reader
// toggles the mask off and watches two things at once: weight flooding
// into the upper triangle, and the prediction row underneath falling
// apart.
//
// Empirical honesty note: this model was TRAINED with the mask, so
// removing it at inference does not make it "cheat" by copying the
// visible answer — it knocks the activations out of distribution and
// the decoder collapses (it predicts end-of-sequence almost everywhere).
// A model trained WITHOUT the mask would learn the copying shortcut the
// paper guards against; this one simply breaks. The caption says exactly
// that, because the two failures teach the same lesson from different
// directions: the mask is what makes the training task honest.

import { encodeText, idsToText, encodeSrc, decoderPass, projectLogits, greedyDecode } from "../model.mjs";
import { hud } from "../hud.mjs";
import { registerFigure } from "../runtime.mjs";

export function initMask(figEl, model) {
  const cfg = model.config;
  figEl.innerHTML = `
    <div class="badge badge-live">live · toy model · decoder self-attention</div>
    <div class="sdpa-top">
      <label class="sdpa-in">input
        <input type="text" value="3 march 2012" maxlength="${cfg.max_src}"
               spellcheck="false" aria-label="date input to the encoder">
      </label>
      <div class="mask-ctls">
        <div class="head-picker" role="radiogroup" aria-label="attention head"></div>
        <button class="ctl-btn mask-toggle" aria-pressed="true"></button>
      </div>
    </div>
    <div class="mask-grid" role="img"></div>
    <div class="mask-preds"></div>
    <div class="sdpa-readout" aria-live="polite">hover any cell — masked cells hold the −∞ sentinel and softmax to exactly 0</div>
    <figcaption>Decoder self-attention over the model's own output
    (layer 1). Rows attend to columns; the upper-right triangle is the
    future. This model was trained under the mask, so switching the mask
    off does not let it cheat — it breaks it: the activations leave the
    distribution it learned and the prediction row collapses to
    end-of-sequence. A model <em>trained</em> without the mask fails in
    the opposite direction, by copying the shifted target it can see.
    Either way, the mask is what keeps the training task honest.</figcaption>`;

  const input = figEl.querySelector("input");
  const gridEl = figEl.querySelector(".mask-grid");
  const predsEl = figEl.querySelector(".mask-preds");
  const readout = figEl.querySelector(".sdpa-readout");
  const headPicker = figEl.querySelector(".head-picker");
  const toggle = figEl.querySelector(".mask-toggle");

  const st = { h: 3, causal: true, srcIds: [], tgtIn: [], tgtText: "",
               traces: { masked: null, unmasked: null }, visible: false };

  for (let h = 0; h < 4; h++) {
    const b = document.createElement("button");
    b.className = "ctl-btn head-btn";
    b.textContent = `head ${h + 1}`;
    b.setAttribute("role", "radio");
    b.addEventListener("click", () => { st.h = h; render(); });
    headPicker.append(b);
  }
  const setToggle = () => {
    toggle.textContent = st.causal ? "mask on · causal" : "mask off · future visible";
    toggle.setAttribute("aria-pressed", st.causal);
    toggle.classList.toggle("off-verm", !st.causal);
  };
  toggle.addEventListener("click", () => { st.causal = !st.causal; render(); setToggle(); });
  setToggle();

  function recompute() {
    st.srcIds = encodeText(cfg, input.value);
    if (!st.srcIds.length) { st.traces.masked = null; return; }
    // Teacher-force the model's own (masked) greedy output, so both grids
    // score the same sequence and only the mask differs.
    const g = greedyDecode(model, st.srcIds);
    st.tgtText = g.text;
    st.tgtIn = [cfg.bos_id, ...g.ids];
    const enc = encodeSrc(model, st.srcIds);
    for (const causal of [true, false]) {
      const dec = decoderPass(model, st.tgtIn, enc.out, { causal });
      const logits = projectLogits(model, dec.out);
      st.traces[causal ? "masked" : "unmasked"] = { dec, logits };
    }
  }

  const charAt = (i) =>
    st.tgtIn[i] === cfg.bos_id ? "⟨s⟩" : cfg.vocab[st.tgtIn[i]];

  function render() {
    if (!st.traces.masked) { gridEl.innerHTML = ""; predsEl.innerHTML = ""; return; }
    headPicker.querySelectorAll(".head-btn").forEach((b, i) => {
      b.setAttribute("aria-checked", i === st.h);
      b.classList.toggle("on", i === st.h);
    });
    const T = st.tgtIn.length;
    const tr = st.traces[st.causal ? "masked" : "unmasked"];
    const at = tr.dec.layers[0].self_attn;
    const W = at.weights[st.h], S = at.scores[st.h];

    gridEl.style.setProperty("--n", T + 1);
    predsEl.style.setProperty("--n", T + 1);
    gridEl.innerHTML = "";
    const corner = document.createElement("span");
    corner.className = "mg-lab";
    gridEl.append(corner);
    for (let j = 0; j < T; j++) {
      const l = document.createElement("span");
      l.className = "mg-lab";
      l.textContent = charAt(j);
      gridEl.append(l);
    }
    for (let t = 0; t < T; t++) {
      const l = document.createElement("span");
      l.className = "mg-lab mg-row";
      l.textContent = charAt(t);
      gridEl.append(l);
      for (let j = 0; j < T; j++) {
        const c = document.createElement("span");
        const masked = st.causal && j > t;
        const w = W[t][j];
        c.className = "mg-cell" + (masked ? " mg-masked" : "");
        if (masked) {
          c.textContent = "−∞";
        } else {
          c.style.background = `rgba(0,114,178,${(0.04 + 0.96 * w).toFixed(3)})`;
          if (w > 0.995) { c.textContent = "1"; c.classList.add("mg-one"); }
        }
        const show = () => {
          readout.textContent = masked
            ? `dec0.self.scores[${st.h}][${t}][${j}] = −∞ (sentinel ${cfg.neg_inf.toExponential(0)}) → weight = 0 exactly`
            : `dec0.self.weights[${st.h}][${t}][${j}] = ${w.toFixed(6)}  ·  score ${S[t][j].toFixed(3)}`;
          hud.value(`dec0.self.weights[${st.h}][${t}][${j}]`, masked ? 0 : w);
          hud.active(st.causal ? 2 : 1);
        };
        c.addEventListener("pointerenter", show);
        c.addEventListener("click", show);
        gridEl.append(c);
      }
    }

    // prediction row vs what the model should produce
    const preds = tr.logits.map((row) => {
      let m = 0;
      for (let i = 1; i < row.length; i++) if (row[i] > row[m]) m = i;
      return m;
    });
    const predChar = (i) =>
      preds[i] === cfg.eos_id ? "⟨/s⟩" : (cfg.vocab[preds[i]] ?? "·");
    const want = [...st.tgtText.split(""), "⟨/s⟩"];
    predsEl.innerHTML = `
      <div class="mask-predrow"><span class="mg-lab">predicts</span>${
        preds.map((_, i) => {
          const ok = predChar(i) === (want[i] ?? "");
          return `<span class="mp ${ok ? "" : "mp-bad"}">${predChar(i)}</span>`;
        }).join("")
      }</div>
      <div class="mask-predrow dim"><span class="mg-lab">should be</span>${
        want.map((c) => `<span class="mp">${c}</span>`).join("")
      }</div>`;

    if (st.visible) hudSet();
  }

  function hudSet() {
    const T = st.tgtIn.length;
    hud.set([
      { label: "x", dims: [T, cfg.d_model] },
      { label: "scores", dims: [4, T, T] },
      { label: st.causal ? "mask+softmax" : "softmax (no mask)", dims: [4, T, T] },
      { label: "out", dims: [T, cfg.d_model] },
    ], 2);
  }

  input.addEventListener("input", () => { recompute(); render(); });
  registerFigure(figEl, {
    start() { st.visible = true; hudSet(); },
    stop() { st.visible = false; hud.idle(); },
  });
  recompute();
  render();
}

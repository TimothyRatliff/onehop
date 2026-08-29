// Module 6 — Multi-head attention (paper S3.2.2).
//
// All four heads of one attention block, running on the reader's input,
// each map in its own hue, converging through concat and W^O. The
// checkboxes zero a head's OUTPUT before the concat — its map still
// shows where it looked; its contribution is simply discarded — and the
// whole decode reruns with that head silenced, so the output line shows
// the real damage (or the real redundancy).
//
// With a real model this is not a metaphor. On the default input,
// silencing head 3 of the decoder's second cross-attention makes the
// model read "september" as month 02: that head is where this model
// keeps its month-words.

import { encodeText, greedyDecode } from "../model.mjs";
import { hud } from "../hud.mjs";
import { registerFigure } from "../runtime.mjs";

const HUES = [
  ["azure", "0,114,178"],
  ["vermilion", "213,94,0"],
  ["moss", "0,158,115"],
  ["plum", "204,121,167"],
];
const BLOCKS = [
  ["enc0.attn", "enc L1"],
  ["enc1.attn", "enc L2"],
  ["dec0.self", "dec L1 self"],
  ["dec1.self", "dec L2 self"],
  ["dec0.cross", "dec L1 cross"],
  ["dec1.cross", "dec L2 cross"],
];

export function initHeads(figEl, model) {
  const cfg = model.config;
  figEl.innerHTML = `
    <div class="badge badge-live">live · toy model · every map recomputed on each change</div>
    <div class="sdpa-top">
      <label class="sdpa-in">input
        <input type="text" value="17 september 1930" maxlength="${cfg.max_src}"
               spellcheck="false" aria-label="date input">
      </label>
      <div class="head-picker hp-blocks" role="radiogroup" aria-label="attention block"></div>
    </div>
    <div class="heads-row"></div>
    <svg class="heads-concat" aria-label="concat and output projection"></svg>
    <div class="heads-out"></div>
    <div class="sdpa-readout" aria-live="polite">hover any map to read exact weights · uncheck a head to silence it</div>
    <figcaption>Four heads of one block, each looking somewhere different
    on your input. Silencing a head does not blind it — its map is
    unchanged — it discards its output at the concat. Some heads are
    load-bearing and some are redundant; the output line tells you which,
    live. Try the decoder blocks: this model keeps its month-words in one
    head and its year digits in another.</figcaption>`;

  const input = figEl.querySelector("input");
  const blockPicker = figEl.querySelector(".hp-blocks");
  const headsRow = figEl.querySelector(".heads-row");
  const concatSvg = figEl.querySelector(".heads-concat");
  const outEl = figEl.querySelector(".heads-out");
  const readout = figEl.querySelector(".sdpa-readout");

  const st = { block: "dec1.cross", on: [true, true, true, true],
               srcIds: [], full: null, run: null, visible: false };

  for (const [key, label] of BLOCKS) {
    const b = document.createElement("button");
    b.className = "ctl-btn head-btn";
    b.textContent = label;
    b.setAttribute("role", "radio");
    b.addEventListener("click", () => { st.block = key; recompute(); render(); });
    blockPicker.append(b);
  }

  // four head cards
  const cards = HUES.map(([name, rgb], h) => {
    const card = document.createElement("div");
    card.className = "head-card";
    card.innerHTML = `
      <label class="head-onoff" style="color: var(--${name})">
        <input type="checkbox" checked aria-label="head ${h + 1} enabled">
        head ${h + 1}
      </label>
      <canvas></canvas>`;
    card.querySelector("input").addEventListener("change", (e) => {
      st.on[h] = e.target.checked;
      recompute();
      render();
    });
    headsRow.append(card);
    return { card, canvas: card.querySelector("canvas"), rgb, name };
  });

  function recompute() {
    st.srcIds = encodeText(cfg, input.value);
    if (!st.srcIds.length) { st.run = null; return; }
    st.full = greedyDecode(model, st.srcIds);
    const masks = st.on.every(Boolean) ? undefined
      : { [st.block]: st.on.map((v) => (v ? 1 : 0)) };
    st.run = masks ? greedyDecode(model, st.srcIds, null, { headMasks: masks }) : st.full;
  }

  // which trace holds the chosen block's attention, and its axis labels
  function blockView(run) {
    const enc = st.block.startsWith("enc");
    const layer = +st.block[3];
    const srcChars = [...st.srcIds.map((i) => cfg.vocab[i])];
    const decChars = ["⟨s⟩", ...run.ids.map((i) => cfg.vocab[i])];
    if (enc) {
      return { at: run.encoder.layers[layer].self_attn,
               rows: srcChars, cols: srcChars };
    }
    const lastDec = run.steps[run.steps.length - 1].decoder;
    const at = st.block.endsWith("self")
      ? lastDec.layers[layer].self_attn
      : lastDec.layers[layer].cross_attn;
    return { at, rows: decChars, cols: st.block.endsWith("self") ? decChars : srcChars };
  }

  function drawMap(h, view) {
    const { canvas, rgb } = cards[h];
    const W = cards[h].card.clientWidth;
    const nR = view.rows.length, nC = view.cols.length;
    const cell = Math.min(16, (W - 18) / nC);
    const w = 18 + nC * cell, ht = 14 + nR * cell;
    const dpr = devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = ht * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${ht}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, ht);
    const dead = !st.on[h];
    const weights = view.at.weights[h];
    for (let r = 0; r < nR; r++) {
      for (let c = 0; c < nC; c++) {
        const wv = weights[r][c];
        ctx.fillStyle = `rgba(${rgb},${((0.04 + 0.96 * wv) * (dead ? 0.22 : 1)).toFixed(3)})`;
        ctx.fillRect(18 + c * cell, 14 + r * cell, cell - 0.75, cell - 0.75);
      }
    }
    ctx.fillStyle = dead ? "rgba(34,30,25,0.25)" : "rgba(34,30,25,0.55)";
    ctx.font = "400 8.5px 'IBM Plex Mono', monospace";
    if (cell >= 9) {
      view.cols.forEach((ch, c) =>
        ctx.fillText(ch === " " ? "␣" : ch[0], 18 + c * cell + cell / 2 - 2.5, 10));
      view.rows.forEach((ch, r) =>
        ctx.fillText(ch === " " ? "␣" : ch[0], 5, 14 + r * cell + cell / 2 + 3));
    }
    if (dead) {
      ctx.fillStyle = "rgba(34,30,25,0.55)";
      ctx.font = "500 11px 'IBM Plex Mono', monospace";
      ctx.fillText("output zeroed", 20, ht - 6);
    }
    // hover -> exact weight
    canvas.onpointermove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const c = Math.floor((e.clientX - rect.left - 18) / cell);
      const r = Math.floor((e.clientY - rect.top - 14) / cell);
      if (r < 0 || r >= nR || c < 0 || c >= nC) return;
      const wv = weights[r][c];
      readout.textContent =
        `${st.block}.weights[${h}][${r}][${c}] = ${wv.toFixed(6)}  ·  ` +
        `"${view.rows[r]}" attends to "${view.cols[c]}"` +
        (st.on[h] ? "" : "  ·  head silenced: it still looks, its output is discarded");
      hud.value(`${st.block}.weights[${h}][${r}][${c}]`, wv);
      hud.active(0);
    };
  }

  function drawConcat() {
    const W = Math.min(620, figEl.clientWidth - 20);
    const H = 66;
    concatSvg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    concatSvg.style.width = `${W}px`;
    concatSvg.style.height = `${H}px`;
    concatSvg.style.display = "block";
    concatSvg.style.margin = "14px auto 0";
    const seg = W / 4 - 8;
    let s = "";
    HUES.forEach(([name], h) => {
      const x = h * (seg + 10.67);
      const on = st.on[h];
      s += `<rect x="${x}" y="4" width="${seg}" height="12" rx="2"
        fill="${on ? `var(--${name})` : "none"}"
        stroke="${on ? "none" : "var(--ink-25)"}" stroke-dasharray="${on ? "" : "3 3"}"/>`;
      s += `<line x1="${x + seg / 2}" y1="20" x2="${W / 2}" y2="38"
        stroke="var(--ink-25)" stroke-width="1"/>`;
    });
    s += `<rect x="${W / 4}" y="40" width="${W / 2}" height="10" rx="2" fill="var(--ink-25)"/>`;
    s += `<text x="${W / 2}" y="63" text-anchor="middle"
      style="font: 400 11px 'IBM Plex Mono', monospace; fill: var(--ink-55)">concat → W^O → ${st.run ? st.run.ids.length + 1 : ""}×48</text>`;
    concatSvg.innerHTML = s;
  }

  function render() {
    if (!st.run) { outEl.innerHTML = ""; return; }
    blockPicker.querySelectorAll(".head-btn").forEach((b, i) => {
      const onB = BLOCKS[i][0] === st.block;
      b.setAttribute("aria-checked", onB);
      b.classList.toggle("on", onB);
    });
    cards.forEach((c, h) => { c.card.querySelector("input").checked = st.on[h]; });
    const view = blockView(st.run);
    for (let h = 0; h < 4; h++) drawMap(h, view);
    drawConcat();

    const fullText = st.full.text;
    const runText = st.run.text;
    const n = Math.max(fullText.length, runText.length);
    let diff = "";
    for (let i = 0; i < n; i++) {
      const ch = runText[i] ?? "·";
      diff += `<span class="mp ${ch === (fullText[i] ?? "") ? "" : "mp-bad"}">${ch}</span>`;
    }
    const changed = runText !== fullText;
    outEl.innerHTML = `
      <div class="heads-outrow"><span class="mg-lab">output</span><span class="mp-seq">${diff}</span>
        <span class="heads-verdict ${changed ? "bad" : ""}">${
          st.on.every(Boolean) ? "all heads on" : changed ? "degraded" : "unchanged — redundant here"
        }</span></div>
      ${changed ? `<div class="heads-outrow dim"><span class="mg-lab">all heads</span><span class="mp-seq">${
        [...fullText].map((c) => `<span class="mp">${c}</span>`).join("")
      }</span></div>` : ""}`;

    if (st.visible) {
      const T = view.rows.length;
      hud.set([
        { label: st.block, dims: [4, T, cfg.d_k] },
        { label: "concat", dims: [T, cfg.d_model] },
        { label: "·W^O", dims: [T, cfg.d_model] },
      ], 0);
    }
  }

  input.addEventListener("input", () => { recompute(); render(); });
  registerFigure(figEl, {
    start() { st.visible = true; render(); },
    stop() { st.visible = false; hud.idle(); },
  });
  recompute();
  render();
  new ResizeObserver(() => st.run && render()).observe(headsRow);
}

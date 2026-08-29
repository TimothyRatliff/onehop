// Module 2 — Scaled dot-product attention (paper S3.2.1, eq. 1).
//
// The reader types a date and picks one character and one head. The figure
// unrolls Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V for that query:
//
//   x  ->  q = x W^Q          (12 dims, one head's slice)
//      ->  q.k_j / sqrt(d_k)  (one score per source position)
//      ->  softmax            (weights, drawn as bars AND as arcs over
//                              the input text)
//      ->  sum_j w_j v_j      (the output vector)
//
// Every number is pulled from the live encoder trace (layer 1) — the same
// trace test/parity.mjs verifies against PyTorch. Hovering or focusing any
// cell publishes its fully-qualified name and value to the tensor HUD.

import { encodeText, encodeSrc } from "../model.mjs";
import { hud } from "../hud.mjs";
import { registerFigure, probeVerb } from "../runtime.mjs";

const NHEADS = 4;

export function initSDPA(figEl, model) {
  const cfg = model.config;
  figEl.innerHTML = `
    <div class="badge badge-live">live · toy model · encoder layer 1</div>
    <div class="sdpa-top">
      <label class="sdpa-in">input
        <input type="text" value="the third of march 2012" maxlength="${cfg.max_src}"
               spellcheck="false" aria-label="date input to the encoder">
      </label>
      <div class="head-picker" role="radiogroup" aria-label="attention head"></div>
    </div>
    <div class="sdpa-stage sdpa-tokens">
      <span class="stage-label">x · one char each</span>
      <div class="tokwrap">
        <svg class="sdpa-arcs" aria-hidden="true"></svg>
        <div class="sdpa-tokrow" role="radiogroup" aria-label="query position"></div>
      </div>
    </div>
    <div class="sdpa-stages"></div>
    <div class="sdpa-readout" aria-live="polite">${probeVerb()} any cell to read its exact value</div>
    <figcaption>One head of the first encoder layer, unrolled for the
    character you select. The arcs and the bars are the same numbers —
    the softmax row of equation 1.</figcaption>`;

  const input = figEl.querySelector("input");
  const tokrow = figEl.querySelector(".sdpa-tokrow");
  const arcsSvg = figEl.querySelector(".sdpa-arcs");
  const stagesEl = figEl.querySelector(".sdpa-stages");
  const readout = figEl.querySelector(".sdpa-readout");
  const headPicker = figEl.querySelector(".head-picker");

  // Defaults chosen for a visibly mixed softmax (head 4 splits its weight
  // across several positions here), not a one-hot.
  const st = { ids: [], chars: [], t: 0, h: 3, trace: null, visible: false };

  // ---------------------------------------------------------- head picker
  for (let h = 0; h < NHEADS; h++) {
    const b = document.createElement("button");
    b.className = "ctl-btn head-btn";
    b.textContent = `head ${h + 1}`;
    b.setAttribute("role", "radio");
    b.addEventListener("click", () => {
      st.h = h;
      render();
    });
    headPicker.append(b);
  }

  // ---------------------------------------------------------- stages
  const STAGE_DEFS = [
    { key: "q", label: "q = x·W^Q", n: () => cfg.d_k },
    { key: "scores", label: "q·kⱼ / √d_k", n: () => st.ids.length },
    { key: "weights", label: "softmax", n: () => st.ids.length },
    { key: "out", label: "Σ wⱼ·vⱼ", n: () => cfg.d_k },
  ];
  for (const def of STAGE_DEFS) {
    const row = document.createElement("div");
    row.className = "sdpa-stage";
    row.dataset.stage = def.key;
    row.innerHTML = `<span class="stage-label">${def.label}</span>
      <div class="strip" tabindex="0" role="group" aria-label="${def.label}"></div>`;
    stagesEl.append(row);
  }

  function hudStages() {
    const T = st.ids.length;
    return [
      { label: "x", dims: [T, cfg.d_model] },
      { label: "q", dims: [1, cfg.d_k] },
      { label: "q·kᵀ/√d_k", dims: [1, T] },
      { label: "softmax", dims: [1, T] },
      { label: "Σw·v", dims: [1, cfg.d_k] },
    ];
  }

  // ---------------------------------------------------------- data
  function recompute() {
    st.chars = [...input.value.toLowerCase()].filter((c) =>
      cfg.vocab.includes(c));
    st.ids = st.chars.map((c) => cfg.vocab.indexOf(c));
    if (st.ids.length === 0) {
      st.trace = null;
      return;
    }
    st.t = Math.min(st.t, st.ids.length - 1);
    st.trace = encodeSrc(model, st.ids);
  }

  // ---------------------------------------------------------- painting
  // Signed value -> paper..azure for +, paper..vermilion for −.
  function cellColor(v, vmax) {
    const a = Math.min(1, Math.abs(v) / (vmax || 1));
    const hue = v >= 0 ? "0,114,178" : "213,94,0";
    return `rgba(${hue},${(0.08 + 0.92 * a).toFixed(3)})`;
  }

  function fillStrip(stripEl, values, names, opts = {}) {
    const vmax = Math.max(...values.map(Math.abs));
    stripEl.innerHTML = "";
    stripEl.style.setProperty("--cells", values.length);
    values.forEach((v, i) => {
      const c = document.createElement("span");
      c.className = "cell" + (opts.bars ? " bar-cell" : "");
      if (opts.bars) {
        const b = document.createElement("i");
        b.style.height = `${Math.round(44 * (v / vmax))}px`;
        c.append(b);
        if (values.length <= 16) {
          const t = document.createElement("em");
          t.textContent = v.toFixed(2).replace("0.", ".");
          c.append(t);
        }
      } else {
        c.style.background = cellColor(v, vmax);
      }
      if (opts.mark === i) c.classList.add("sel");
      const show = () => {
        readout.textContent = `${names(i)} = ${v.toFixed(6)}`;
        hud.value(names(i), v);
        hud.active(opts.hudStage);
      };
      c.addEventListener("pointerenter", show);
      c.addEventListener("click", show);
      stripEl.append(c);
    });
    // Keyboard: the strip is one tab stop; arrows walk an inspection cursor.
    stripEl.onkeydown = (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      const cur = +(stripEl.dataset.cursor ?? -1);
      const nxt = Math.max(0, Math.min(values.length - 1,
        cur + (e.key === "ArrowRight" ? 1 : -1)));
      stripEl.dataset.cursor = nxt;
      [...stripEl.children].forEach((el, i) =>
        el.classList.toggle("kbd-cursor", i === nxt));
      readout.textContent = `${names(nxt)} = ${values[nxt].toFixed(6)}`;
      hud.value(names(nxt), values[nxt]);
    };
  }

  function drawArcs(weights) {
    const toks = [...tokrow.children];
    if (!toks.length) return;
    const rowRect = tokrow.getBoundingClientRect();
    const x = (i) => {
      const r = toks[i].getBoundingClientRect();
      return r.left - rowRect.left + r.width / 2;
    };
    const H = 46;
    arcsSvg.setAttribute("viewBox", `0 0 ${rowRect.width} ${H}`);
    arcsSvg.style.height = `${H}px`;
    const wmax = Math.max(...weights);
    arcsSvg.innerHTML = weights
      .map((w, j) => {
        if (w < 0.004 || j === st.t) return "";
        const lift = Math.min(40, 8 + Math.abs(x(j) - x(st.t)) * 0.12);
        return `<path d="M ${x(st.t)} ${H - 2} Q ${(x(st.t) + x(j)) / 2} ${H - 2 - lift} ${x(j)} ${H - 2}"
          fill="none" stroke="var(--azure)" stroke-opacity="${(0.15 + 0.85 * w / wmax).toFixed(3)}"
          stroke-width="${(0.75 + 2.5 * w / wmax).toFixed(2)}"/>`;
      })
      .join("");
  }

  function render() {
    if (!st.trace) {
      stagesEl.querySelectorAll(".strip").forEach((s) => (s.innerHTML = ""));
      arcsSvg.innerHTML = "";
      tokrow.innerHTML = "";
      readout.textContent = "type a date above — the vocabulary is digits, letters, space, and , - . /";
      return;
    }
    const at = st.trace.layers[0].self_attn;
    const h = st.h, t = st.t;

    headPicker.querySelectorAll(".head-btn").forEach((b, i) => {
      b.setAttribute("aria-checked", i === h);
      b.classList.toggle("on", i === h);
    });

    // token row
    tokrow.innerHTML = "";
    st.chars.forEach((ch, i) => {
      const b = document.createElement("button");
      b.className = "tok" + (i === t ? " sel" : "");
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", i === t);
      b.setAttribute("aria-label", `query position ${i + 1}: "${ch}"`);
      b.textContent = ch === " " ? "␣" : ch;
      b.addEventListener("click", () => {
        st.t = i;
        render();
      });
      tokrow.append(b);
    });
    tokrow.onkeydown = (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      st.t = Math.max(0, Math.min(st.ids.length - 1,
        st.t + (e.key === "ArrowRight" ? 1 : -1)));
      render();
      tokrow.children[st.t].focus();
    };

    const strips = stagesEl.querySelectorAll(".strip");
    fillStrip(strips[0], at.q[h][t],
      (i) => `enc0.attn.q[${h}][${t}][${i}]`, { hudStage: 1 });
    fillStrip(strips[1], at.scores[h][t],
      (j) => `enc0.attn.scores[${h}][${t}][${j}]`, { hudStage: 2, mark: t });
    fillStrip(strips[2], at.weights[h][t],
      (j) => `enc0.attn.weights[${h}][${t}][${j}]`,
      { hudStage: 3, bars: true, mark: t });
    fillStrip(strips[3], at.heads[h][t],
      (i) => `enc0.attn.heads[${h}][${t}][${i}]`, { hudStage: 4 });

    drawArcs(at.weights[h][t]);
    if (st.visible) hud.set(hudStages(), 3);
  }

  input.addEventListener("input", () => {
    recompute();
    render();
  });

  // ---------------------------------------------------------- lifecycle
  // No animation loop — this figure is recompute-on-interact — but entering
  // the viewport points the HUD at it, and leaving returns the HUD to idle.
  registerFigure(figEl, {
    start() {
      st.visible = true;
      if (st.trace) hud.set(hudStages(), 3);
    },
    stop() {
      st.visible = false;
      hud.idle();
    },
  });

  recompute();
  render();
  new ResizeObserver(() => st.trace && drawArcs(
    st.trace.layers[0].self_attn.weights[st.h][st.t])).observe(tokrow);
}

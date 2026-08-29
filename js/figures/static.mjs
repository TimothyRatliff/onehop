// Tier 3 — precise static figures (modules 12–15). No interaction by
// design: each is a redrawing of the paper's own material in the site's
// visual language. All numbers are the paper's (reported), never the toy
// model's.

// Shared bits ---------------------------------------------------------

const T = (x, y, text, opts = {}) =>
  `<text x="${x}" y="${y}" ${opts.anchor ? `text-anchor="${opts.anchor}"` : ""}
     style="font: ${opts.w ?? 400} ${opts.s ?? 11}px 'IBM Plex Mono', monospace;
     fill: var(--${opts.c ?? "ink-55"})">${text}</text>`;

const BOX = (x, y, w, h, hue, label, sub) => `
  <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3"
    fill="${hue ? `color-mix(in srgb, var(--${hue}) 13%, transparent)` : "none"}"
    stroke="var(--${hue ?? "ink-25"})" stroke-width="1"/>
  ${T(x + w / 2, y + h / 2 + (sub ? -2 : 4), label, { anchor: "middle", c: "ink", w: 500 })}
  ${sub ? T(x + w / 2, y + h / 2 + 12, sub, { anchor: "middle" }) : ""}`;

const ARROW = (x1, y1, x2, y2) => `
  <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"
    stroke="var(--ink-25)" stroke-width="1" marker-end="url(#arr)"/>`;

const DEFS = `<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4"
  markerWidth="7" markerHeight="7" orient="auto">
  <path d="M0 0 L8 4 L0 8 z" fill="var(--ink-25)"/></marker></defs>`;

function svgFig(figEl, badge, viewBox, inner, caption, maxW = 820) {
  figEl.innerHTML = `
    <div class="badge">${badge}</div>
    <svg viewBox="${viewBox}" style="display:block;width:100%;max-width:${maxW}px;margin:0 auto"
         role="img">${DEFS}${inner}</svg>
    <figcaption>${caption}</figcaption>`;
}

// Module 12 — the stack (Figure 1 redrawn) ----------------------------

export function initStack(figEl) {
  const enc = (x) => {
    let s = "";
    s += BOX(x, 320, 190, 44, "azure", "multi-head", "self-attention");
    s += T(x + 205, 342, "add & norm");
    s += BOX(x, 240, 190, 44, "moss", "feed-forward", "d_ff 2048");
    s += T(x + 205, 262, "add & norm");
    // residual skips
    s += `<path d="M ${x - 14} 392 C ${x - 34} 370 ${x - 34} 320 ${x - 14} 342" fill="none" stroke="var(--ink-25)"/>`;
    s += `<path d="M ${x - 14} 312 C ${x - 34} 290 ${x - 34} 240 ${x - 14} 262" fill="none" stroke="var(--ink-25)"/>`;
    s += ARROW(x + 95, 320, x + 95, 288);
    return s;
  };
  const dec = (x) => {
    let s = "";
    s += BOX(x, 320, 190, 44, "azure", "masked multi-head", "self-attention");
    s += T(x + 205, 342, "add & norm");
    s += BOX(x, 240, 190, 44, "plum", "multi-head", "over encoder output");
    s += T(x + 205, 262, "add & norm");
    s += BOX(x, 160, 190, 44, "moss", "feed-forward", "d_ff 2048");
    s += T(x + 205, 182, "add & norm");
    s += `<path d="M ${x - 14} 392 C ${x - 34} 370 ${x - 34} 320 ${x - 14} 342" fill="none" stroke="var(--ink-25)"/>`;
    s += `<path d="M ${x - 14} 312 C ${x - 34} 290 ${x - 34} 240 ${x - 14} 262" fill="none" stroke="var(--ink-25)"/>`;
    s += `<path d="M ${x - 14} 232 C ${x - 34} 210 ${x - 34} 160 ${x - 14} 182" fill="none" stroke="var(--ink-25)"/>`;
    s += ARROW(x + 95, 320, x + 95, 288);
    s += ARROW(x + 95, 240, x + 95, 208);
    return s;
  };
  const inner = `
    ${T(155, 20, "encoder", { c: "ink", w: 500, s: 13 })}${T(505, 20, "decoder", { c: "ink", w: 500, s: 13 })}
    ${enc(60)}${dec(440)}
    ${T(292, 300, "× 6", { c: "ink", w: 500, s: 14 })}
    ${T(672, 220, "× 6", { c: "ink", w: 500, s: 14 })}
    <path d="M 155 236 C 155 120 340 300 426 262" fill="none" stroke="var(--plum)"
      stroke-width="1.5" marker-end="url(#arr)"/>
    ${T(285, 158, "keys and values", { c: "plum" })}
    ${BOX(440, 96, 190, 30, null, "linear → softmax")}
    ${ARROW(535, 160, 535, 130)}
    ${T(535, 76, "output probabilities", { anchor: "middle", c: "ink" })}
    ${BOX(60, 420, 190, 30, null, "embed + positional enc.")}
    ${BOX(440, 420, 190, 30, null, "embed + positional enc.")}
    ${ARROW(155, 420, 155, 368)}
    ${ARROW(535, 420, 535, 368)}
    ${T(155, 478, "inputs", { anchor: "middle", c: "ink" })}
    ${T(535, 478, "outputs, shifted right one position", { anchor: "middle", c: "ink" })}
    ${T(345, 445, "d_model 512", { anchor: "middle" })}`;
  svgFig(figEl,
    "reported · the paper's model — figure 1 redrawn",
    "0 0 720 492", inner,
    `Six identical layers per stack, every sub-layer wrapped in a
     residual connection (the curved bypasses) and layer normalization.
     The decoder's middle sub-layer attends over the encoder's output —
     the third use of attention. The output embedding is shifted right so
     position i may only depend on outputs before i.`);
}

// Module 13 — the feed-forward sub-layer ------------------------------

export function initFFN(figEl) {
  const inner = `
    ${[0, 1, 2].map((i) => `
      <circle cx="40" cy="${70 + i * 44}" r="4" fill="var(--ink-25)"/>
      ${ARROW(52, 70 + i * 44, 118, 128)}`).join("")}
    ${T(40, 210, "every", { anchor: "middle" })}${T(40, 224, "position", { anchor: "middle" })}
    ${BOX(120, 100, 96, 56, null, "x", "1×512")}
    ${ARROW(216, 128, 268, 128)}
    ${T(242, 118, "W₁", { anchor: "middle", c: "ink" })}
    ${BOX(270, 76, 200, 104, "moss", "max(0, xW₁ + b₁)", "1×2048 · ReLU")}
    ${ARROW(470, 128, 522, 128)}
    ${T(496, 118, "W₂", { anchor: "middle", c: "ink" })}
    ${BOX(524, 100, 96, 56, null, "out", "1×512")}
    ${T(370, 250, "same W₁, W₂ at every position within a layer; different across layers", { anchor: "middle", c: "ink" })}
    ${T(370, 268, "equivalently: two convolutions with kernel size 1", { anchor: "middle" })}`;
  svgFig(figEl,
    "reported · §3.3, d_ff = 2048",
    "0 0 720 290", inner,
    `The other half of every layer: each position, independently, is
     expanded four-fold, rectified, and projected back. All the mixing
     between positions happens in attention; all of this is per-position.`);
}

// Module 14 — the training recipe -------------------------------------

export function initRecipe(figEl) {
  const rows = [
    ["optimizer", "Adam · β₁ 0.9 · β₂ 0.98 · ε 10⁻⁹"],
    ["schedule", "linear warmup 4000 steps, then step⁻⁰·⁵ decay (§9 above)"],
    ["regularization", "residual dropout P_drop 0.1 · label smoothing ε_ls 0.1"],
    ["label smoothing", "hurts perplexity, improves accuracy and BLEU"],
    ["hardware", "8 × NVIDIA P100"],
    ["base model", "100K steps ≈ 12 hours at ~0.4 s/step · 65M parameters"],
    ["big model", "300K steps ≈ 3.5 days at ~1.0 s/step · 213M parameters"],
  ];
  figEl.innerHTML = `
    <div class="badge">reported · §5, Table 3 bottom rows</div>
    <div class="recipe">
      ${rows.map(([k, v]) => `<div class="recipe-row"><span class="recipe-k">${k}</span><span class="recipe-v">${v}</span></div>`).join("")}
    </div>
    <figcaption>The entire recipe. Nothing exotic: the architecture, not
    the optimization, is the contribution.</figcaption>`;
}

// Module 15 — what the heads learned ----------------------------------

export function initLearned(figEl) {
  const sent1 = "… passed new laws since 2009 making the registration or voting process more difficult .".split(" ");
  const sent2 = "The Law will never be perfect , but its application should be just …".split(" ");

  function row(words, y, arcs, hue) {
    const startX = 24;
    let x = startX;
    const pos = words.map((w) => {
      const cx = x + w.length * 3.6;
      x += w.length * 7.2 + 9;
      return cx;
    });
    let s = words.map((w, i) =>
      T(pos[i], y, w, { anchor: "middle", c: "ink", s: 12 })).join("");
    for (const [a, b, wgt] of arcs) {
      const x1 = pos[a], x2 = pos[b];
      const lift = 16 + Math.abs(x2 - x1) * 0.10;
      s += `<path d="M ${x1} ${y - 14} Q ${(x1 + x2) / 2} ${y - 14 - lift} ${x2} ${y - 14}"
        fill="none" stroke="var(--${hue})" stroke-opacity="${wgt}" stroke-width="${1 + wgt * 1.6}"/>`;
    }
    return s;
  }
  const mk = sent1.indexOf("making");
  const inner = `
    ${T(24, 24, "figure 3 · a head completing “making … more difficult” seven positions later", { c: "ink", w: 500 })}
    ${row(sent1, 92, [[mk, sent1.indexOf("more"), 0.95], [mk, sent1.indexOf("difficult"), 0.85],
      [mk, sent1.indexOf("laws"), 0.25], [mk, sent1.indexOf("2009"), 0.2]], "azure")}
    ${T(24, 140, "figure 4 · two heads resolving “its” — one onto “Law”, one onto “application”", { c: "ink", w: 500 })}
    ${row(sent2, 208, [[sent2.indexOf("its"), 1, 0.95]], "azure")}
    ${row(sent2, 208, [[sent2.indexOf("its"), sent2.indexOf("application"), 0.9]], "moss")}
    <rect x="24" y="240" width="672" height="1" fill="var(--ink-08)"/>
    ${T(24, 270, "and it generalized on the first try — english constituency parsing:", { c: "ink" })}
    ${T(24, 292, "4 layers · d_model 1024 · 40K WSJ sentences", {})}
    ${T(696, 270, "WSJ only  91.3 F1", { anchor: "end", c: "ink", w: 500 })}
    ${T(696, 292, "semi-supervised  92.7 F1", { anchor: "end", c: "ink", w: 500 })}`;
  svgFig(figEl,
    "reported · traced from the paper's figures 3–5, not generated",
    "0 0 720 310", inner,
    `The arcs are redrawn from the attention visualizations the paper
     published for its trained EN-DE model, showing the edges the text
     calls out; they are not computed by the toy model on this page.
     Long-range syntax and anaphora, learned without ever being asked
     for.`);
}

// Module 1 — One hop. The thesis figure.
//
// Three lanes carry the same signal between the same two positions:
//   recurrent      one hand-off per position, signal attenuates each time
//   convolutional  reach multiplies by k per layer (dilated stack), S4/Table 1
//   self-attention any position to any position, one hop
//
// Path lengths per Table 1: O(n), O(log_k n), O(1). The reader drags the
// source and target — "any two positions" is the claim, so both ends are
// handles — and sets n and k. A shared clock races the lanes; a scrubber
// drives the same clock by hand (and is the whole interface under
// prefers-reduced-motion).
//
// Pure simulation: no model weights, no paper measurements. The exact
// numbers it shows — step counts and the compounding signal loss — follow
// from the stated rules (one hop per step; reach ×k per layer; each
// hand-off keeps 90%).

import { motion, registerFigure, makeSlider, keyboardNudge } from "../runtime.mjs";

const KEEP = 0.9; // fraction of signal surviving each hand-off

const css = getComputedStyle(document.documentElement);
const C = {
  paper: css.getPropertyValue("--paper").trim(),
  ink: css.getPropertyValue("--ink").trim(),
  ink55: css.getPropertyValue("--ink-55").trim(),
  ink25: css.getPropertyValue("--ink-25").trim(),
  azure: css.getPropertyValue("--azure").trim(),
  vermilion: css.getPropertyValue("--vermilion").trim(),
  moss: css.getPropertyValue("--moss").trim(),
};

export function initOneHop(figEl) {
  figEl.innerHTML = `
    <div class="badge">simulation · path lengths only, no model</div>
    <div class="fig-body" style="position:relative">
      <canvas aria-label="three lanes racing a signal: recurrent, convolutional, self-attention"></canvas>
      <button class="fig-handle" data-h="src" aria-label="source position"></button>
      <button class="fig-handle" data-h="tgt" aria-label="target position"></button>
    </div>
    <div class="fig-controls"></div>
    <figcaption>Three ways to carry a signal between two positions.
    Drag either endpoint. The step counters are the argument; the fading
    pulse in the top lane is why depth is not free.</figcaption>`;

  const canvas = figEl.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const controls = figEl.querySelector(".fig-controls");
  const handles = {
    src: figEl.querySelector('[data-h="src"]'),
    tgt: figEl.querySelector('[data-h="tgt"]'),
  };

  // ------------------------------------------------------------ state
  const st = {
    n: 24,
    k: 3,
    src: 0,      // 0-based position index
    tgt: 23,
    t: 0,        // shared clock, ms
    playing: !motion.reduced,
  };

  // ------------------------------------------------------------ timing
  // The recurrent lane compresses when d is large — the point lands in
  // seconds, not half a minute — but hops stay discrete and countable.
  function timing() {
    const d = Math.abs(st.tgt - st.src);
    const hopRec = Math.min(380, 6500 / Math.max(d, 1));
    const layers = d <= 1 ? (d === 0 ? 0 : 1) : Math.ceil(Math.log(d) / Math.log(st.k));
    const hopConv = 700;
    const hopAttn = 900;
    const tEnd = Math.max(d * hopRec, layers * hopConv, hopAttn, 1);
    return { d, layers, hopRec, hopConv, hopAttn, tEnd, period: tEnd + 1600 };
  }

  // ------------------------------------------------------------ layout
  const PADL = 14, LANE0 = 64, LANEH = 104, AXIS_Y = 26;
  let W = 0, H = 0, dpr = 1, PADR = 158, narrow = false;

  function resize() {
    const w = figEl.querySelector(".fig-body").clientWidth;
    if (!w) return;
    dpr = devicePixelRatio || 1;
    W = w;
    narrow = W < 640;
    PADR = narrow ? 10 : 158;
    H = LANE0 + 3 * LANEH;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    placeHandles();
    draw();
  }

  const px = (i) => PADL + ((W - PADL - PADR) * i) / (st.n - 1);
  const laneY = (l) => LANE0 + l * LANEH + 46;

  function placeHandles() {
    for (const [name, el] of Object.entries(handles)) {
      el.style.left = `${px(st[name]) - 11}px`;
      el.style.top = `${AXIS_Y - 11}px`;
    }
  }

  // ------------------------------------------------------------ drawing
  function drawLaneDots(y, reach) {
    for (let i = 0; i < st.n; i++) {
      const r = st.n > 40 ? 2.5 : 3.5;
      ctx.beginPath();
      ctx.arc(px(i), y, r, 0, 7);
      ctx.fillStyle = reach?.(i) ? reach(i) : C.ink25;
      ctx.fill();
    }
  }

  function arc(x1, x2, y, lift, color, width, alpha = 1, partial = 1) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    const mx = (x1 + x2) / 2, my = y - lift;
    if (partial >= 1) {
      ctx.moveTo(x1, y);
      ctx.quadraticCurveTo(mx, my, x2, y);
    } else {
      // trace the quadratic bezier up to parameter `partial`
      ctx.moveTo(x1, y);
      const steps = 24;
      for (let s = 1; s <= steps * partial; s++) {
        const u = s / steps;
        const xa = x1 + (mx - x1) * u, ya = y + (my - y) * u;
        const xb = mx + (x2 - mx) * u, yb = my + (y - my) * u;
        ctx.lineTo(xa + (xb - xa) * u, ya + (yb - ya) * u);
      }
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function label(l, title, sub, subNarrow) {
    const y = laneY(l);
    const text = narrow ? subNarrow : sub;
    // paper underlay so the endpoint guides never run through the words
    ctx.font = `500 12px 'IBM Plex Mono', monospace`;
    const w1 = ctx.measureText(title).width;
    ctx.font = `400 11px 'IBM Plex Mono', monospace`;
    const w2 = ctx.measureText(text).width;
    ctx.fillStyle = C.paper;
    ctx.fillRect(PADL - 4, y - 52, Math.max(w1, w2) + 8, 32);
    ctx.fillStyle = C.ink;
    ctx.font = `500 12px 'IBM Plex Mono', monospace`;
    ctx.fillText(title, PADL, y - 40);
    ctx.fillStyle = C.ink55;
    ctx.font = `400 11px 'IBM Plex Mono', monospace`;
    ctx.fillText(text, PADL, y - 25);
  }

  function counter(l, steps, total, strength) {
    const y = laneY(l);
    const hue = [C.vermilion, C.moss, C.azure][l];
    const pct = `${(strength * 100).toFixed(strength < 0.1 ? 1 : 0)}%`;
    if (narrow) {
      // counters live on the title rows, right-aligned, over paper
      const line = `${steps}/${total} step${total === 1 ? "" : "s"}`;
      ctx.font = `500 12px 'IBM Plex Mono', monospace`;
      const lw = ctx.measureText(line).width;
      ctx.fillStyle = C.paper;
      ctx.fillRect(W - PADR - lw - 78, y - 52, lw + 82, 32);
      ctx.fillStyle = C.ink;
      ctx.fillText(line, W - PADR - lw, y - 40);
      const bw = 44;
      ctx.fillStyle = C.ink25;
      ctx.fillRect(W - PADR - bw, y - 30, bw, 2);
      ctx.fillStyle = hue;
      ctx.fillRect(W - PADR - bw, y - 30, bw * strength, 2);
      ctx.fillStyle = C.ink55;
      ctx.font = `400 11px 'IBM Plex Mono', monospace`;
      ctx.fillText(pct, W - PADR - bw - 8 - ctx.measureText(pct).width, y - 25);
      return;
    }
    const x = W - PADR + 16;
    ctx.font = `500 13px 'IBM Plex Mono', monospace`;
    ctx.fillStyle = C.ink;
    ctx.fillText(`${steps}/${total} step${total === 1 ? "" : "s"}`, x, y - 14);
    const bw = 100;
    ctx.fillStyle = C.ink25;
    ctx.fillRect(x, y - 4, bw, 2);
    ctx.fillStyle = hue;
    ctx.fillRect(x, y - 4, bw * strength, 2);
    ctx.fillStyle = C.ink55;
    ctx.font = `400 11px 'IBM Plex Mono', monospace`;
    ctx.fillText(`signal ${pct}`, x, y + 14);
  }

  function draw() {
    const { d, layers, hopRec, hopConv, hopAttn, tEnd } = timing();
    const t = Math.min(st.t, tEnd);
    const dir = Math.sign(st.tgt - st.src) || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // ---- axis strip: positions + endpoint guides
    ctx.font = `400 11px 'IBM Plex Mono', monospace`;
    ctx.fillStyle = C.ink55;
    ctx.fillText(`position ${st.src + 1}`, Math.min(px(st.src), px(st.tgt)) - 8, AXIS_Y - 16);
    const tl = `position ${st.tgt + 1}`;
    ctx.fillText(tl, Math.max(px(st.src), px(st.tgt)) - ctx.measureText(tl).width + 8, AXIS_Y - 16);
    ctx.strokeStyle = C.ink25;
    ctx.setLineDash([2, 4]);
    for (const p of [st.src, st.tgt]) {
      ctx.beginPath();
      ctx.moveTo(px(p), AXIS_Y + 4);
      ctx.lineTo(px(p), H - 26);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // ================= lane 0: recurrent =================
    {
      const y = laneY(0);
      const prog = Math.min(d, t / hopRec);          // hops completed (fractional)
      const done = Math.floor(prog);
      label(0, "recurrent",
        `one hand-off per position · keeps ${KEEP * 100}% each`,
        `keeps ${KEEP * 100}%/hop`);
      drawLaneDots(y, (i) => {
        const off = (i - st.src) * dir;
        return off >= 0 && off <= done ? C.vermilion : null;
      });
      // trail arcs, fading with compounded loss
      for (let h = 0; h < done; h++) {
        const a = KEEP ** h;
        arc(px(st.src + h * dir), px(st.src + (h + 1) * dir), y, 14, C.vermilion, 1.5, 0.25 + 0.75 * a);
      }
      // the moving pulse
      if (d > 0 && prog < d) {
        const x = px(st.src + prog * dir);
        const s = KEEP ** prog;
        ctx.beginPath();
        ctx.arc(x, y - 8, 3 + 4 * s, 0, 7);
        ctx.fillStyle = C.vermilion;
        ctx.globalAlpha = 0.25 + 0.75 * s;
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      counter(0, Math.min(done, d), d, KEEP ** d);
    }

    // ================= lane 1: convolutional =================
    {
      const y = laneY(1);
      const prog = layers ? Math.min(layers, t / hopConv) : 0;
      const done = Math.floor(prog);
      label(1, "convolutional", `reach ×${st.k} per layer (dilated)`,
        `reach ×${st.k}/layer`);
      const reachAt = (ly) => Math.min(d, st.k ** ly);
      drawLaneDots(y, (i) => {
        const off = (i - st.src) * dir;
        return off >= 0 && off <= (done ? reachAt(done) : 0) ? C.moss : null;
      });
      for (let ly = 1; ly <= Math.ceil(prog); ly++) {
        const from = ly === 1 ? 0 : reachAt(ly - 1);
        const to = reachAt(ly);
        const part = ly <= done ? 1 : prog - done;
        arc(px(st.src + from * dir), px(st.src + to * dir), y, 16 + 8 * ly, C.moss, 1.5, 0.9, part);
      }
      counter(1, Math.min(done, layers), layers, KEEP ** layers);
    }

    // ================= lane 2: self-attention =================
    {
      const y = laneY(2);
      const prog = Math.min(1, t / hopAttn);
      label(2, "self-attention", "every position sees every other",
        "any → any");
      drawLaneDots(y, (i) => (i === st.src || (prog >= 1 && i === st.tgt) ? C.azure : null));
      const lift = Math.max(40, Math.min(64, (W - PADL - PADR) * 0.055));
      if (d > 0) arc(px(st.src), px(st.tgt), y, lift, C.azure, 3, 1, prog);
      counter(2, prog >= 1 ? 1 : 0, d === 0 ? 0 : 1, KEEP);
    }
  }

  // ------------------------------------------------------------ clock
  let raf = null, lastTs = null;
  function tick(ts) {
    raf = requestAnimationFrame(tick);
    if (lastTs != null && st.playing) {
      st.t = (st.t + (ts - lastTs)) % timing().period;
      scrub.set(Math.round(Math.min(st.t, timing().tEnd) / timing().tEnd * 100));
      draw();
    }
    lastTs = ts;
  }

  // ------------------------------------------------------------ controls
  const nSlider = makeSlider({
    label: "n", min: 8, max: 64, value: st.n,
    onInput: (v) => {
      const rel = { src: st.src / (st.n - 1), tgt: st.tgt / (st.n - 1) };
      st.n = v;
      st.src = Math.round(rel.src * (v - 1));
      st.tgt = Math.round(rel.tgt * (v - 1));
      if (st.src === st.tgt) st.tgt = (st.src + 1) % v;
      reset();
    },
  });
  const kSlider = makeSlider({
    label: "k", min: 2, max: 8, value: st.k,
    onInput: (v) => { st.k = v; reset(); },
  });
  const playBtn = document.createElement("button");
  playBtn.className = "ctl-btn";
  const setPlayLabel = () => {
    playBtn.textContent = st.playing ? "pause" : "play";
    playBtn.setAttribute("aria-label", st.playing ? "pause the race" : "play the race");
  };
  playBtn.addEventListener("click", () => {
    st.playing = !st.playing;
    if (st.playing && st.t >= timing().tEnd) st.t = 0;
    setPlayLabel();
  });
  setPlayLabel();
  const scrub = makeSlider({
    label: "scrub", min: 0, max: 100, value: 0, unit: "%",
    onInput: (v) => {
      st.playing = false;
      setPlayLabel();
      st.t = (v / 100) * timing().tEnd;
      draw();
    },
  });
  controls.append(nSlider.el, kSlider.el, playBtn, scrub.el);

  function reset() {
    st.t = motion.reduced && !st.playing ? timing().tEnd : 0;
    placeHandles();
    scrub.set(st.t ? 100 : 0);
    draw();
  }

  // ------------------------------------------------------------ handles
  for (const [name, el] of Object.entries(handles)) {
    const move = (pos) => {
      const p = Math.max(0, Math.min(st.n - 1, pos));
      if (p === st[name === "src" ? "tgt" : "src"]) return;
      st[name] = p;
      reset();
    };
    el.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener("pointermove", (e) => {
      if (!el.hasPointerCapture?.(e.pointerId)) return;
      const rect = canvas.getBoundingClientRect();
      const frac = (e.clientX - rect.left - PADL) / (W - PADL - PADR);
      move(Math.round(frac * (st.n - 1)));
    });
    keyboardNudge(el, {
      label: `${name === "src" ? "source" : "target"} position`,
      onNudge: (dx) => move(st[name] + dx),
    });
  }

  // ------------------------------------------------------------ lifecycle
  new ResizeObserver(resize).observe(figEl.querySelector(".fig-body"));
  registerFigure(figEl, {
    start() {
      lastTs = null;
      if (!raf) raf = requestAnimationFrame(tick);
    },
    stop() {
      if (raf) cancelAnimationFrame(raf);
      raf = null;
    },
  });
  if (motion.reduced) {
    st.playing = false;
    setPlayLabel();
    st.t = timing().tEnd;
    scrub.set(100);
  }
  resize();
}

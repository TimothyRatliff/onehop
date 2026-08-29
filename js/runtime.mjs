// onehop — shared figure runtime.
//
// Everything figures have in common lives here: weight loading with a real
// progress state, a figure registry with IntersectionObserver pause/resume,
// the reduced-motion flag, and the interaction grammar from DESIGN.md
// (sliders with visible mono value chips, drag helpers, keyboard nudging).
// No framework; each figure module gets a small controller object.

import { loadModel } from "./model.mjs";

// ---------------------------------------------------------------- motion

export const motion = {
  reduced: matchMedia("(prefers-reduced-motion: reduce)").matches,
};
matchMedia("(prefers-reduced-motion: reduce)").addEventListener("change", (e) => {
  motion.reduced = e.matches;
});

// ---------------------------------------------------------------- weights

/**
 * Fetch config.json + model.bin and build the model. onProgress(loaded,
 * total) fires during the weight download so the page can show a real
 * loading state, not a spinner over a blank page.
 */
export async function loadWeights(onProgress) {
  const config = await (await fetch("config.json")).json();
  const res = await fetch("model.bin");
  const total = +res.headers.get("Content-Length") || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  return loadModel(config, buf.buffer);
}

// ---------------------------------------------------------------- registry

const observer = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      const fig = figures.get(e.target);
      if (!fig) continue;
      if (e.isIntersecting) {
        if (!fig.running) {
          fig.running = true;
          fig.start?.();
        }
      } else if (fig.running) {
        fig.running = false;
        fig.stop?.();
      }
    }
  },
  { rootMargin: "64px" },
);
const figures = new Map();

/**
 * Register a figure. start() is called when it scrolls into view, stop()
 * when it leaves — fifteen live figures animating at once would melt a
 * laptop, so anything with a rAF loop must start/stop here.
 */
export function registerFigure(el, { start, stop } = {}) {
  const fig = { start, stop, running: false };
  figures.set(el, fig);
  observer.observe(el);
  return {
    get running() {
      return fig.running;
    },
    unregister() {
      observer.unobserve(el);
      figures.delete(el);
    },
  };
}

// ---------------------------------------------------------------- format

/** Fixed-width number for live readouts (mono font keeps them steady). */
export function fmt(v, digits = 4) {
  if (!Number.isFinite(v)) return v > 0 ? "+∞" : "−∞";
  const s = v.toFixed(digits);
  return v >= 0 ? " " + s : s.replace("-", "−");
}

// ---------------------------------------------------------------- controls

/**
 * A slider per the figure grammar: hairline track, azure dot handle,
 * always-visible mono value chip. Keyboard: arrows nudge one step,
 * shift-arrows ten. Returns { el, get value, set(v) }.
 */
export function makeSlider({ label, min, max, step = 1, value, unit = "", onInput, format }) {
  const el = document.createElement("label");
  el.className = "ctl-slider";
  const fmtV = format ?? ((v) => String(v));
  el.innerHTML =
    `<span class="ctl-label">${label}</span>` +
    `<input type="range" min="${min}" max="${max}" step="${step}" value="${value}">` +
    `<output class="chip">${fmtV(value)}${unit}</output>`;
  const input = el.querySelector("input");
  const out = el.querySelector("output");
  input.addEventListener("input", () => {
    out.textContent = fmtV(+input.value) + unit;
    onInput?.(+input.value);
  });
  input.addEventListener("keydown", (e) => {
    if (e.shiftKey && (e.key === "ArrowLeft" || e.key === "ArrowRight" ||
                       e.key === "ArrowUp" || e.key === "ArrowDown")) {
      e.preventDefault();
      const dir = e.key === "ArrowRight" || e.key === "ArrowUp" ? 1 : -1;
      input.value = String(
        Math.min(max, Math.max(min, +input.value + dir * step * 10)),
      );
      input.dispatchEvent(new Event("input"));
    }
  });
  return {
    el,
    get value() {
      return +input.value;
    },
    set(v) {
      input.value = String(v);
      out.textContent = fmtV(+input.value) + unit;
    },
  };
}

/**
 * Pointer-drag helper for in-figure handles. Calls onDrag(dx, dy, ev)
 * with movement since the last call; captures the pointer for the drag.
 */
export function attachDrag(el, { onStart, onDrag, onEnd }) {
  let last = null;
  el.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    el.setPointerCapture(e.pointerId);
    last = [e.clientX, e.clientY];
    el.classList.add("dragging");
    onStart?.(e);
  });
  el.addEventListener("pointermove", (e) => {
    if (!last) return;
    onDrag?.(e.clientX - last[0], e.clientY - last[1], e);
    last = [e.clientX, e.clientY];
  });
  const end = (e) => {
    if (!last) return;
    last = null;
    el.classList.remove("dragging");
    onEnd?.(e);
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
}

/** Make an SVG/DOM handle keyboard-operable: arrows call onNudge(dx, dy)
 * in steps (shift = x10), with the element focusable and labeled. */
export function keyboardNudge(el, { label, onNudge, step = 1 }) {
  el.tabIndex = 0;
  el.setAttribute("role", "slider");
  el.setAttribute("aria-label", label);
  el.addEventListener("keydown", (e) => {
    const dirs = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, -1],
      ArrowDown: [0, 1],
    };
    const d = dirs[e.key];
    if (!d) return;
    e.preventDefault();
    const k = (e.shiftKey ? 10 : 1) * step;
    onNudge(d[0] * k, d[1] * k);
  });
}

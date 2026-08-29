# onehop — design system

The register is Ciechanowski: a quiet page where figures are the only
loud thing. Every choice below is subordinate to that.

## Color

Six named values. Light mode, like the model for this site — heatmaps
and vector readouts encode magnitude, and magnitude needs a neutral
paper ground to read against. No dark mode.

| name        | hex       | role |
|-------------|-----------|------|
| `paper`     | `#FAF8F3` | ground. Warm off-white, not pure white — less glare over long reads |
| `ink`       | `#221E19` | prose, axes, structural lines |
| `azure`     | `#0072B2` | self-attention; head 1; **every interactive handle** |
| `vermilion` | `#D55E00` | recurrence; head 2; failure states (attenuation, softmax collapse) |
| `moss`      | `#009E73` | convolution; head 3 |
| `plum`      | `#CC79A7` | head 4 |

The four hues are Okabe–Ito values: distinguishable under the common
color-vision deficiencies, which matters because module 6 asks the
reader to tell four heads apart and module 1 to tell three lanes apart.
The same four hues serve both jobs — the palette is the cast of
characters, not decoration. One hue = one meaning per module, stated in
the figure's key.

Derived, not named: secondary text and hairlines are `ink` at 55% / 25% /
8% opacity; attention heatmaps ramp `paper → azure` (weight 0 → 1);
masked cells are flat `ink` with the −∞ sign knocked out in `paper`.
No gradients anywhere else.

## Type

Two font families, three roles, self-hosted woff2 (no build step, no
third-party requests).

- **Body — Source Serif 4** (variable, opsz axis), ~18.5px/1.65.
  A workhorse text serif with real italics; long-form readability is its
  design brief. Not Inter, not a geometric sans — the prose should read
  like an essay, not an app.
- **Display — Source Serif 4 at the display end of its optical-size
  axis**, weight 600, tight leading. Deliberately the same family: a
  second display personality would be the loudest thing on a page whose
  loudest thing must be the figures. The optical axis makes large
  settings sharper and higher-contrast, so headings still feel set, not
  merely enlarged.
- **Mono — IBM Plex Mono** for tensor shapes, live values, axis labels,
  and the HUD. Humanist and warm next to a serif, and monospaced digits
  mean live-updating numbers don't jitter in width — every number in
  every figure is set in it.

Scale: 40/28/22 display, 18.5 body, 15 serif-italic captions, 13 mono
labels. Two variable files (roman + italic) plus two Plex Mono weights
≈ 350 KB total, `font-display: swap`.

## Layout

**Concept:** one quiet column of prose down the center; figures break
out onto a wider fixed stage; the tensor HUD keeps station in the right
margin, always visible, never in the text's way.

```
                    ├────── 660px prose ──────┤
                      You are reading about
                      attention. The text sets
                      up what the figure shows.
                                                          ┌─ HUD ────┐
        ┌───────────── figure stage, 960px ─────────┐     │ x  13×48 │
        │                                           │     │ ▓▓▓▓▓▓▓  │
        │         (canvas / svg simulation)         │     │ QKᵀ 4×13²│
        │                                           │     │ ▓▓ ▓▓ ▓▓ │
        │    n ──────────○──── 12   [k: 3]          │     │ ·V 13×48 │
        └───────────────────────────────────────────┘     │──────────│
                      The prose resumes and               │ w[2][7]  │
                      interprets what you just            │ = 0.4183 │
                      touched.                            └──────────┘
```

- Prose column: 660px max (~66ch), centered. 8px vertical rhythm.
  Sections are separated by whitespace and a small mono section number
  (`§ 4`), nothing else. No cards, no rules, no boxes around text.
- Figure stage: 960px max, centered; module 1 alone may go to 1100px.
- HUD rail: ~180px, sticky, right of the stage at ≥1280px viewport;
  below that it docks as a slim strip at the bottom edge.
- Mobile (≥380px): single column, figures full-width and redesigned for
  touch (bigger handles, tap-to-inspect instead of hover); HUD is the
  bottom strip, one stage at a time.

## The tensor HUD

An instrument, not a caption. `js/hud.mjs`, one SVG panel, always
present.

- **What it shows:** the shape pipeline of the figure the reader is
  currently at — each stage a small rectangle pair drawn roughly
  proportional to its dimensions, labeled in mono:
  `x 13×48 → split 4×(13×12) → QKᵀ 4×(13×13) → softmax → ·V → concat
  13×48 → FFN 13×128 → 13×48`. The active stage (what the figure is
  showing now, or the sub-step the reader has scrubbed to) is filled in
  that module's hue; the rest are `ink` 25%.
- **Reshapes animate** (150ms, linear): the 13×48 bar visibly splits
  into four 13×12 bars when heads split, and rejoins at concat. That
  animation *is* the content — it's the one place the site shows
  reshaping as motion. Under `prefers-reduced-motion` stages swap
  instantly.
- **The value line:** the bottom row of the HUD is a live readout.
  Hover or tap any cell, vector, or curve point in any figure and it
  shows the fully-qualified name and value:
  `enc0.attn.weights[2][7][3] = 0.4183`. This is where "show its real
  numbers" lives globally — every figure gets value inspection for
  free by publishing to it.
- **Idle state:** when no figure is in view it shows the model card —
  `2 enc + 2 dec · d_model 48 · h 4 · 109,376 params · fp16` — with the
  one-time "this is a toy model, not the paper's" note beside it. The
  honesty rule gets a permanent, visible home.
- **Wiring:** figures call `hud.set(stages, active)` and
  `hud.value(name, v)` — a ~30-line pub/sub, no framework. The HUD
  never scrolls the page, never captures the pointer, and is
  `aria-live="polite"` so the value line is announced to screen
  readers.

## Figure grammar (shared by every module)

- **Handles** are a filled `azure` dot on a thin ring; they grow on
  hover/focus and show a mono value chip while dragging. Anything
  draggable looks like this; nothing that looks like this is static.
- **Sliders** are a hairline track + the same dot; the value chip is
  always visible, in mono, and is itself click-to-type-editable.
- Keyboard: every handle is a real focusable element; arrows nudge,
  shift-arrows nudge ×10; focus is a visible 2px `azure` outline offset
  from the shape.
- Canvas at `devicePixelRatio`, for anything with many moving parts;
  SVG for structure. Hairlines 1px `ink` 25%.
- Every live figure carries a small mono badge `live · toy model`;
  every reported figure carries `reported · Vaswani et al. 2017`. Same
  position, same set, every time — the two-kinds-of-number rule made
  visual.
- IntersectionObserver pauses any figure fully offscreen.

## Defaults rejected on review

Inter and the system-sans stack (app voice, not essay voice); a second
display family (competes with figures); dark mode (heatmaps need
paper); pure-white ground (glare); Tailwind-style neutral gray ramps
(cool grays fight the warm paper); card grids, section icons, and
numbered-chip headers (chrome); scroll-triggered reveals of any kind
(the reader controls the pace); viridis for heatmaps (a fifth hue that
means nothing here — the azure ramp keeps hue = role).

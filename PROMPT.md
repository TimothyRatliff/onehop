# onehop — build prompt

Paste as `PROMPT.md` in an empty repo with the paper PDF alongside it. Phase 0 produces artifacts every later phase depends on, so don't let it be skipped.

---

## THE PROMPT

You are building **onehop**, an interactive explainer for *Attention Is All You Need* (Vaswani et al., NIPS 2017). The paper is attached and is the source of truth for every number, claim, and equation on the site.

The site has an argument, not just a subject. **A self-attention layer connects any two positions in one hop.** Recurrence needs n hops, convolution needs log_k(n), and every other property the paper reports — the parallelism, the training speed, the long-range dependencies the heads actually learned — falls out of that one fact. The name is the thesis. Build the narrative so the reader arrives at Table 1 already convinced.

### The register: Ciechanowski

The model for this site is Bartosz Ciechanowski's work (ciechanow.ski). Study what that actually means, because it is a specific and demanding discipline, not a vibe:

- **Every figure is a real simulation with handles.** Not an animation that plays. Something the reader grabs, perturbs, and breaks. If a figure has no handle, justify why.
- **The reader controls the pace.** No scroll-jacking, no forced sequences, no autoplay, no "continue" gates. Scroll position never hijacks the camera.
- **Figures are stateful and persistent.** A slider the reader moved stays moved. Figures keep running while the reader thinks.
- **The prose is patient and plain.** Second person, present tense, short sentences, no hype, no exclamation, no "let's dive in." It explains what the reader is about to touch and then gets out of the way.
- **The page is quiet.** Generous whitespace, restrained type, almost no chrome. The figures are the only loud thing, and they are loud because they move when you touch them.

**What this is not.** If any of the following appear, you have failed the brief: a hero section with an animated particle background; gradient meshes; scroll-triggered fade-up-and-in on every paragraph; a "The Transformer, Reimagined" headline; parallax; decorative easing on things that are not simulating anything; a card grid of "features"; emoji as section markers. Restraint is the aesthetic. If a reader would describe the page as "slick," that is a bad sign.

### Phase 0 — Train the model

The site runs a **real transformer with real weights**, trained by you, in the reader's browser. Every attention map, softmax distribution, and intermediate tensor the reader sees is computed live from those weights. No mock matrices, no random noise, no hand-drawn heatmaps.

The model must be **the paper's architecture**, scaled down — encoder–decoder, sinusoidal positional encoding, multi-head scaled dot-product attention, post-LN residual sub-layers, position-wise FFN. All three uses of attention must exist and be real, because three modules of the site depend on showing encoder self-attention, masked decoder self-attention, and encoder–decoder cross-attention on live data.

**Task: date normalization.** Character-level. Input is a messily formatted date (`3 March 2012`, `Mar 3, 2012`, `03/03/12`, `the third of march 2012`, `2012.03.03`); output is `2012-03-03`. This task is chosen deliberately: the vocabulary is tiny, it trains to near-perfect accuracy on CPU in minutes, and the cross-attention produces a legible, near-monotone alignment that a reader can immediately recognize as the model finding the year, then the month, then the day. Generate the training data synthetically with a fixed seed.

Starting configuration, tune to fit the size budget: N = 2 encoder and 2 decoder layers, d_model = 48, h = 4, d_k = d_v = 12, d_ff = 128, max source length 32, target length 10. **Total weight payload under 500 KB**, ideally under 250 KB; use fp16 storage if that's what it takes.

Deliverables from this phase:

1. `train.py` — PyTorch, fixed seed, reproducible, with a short README noting final accuracy.
2. `model.bin` + `config.json` — flat Float32Array (or fp16) weights and the architecture config.
3. `golden.json` — a handful of stored test cases containing input, and PyTorch's exact intermediate activations at every sub-layer for that input.

Do not train in the browser. Do not skip step 3.

### Phase 1 — The inference engine

Write the forward pass **by hand in plain JavaScript**. Do not use ONNX Runtime, transformers.js, or any inference library.

This is not a performance decision — at this size a forward pass is trivial. It is an architectural one. The site's entire purpose is exposing intermediates: pre-softmax scores, post-softmax weights, per-head outputs, residual stream before and after each add, LayerNorm statistics. An inference library gives you outputs and hides the guts. You need the guts, on every step, addressable by name.

Structure it so every operation can be observed: a forward pass returns a full trace object, and figures subscribe to the slices they care about. Include a greedy decode loop that can be stepped one token at a time.

**Verify against `golden.json` to within 1e-4 before building a single figure.** A JS implementation that is subtly wrong will produce plausible-looking attention maps and poison the entire site. Write this as an actual test that runs, not a claim in a comment.

### Phase 2 — Design system

Before writing page code, produce a compact plan and show it to me:

- **Color:** 4–6 named hex values. Note that Ciechanowski works in light mode with a restrained palette; you may depart from that, but justify it.
- **Type:** a display face, a body face with excellent long-form readability, and a monospace face for tensor shapes and numerals. Do not reach for Inter as the default answer.
- **Layout:** a text column narrow enough to read comfortably, with figures allowed to break wider. Describe with a one-sentence concept and an ASCII wireframe.
- **The tensor HUD:** the site's signature element. A persistent, well-designed instrument showing the shape of whatever the reader is currently looking at, mutating as they move through the stack: `32×48` → `4 × (32×12)` → concat → `32×48` → FFN `32×128` → `32×48`. Design it as an instrument, not a caption.

Review the plan against this brief and revise anything that reads like a default before building.

### Phase 3 — The modules

Fifteen modules, tiered by effort. **Tier 1 gets full Ciechanowski treatment.** Tier 2 is interactive but modest. Tier 3 is a precise static figure in the same visual language, with no interaction. Do not silently promote or demote a module; if you think a tier is wrong, say so before building.

**Tier 1 — live simulations, real weights, everything draggable**

1. **One hop.** The thesis. Signal must travel from position 1 to position n. Three lanes run at once: recurrent (n sequential steps, and let the signal visibly attenuate), convolutional (log_k n), self-attention (one hop). Reader sets n and k. Step counters on each lane. This is the module the domain is named after; it should be the best thing on the site.
2. **Scaled dot-product attention.** Reader types a date. Q, K, V for a chosen token are shown as real vectors from the real model. Dot products against every key, then the softmax, then the weighted sum of values — each stage separately grabbable, with the numbers visible and live.
3. **Why √d_k.** A slider for d_k. As it rises, show the dot-product distribution widening, the softmax collapsing toward one-hot, and a gradient-magnitude gauge falling. Then a toggle that divides by √d_k and snaps it back. This makes footnote 4 — components of q and k independent with mean 0 and variance 1, so q·k has variance d_k — into something the reader feels rather than reads.
4. **Positional encoding.** Render the encoding as a bank of clock hands, one per dimension pair, spinning at geometrically spaced rates from 2π to 10000·2π. Let the reader drag an offset k and watch every hand advance by a fixed rotation. That is the paper's claim that PE_pos+k is a linear function of PE_pos, demonstrated instead of asserted. Include the sinusoid heatmap and a PE·PE similarity matrix showing the band structure.
5. **The mask.** A live decoder attention grid where the reader toggles causal masking and watches −∞ flood the upper triangle and softmax to exactly zero. Include a cheat mode with masking off, where the model trivially copies the target it was allowed to see. The failure is the explanation.
6. **Multi-head.** All four heads of the real model, running simultaneously on the reader's input, each with its own map and color, converging through concat and W^O. Let the reader ablate individual heads and watch the output degrade. With a real model this is not a metaphor — zeroing a head actually changes the prediction.

**Tier 2 — interactive, modest**

7. **The bottleneck.** Why recurrence hurts: h_t depends on h_t−1, so no parallelization within a training example, and memory constraints cap batching at long sequence lengths.
8. **The complexity crossover.** Table 1 as a live plot with n and d sliders, showing where n²·d overtakes n·d², with d = 512 marked and typical sentence lengths shaded.
9. **The learning-rate schedule.** The formula lrate = d_model^−0.5 · min(step^−0.5, step · warmup^−1.5) plotted with a warmup_steps handle, so the linear ramp and the inverse-square-root decay visibly meet at the kink. warmup_steps = 4000 in the paper.
10. **Ablations.** Table 3 as something the reader manipulates: single-head is 0.9 BLEU worse than the best setting, but quality also falls with too many heads; shrinking d_k hurts; bigger models help; dropout matters.
11. **Cost against quality.** Table 2 as a scatter with FLOPs on a log axis. Transformer big at 28.4 BLEU EN-DE and 41.8 EN-FR for 2.3·10¹⁹ FLOPs, base at 27.3 and 38.1 for 3.3·10¹⁸, against the GNMT+RL ensemble at 1.1·10²¹. The argument makes itself; don't narrate it.

**Tier 3 — precise static figures**

12. **The stack.** Figure 1, redrawn in the site's visual language. N = 6 encoder and decoder layers, d_model = 512, residual connection and LayerNorm around every sub-layer, output embeddings offset by one position.
13. **The feed-forward sub-layer.** max(0, xW₁+b₁)W₂+b₂, d_ff = 2048, same across positions within a layer, different across layers — equivalently two convolutions of kernel size 1.
14. **Training.** Adam with β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹. Residual dropout P_drop = 0.1. Label smoothing ε_ls = 0.1, which hurts perplexity but improves BLEU. 8 P100 GPUs; base 100K steps ≈ 12 hours at ~0.4 s/step, big 300K steps ≈ 3.5 days at ~1.0 s/step. 65M and 213M parameters.
15. **What the heads learned, and where it generalized.** The paper's own Figures 3–5: heads following the long-distance 'making … more difficult' dependency, and the sharp anaphora resolution of 'its' onto 'Law' and 'application'. Plus constituency parsing — 4 layers, d_model = 1024, 40K WSJ sentences, 91.3 F1 WSJ-only and 92.7 semi-supervised.

### The figure contract

Every Tier 1 and Tier 2 figure must:

- Have at least one handle the reader can grab, and respond within a frame.
- Show its numbers. Tensor shapes, actual values, live. A reader who wants to check the arithmetic can.
- Pause when scrolled out of view, via IntersectionObserver. Fifteen live figures animating simultaneously will melt a laptop.
- Have a static, meaningful end-state under `prefers-reduced-motion`, plus manual scrubbing. Reduced motion means the reader drives; it does not mean a blank box.
- Be keyboard operable with visible focus. A slider that only responds to a mouse excludes people.

### Honesty rules

The site shows two different kinds of number and must never blur them.

- **Live** — computed from the toy date model, in the browser, right now. Label these clearly. The toy model has 2 layers and d_model = 48; it is not the paper's model and the site must say so, once, plainly, at the point where the reader first meets it.
- **Reported** — from the paper: BLEU scores, FLOPs, parameter counts, ablations, Figures 3–5. These are the paper's model. Attention maps in module 15 are traced from the paper's published figures, not generated.

An expert reader will check this. Getting it right is a credibility feature; getting it wrong discredits everything else on the page.

One deliberate easter egg: §6.1 reports 41.0 BLEU for EN-FR while the abstract and Table 2 both say 41.8. Acknowledge the discrepancy somewhere a close reader will find it.

### Technical constraints

- Static site, no build step, deployable to S3 + CloudFront as-is. Single HTML file plus `model.bin`, `config.json`, and fonts; a small number of JS modules is acceptable if you use native ES modules.
- Canvas for anything with many moving elements, SVG for structural diagrams. Nothing is a screenshot or a video.
- No framework. No bundler. Vanilla JS with ES modules.
- 60fps on a laptop. Weights load once, cached, with a real loading state — not a spinner over a blank page.
- Desktop-first, but every Tier 1 figure needs a genuine mobile form. Where an interaction cannot survive a 380px viewport, redesign it for touch rather than hiding the section.
- Comment the math against the paper's section numbers so a reader who opens devtools can check your work.

### When it runs long

It will. In order: cut Tier 3 modules to a sentence and a single figure; demote Tier 2 to Tier 3; never demote Tier 1. Six exceptional simulations and nine honest static figures is a far better site than fifteen mediocre ones. If you are choosing between polishing module 1 and building module 11, polish module 1.

### Process

Do Phase 0 and Phase 1 completely, and verify against `golden.json`, before any page code. Then show me the Phase 2 plan and stop. Then build Tier 1 one module at a time, screenshotting and critiquing each before moving on.

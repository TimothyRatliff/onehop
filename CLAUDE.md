# onehop

Interactive explainer for *Attention Is All You Need* (Vaswani et al., 2017). Deploys as a static site to S3 + CloudFront.

**The thesis:** a self-attention layer connects any two positions in one hop. Recurrence needs n, convolution needs log_k(n). Everything else in the paper follows from that. The site is an argument, not a summary.

Full spec is in `PROMPT.md`. Agreed design system is in `DESIGN.md`. This file holds only the things that must not drift.

## Hard constraints

- **No framework, no bundler, no build step.** Vanilla JS, native ES modules, served as files.
- **No inference library.** The forward pass is hand-written so every intermediate is observable by name. ONNX Runtime and transformers.js are both out — they return outputs and hide the guts, and the guts are the product.
- **No training in the browser.** `model.bin` is a committed static asset, trained once by `train.py`. Visitors run inference only.
- **The parity test gates everything.** `test/parity.mjs` compares the JS forward pass against PyTorch activations in `golden.json` to 1e-4. If it fails, fix it before touching any figure. Wrong attention maps look completely plausible.
- **Never demote Tier 1.** Tiers are in `PROMPT.md`. When scope runs long, cut Tier 3 to a sentence, demote Tier 2, and leave Tier 1 alone. Say so before changing any module's tier.

## Git & Commit Guidelines
- Always use the Conventional Commits specification: `<type>(<optional scope>): <description>`.
- Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- Use the imperative mood in the subject line (e.g., "add feature", not "added feature").
- Keep the first line under 50 characters.
- Do not add punctuation (periods) at the end of the subject line.
- For breaking changes, use `!` after type/scope (e.g., `feat(api)!: change response payload`) or include `BREAKING CHANGE:` in the footer.

## Two kinds of number, never blurred

- **Live** — computed in-browser from the toy date model (2 layers, d_model 48). Label as live. The site says once, plainly, that this is not the paper's model.
- **Reported** — BLEU scores, FLOPs, parameter counts, ablations, Figures 3–5. These are the paper's model. Attention maps in module 15 are traced from published figures, not generated.

Blurring these discredits the whole site. An expert reader checks.

## The figure contract

Every Tier 1 and Tier 2 figure:

- Has a handle the reader can grab, and responds within a frame.
- Shows its real numbers — tensor shapes and values, live.
- Pauses when offscreen via IntersectionObserver.
- Has a meaningful static end-state plus manual scrubbing under `prefers-reduced-motion`.
- Is keyboard operable with visible focus.

## Register

Ciechanowski, not conference landing page. The reader controls the pace: no scroll-jacking, no autoplay, no forced sequences. Prose is second person, plain, unhurried, no hype.

**Banned:** particle backgrounds, gradient meshes, fade-up-on-scroll, parallax, decorative easing on things that aren't simulating anything, feature card grids, emoji section markers.

## Layout

```
train.py            Phase 0, PyTorch, fixed seed
model.bin           frozen weights, <500KB, committed
config.json         architecture config
golden.json         PyTorch activations for parity testing
index.html          the site
js/model.mjs        hand-written forward pass, returns full trace
js/figures/         one module per file
test/parity.mjs     must pass before any figure work
```

## Commands

```
python train.py               retrain (rarely needed; weights are committed)
node test/parity.mjs          verify JS matches PyTorch
python -m http.server 8000    serve locally
```

## Working agreement

One module per session, then commit. Screenshot and critique each figure before reporting it done. Ask of every figure: *what does a reader who already knows this material learn from it?* If the answer is nothing, it's decoration — cut it or rethink it.

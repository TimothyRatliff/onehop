# onehop

Interactive explainer for *Attention Is All You Need* (Vaswani et al., 2017).
Live at <https://www.onehop.world>.

## The model

`train.py` trains the paper's architecture scaled down — 2 encoder + 2
decoder layers, d_model 48, 4 heads, d_k = d_v = 12, d_ff 128, post-LN,
sinusoidal positional encoding, tied embeddings — on character-level date
normalization (`3 march 2012` → `2012-03-03`). 109,376 parameters, stored
as fp16 in `model.bin` (214 KB). Fixed seed (1337), reproducible.

**Final accuracy: 100.0% exact match** on a held-out validation set of
3,000 dates (uniform over all nine input formats), measured *after* fp16
quantization — the accuracy of exactly the weights the browser loads.

## Two kinds of number

Figures that compute from the toy weights in your browser are labelled
**live**. BLEU scores, FLOPs, parameter counts, ablations and the paper's
own figures are labelled **reported** — those are the paper's model, not
this one. The site never blurs the two, and says plainly that the toy is
not the paper's model.

## Parity

`golden.json` stores PyTorch's intermediate activations at every
sub-layer for three inputs, computed in float64 from the fp16 weights.
`test/parity.mjs` checks the hand-written JS forward pass
(`js/model.mjs`) against them to within 1e-4. It must pass before any
figure work.

## Commands

```
npm run serve                 serve locally on :8000
npm run test:parity           verify JS matches PyTorch
npm run test:e2e              e2e smoke tests
python3 train.py              retrain (rarely needed; weights are committed)
```

`train.py` needs PyTorch, which is not otherwise a dependency of this
repo: `python3 -m venv .venv && .venv/bin/pip install torch`, then
`.venv/bin/python train.py`.

## License

MIT — see `LICENSE`. The bundled fonts (Source Serif 4, IBM Plex Mono) are
SIL OFL 1.1 and the paper PDF is the authors'; both are carved out in
`NOTICE`, which is kept separate so GitHub can identify the MIT license.

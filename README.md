# onehop

Interactive explainer for *Attention Is All You Need* (Vaswani et al., 2017).

## The model

`train.py` trains the paper's architecture scaled down — 2 encoder + 2
decoder layers, d_model 48, 4 heads, d_k = d_v = 12, d_ff 128, post-LN,
sinusoidal positional encoding, tied embeddings — on character-level date
normalization (`3 march 2012` → `2012-03-03`). 109,376 parameters, stored
as fp16 in `model.bin` (214 KB). Fixed seed (1337), reproducible.

**Final accuracy: 100.0% exact match** on a held-out validation set of
3,000 dates (uniform over all nine input formats), measured *after* fp16
quantization — the accuracy of exactly the weights the browser loads.

## Parity

`golden.json` stores PyTorch's intermediate activations at every
sub-layer for three inputs, computed in float64 from the fp16 weights.
`test/parity.mjs` checks the hand-written JS forward pass
(`js/model.mjs`) against them to within 1e-4. It must pass before any
figure work.

## Commands

```
python train.py               retrain (rarely needed; weights are committed)
node test/parity.mjs          verify JS matches PyTorch
python -m http.server 8000    serve locally
npx playwright test           e2e smoke tests
```

## Deploying

Static files only — sync the repo (minus `node_modules`, `test`,
`.venv`, `train.py`, the PDF) to S3 behind CloudFront. Suggested
headers: `Cache-Control: public, max-age=31536000, immutable` for
`model.bin`, `fonts/*`, and `golden.json` (they change only with a
retrain), and `public, max-age=300` for `index.html` and `js/*`.
`golden.json` is only needed by the parity test and can be excluded
from the deployment entirely.

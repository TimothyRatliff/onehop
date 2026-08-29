// onehop Phase 1 — the inference engine.
//
// A hand-written forward pass for the toy date-normalization transformer
// trained by train.py. No inference library: the site's purpose is exposing
// intermediates, so every operation here is a named, observable step.
//
// Math is commented against section numbers of Vaswani et al.,
// "Attention Is All You Need" (NIPS 2017).
//
// The forward pass returns a full trace object whose shape mirrors
// golden.json exactly; test/parity.mjs compares the two to 1e-4.
//
// Everything works on plain nested arrays (number[][]) — at 2 layers and
// d_model 48 a forward pass is ~1 ms, and plain arrays keep every value
// addressable by name for the figures.

// ---------------------------------------------------------------- weights

/** Decode one IEEE 754 half-precision value (stored as uint16). */
export function halfToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;            // subnormal
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

/**
 * Build the model from config.json content and the raw model.bin buffer.
 * Each manifest entry becomes a nested array under params[name]:
 * 2-D shapes as number[out][in] (PyTorch nn.Linear layout), 1-D as number[].
 */
export function loadModel(config, buffer) {
  const u16 = new Uint16Array(buffer);
  const params = {};
  for (const { name, shape, offset } of config.manifest) {
    const n = shape.reduce((a, b) => a * b, 1);
    const flat = new Float64Array(n);
    for (let i = 0; i < n; i++) flat[i] = halfToFloat(u16[offset + i]);
    if (shape.length === 1) {
      params[name] = Array.from(flat);
    } else {
      const [rows, cols] = shape;
      const m = new Array(rows);
      for (let r = 0; r < rows; r++) {
        m[r] = Array.from(flat.subarray(r * cols, (r + 1) * cols));
      }
      params[name] = m;
    }
  }
  return { config, params, pe: sinusoidalPE(config.pe_max, config.d_model) };
}

// ---------------------------------------------------------------- tokenizer

/** Lowercase and encode a string; unknown characters are dropped. */
export function encodeText(config, str) {
  const stoi = new Map(config.vocab.map((c, i) => [c, i]));
  const ids = [];
  for (const ch of str.toLowerCase()) {
    if (stoi.has(ch)) ids.push(stoi.get(ch));
  }
  return ids;
}

export function idsToText(config, ids) {
  return ids
    .filter((i) => i > config.eos_id)
    .map((i) => config.vocab[i])
    .join("");
}

// ---------------------------------------------------------------- math

/** y = x W^T + b, with W as number[out][in] (PyTorch Linear layout). */
function linear(x, W, b) {
  const out = new Array(x.length);
  for (let t = 0; t < x.length; t++) {
    const xt = x[t];
    const row = new Array(W.length);
    for (let o = 0; o < W.length; o++) {
      const wo = W[o];
      let s = b[o];
      for (let i = 0; i < wo.length; i++) s += xt[i] * wo[i];
      row[o] = s;
    }
    out[t] = row;
  }
  return out;
}

function addRows(a, b) {
  return a.map((row, t) => row.map((v, i) => v + b[t][i]));
}

/** LayerNorm (S3.1 wrapper): (x - mean) / sqrt(var + eps) * g + b.
 * Variance is the biased estimator (divide by N), matching PyTorch. */
function layerNorm(x, g, b, eps) {
  return x.map((row) => {
    const n = row.length;
    let mean = 0;
    for (const v of row) mean += v;
    mean /= n;
    let variance = 0;
    for (const v of row) variance += (v - mean) ** 2;
    variance /= n;
    const inv = 1 / Math.sqrt(variance + eps);
    return row.map((v, i) => (v - mean) * inv * g[i] + b[i]);
  });
}

function softmaxRow(row) {
  let max = -Infinity;
  for (const v of row) if (v > max) max = v;
  let sum = 0;
  const e = row.map((v) => {
    const x = Math.exp(v - max);
    sum += x;
    return x;
  });
  return e.map((v) => v / sum);
}

function relu(x) {
  return x.map((row) => row.map((v) => (v > 0 ? v : 0)));
}

/** Sinusoidal positional encoding, S3.5:
 * PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...). */
export function sinusoidalPE(maxLen, d) {
  const pe = new Array(maxLen);
  for (let pos = 0; pos < maxLen; pos++) {
    const row = new Array(d);
    for (let i = 0; i < d; i += 2) {
      const angle = pos / 10000 ** (i / d);
      row[i] = Math.sin(angle);
      row[i + 1] = Math.cos(angle);
    }
    pe[pos] = row;
  }
  return pe;
}

// ---------------------------------------------------------------- attention

/**
 * Multi-head scaled dot-product attention, S3.2.
 * xq: [Tq][d], xkv: [Tk][d]. mask: null or additive [Tq][Tk].
 * Returns every intermediate: q/k/v/scores/weights/heads are per-head
 * ([h][T][...]), concat and out are [Tq][d].
 */
function mha(model, prefix, xq, xkv, mask, headMask) {
  const { n_heads: H, d_k: dk } = model.config;
  const p = model.params;
  const qf = linear(xq, p[`${prefix}.wq.w`], p[`${prefix}.wq.b`]);
  const kf = linear(xkv, p[`${prefix}.wk.w`], p[`${prefix}.wk.b`]);
  const vf = linear(xkv, p[`${prefix}.wv.w`], p[`${prefix}.wv.b`]);
  const slice = (m, h) => m.map((row) => row.slice(h * dk, (h + 1) * dk));
  const scale = 1 / Math.sqrt(dk);

  const q = [], k = [], v = [], scores = [], weights = [], heads = [];
  for (let h = 0; h < H; h++) {
    const qh = slice(qf, h), kh = slice(kf, h), vh = slice(vf, h);
    // eq. (1): softmax(Q K^T / sqrt(d_k)) V
    const sh = qh.map((qrow, tq) =>
      kh.map((krow, tk) => {
        // Masked entries are replaced by the -inf sentinel (not added to),
        // matching train.py's masked_fill exactly.
        if (mask && mask[tq][tk] !== 0) return mask[tq][tk];
        let s = 0;
        for (let i = 0; i < dk; i++) s += qrow[i] * krow[i];
        return s * scale;
      }),
    );
    const wh = sh.map(softmaxRow);
    const oh = wh.map((wrow) => {
      const acc = new Array(dk).fill(0);
      for (let tk = 0; tk < wrow.length; tk++) {
        const w = wrow[tk], vrow = vh[tk];
        for (let i = 0; i < dk; i++) acc[i] += w * vrow[i];
      }
      return acc;
    });
    q.push(qh); k.push(kh); v.push(vh);
    scores.push(sh); weights.push(wh); heads.push(oh);
  }
  // Ablation hook (module 6): zero a head's output before concat. Not part
  // of the paper's forward pass — figures use it to show what a head
  // contributes by removing it.
  if (headMask) {
    for (let h = 0; h < H; h++) {
      if (headMask[h] === false || headMask[h] === 0) {
        heads[h] = heads[h].map((row) => row.map(() => 0));
      }
    }
  }
  // S3.2.2: Concat(head_1..head_h) W^O
  const concat = xq.map((_, t) => heads.flatMap((oh) => oh[t]));
  const out = linear(concat, p[`${prefix}.wo.w`], p[`${prefix}.wo.b`]);
  return { q, k, v, scores, weights, heads, concat, out };
}

function ffn(model, prefix, x) {
  const p = model.params;
  // S3.3: FFN(x) = max(0, x W1 + b1) W2 + b2
  const pre = linear(x, p[`${prefix}.w1.w`], p[`${prefix}.w1.b`]);
  const hidden = relu(pre);
  const out = linear(hidden, p[`${prefix}.w2.w`], p[`${prefix}.w2.b`]);
  return { pre, hidden, out };
}

function causalMask(T, neg) {
  // S3.2.3: mask out connections to future positions
  return Array.from({ length: T }, (_, tq) =>
    Array.from({ length: T }, (_, tk) => (tk > tq ? neg : 0)),
  );
}

// ---------------------------------------------------------------- forward

function embed(model, ids) {
  const { d_model: d } = model.config;
  const scale = Math.sqrt(d); // S3.4: embeddings multiplied by sqrt(d_model)
  const E = model.params.embed;
  const tok_emb = ids.map((id) => E[id].map((v) => v * scale));
  const x0 = tok_emb.map((row, t) => row.map((v, i) => v + model.pe[t][i]));
  return { tok_emb, x0 };
}

/**
 * Encoder stack. Returns the full trace; trace.out is the encoder output.
 * opts.headMasks: {"enc0.attn": [1,1,0,1], ...} zeroes chosen heads
 * (module 6's ablation). Defaults leave the forward pass exactly as
 * verified by test/parity.mjs.
 */
export function encodeSrc(model, srcIds, opts = {}) {
  const cfg = model.config;
  const tr = { ...embed(model, srcIds), layers: [] };
  let x = tr.x0;
  for (let l = 0; l < cfg.n_enc; l++) {
    const lt = {};
    const attn = mha(model, `enc${l}.attn`, x, x, null,
      opts.headMasks?.[`enc${l}.attn`]);
    lt.self_attn = attn;
    lt.res1 = addRows(x, attn.out); // post-LN residual, S3.1
    lt.ln1 = layerNorm(lt.res1, model.params[`enc${l}.ln1.g`],
      model.params[`enc${l}.ln1.b`], cfg.ln_eps);
    lt.ffn = ffn(model, `enc${l}.ffn`, lt.ln1);
    lt.res2 = addRows(lt.ln1, lt.ffn.out);
    lt.ln2 = layerNorm(lt.res2, model.params[`enc${l}.ln2.g`],
      model.params[`enc${l}.ln2.b`], cfg.ln_eps);
    tr.layers.push(lt);
    x = lt.ln2;
  }
  tr.out = x;
  return tr;
}

/**
 * Decoder stack over the full target prefix (no KV cache — the model is
 * tiny and recomputing keeps every step's full trace observable).
 * opts.causal: false disables the causal mask (module 5's cheat mode —
 * never used in real decoding). opts.headMasks as in encodeSrc, keyed
 * "dec0.self" / "dec0.cross".
 */
export function decoderPass(model, tgtIds, encOut, opts = {}) {
  const cfg = model.config;
  const tr = { ...embed(model, tgtIds), layers: [] };
  const causal = opts.causal === false
    ? null
    : causalMask(tgtIds.length, cfg.neg_inf);
  let x = tr.x0;
  for (let l = 0; l < cfg.n_dec; l++) {
    const lt = {};
    lt.self_attn = mha(model, `dec${l}.self`, x, x, causal,
      opts.headMasks?.[`dec${l}.self`]);
    lt.res1 = addRows(x, lt.self_attn.out);
    lt.ln1 = layerNorm(lt.res1, model.params[`dec${l}.ln1.g`],
      model.params[`dec${l}.ln1.b`], cfg.ln_eps);
    lt.cross_attn = mha(model, `dec${l}.cross`, lt.ln1, encOut, null,
      opts.headMasks?.[`dec${l}.cross`]);
    lt.res2 = addRows(lt.ln1, lt.cross_attn.out);
    lt.ln2 = layerNorm(lt.res2, model.params[`dec${l}.ln2.g`],
      model.params[`dec${l}.ln2.b`], cfg.ln_eps);
    lt.ffn = ffn(model, `dec${l}.ffn`, lt.ln2);
    lt.res3 = addRows(lt.ln2, lt.ffn.out);
    lt.ln3 = layerNorm(lt.res3, model.params[`dec${l}.ln3.g`],
      model.params[`dec${l}.ln3.b`], cfg.ln_eps);
    tr.layers.push(lt);
    x = lt.ln3;
  }
  tr.out = x;
  return tr;
}

/** Output projection tied to the embedding matrix, S3.4. */
export function projectLogits(model, decOut) {
  const E = model.params.embed;
  return decOut.map((row) =>
    E.map((erow) => {
      let s = 0;
      for (let i = 0; i < row.length; i++) s += row[i] * erow[i];
      return s;
    }),
  );
}

/**
 * Full teacher-forced pass: encoder over srcIds, decoder over tgtInIds
 * (BOS-prefixed target). Trace shape mirrors golden.json cases.
 */
export function forward(model, srcIds, tgtInIds, opts = {}) {
  const encoder = encodeSrc(model, srcIds, opts);
  const decoder = decoderPass(model, tgtInIds, encoder.out, opts);
  const logits = projectLogits(model, decoder.out);
  return { encoder, decoder, logits };
}

/**
 * One greedy step: given the encoder trace and the target prefix, return
 * the decoder trace, last-position logits, and the chosen token.
 * Figures step decoding one token at a time through this.
 */
export function decodeStep(model, encTrace, prefixIds, opts = {}) {
  const decoder = decoderPass(model, prefixIds, encTrace.out, opts);
  const logits = projectLogits(model, [decoder.out[decoder.out.length - 1]])[0];
  let chosen = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[chosen]) chosen = i;
  return { decoder, logits, chosen };
}

/** Greedy decode. Returns the output text plus every step's trace. */
export function greedyDecode(model, srcIds, maxLen, opts = {}) {
  const cfg = model.config;
  const limit = maxLen ?? cfg.tgt_len + 1;
  const encoder = encodeSrc(model, srcIds, opts);
  const prefix = [cfg.bos_id];
  const steps = [];
  for (let i = 0; i < limit; i++) {
    const step = decodeStep(model, encoder, [...prefix], opts);
    steps.push(step);
    if (step.chosen === cfg.eos_id) break;
    prefix.push(step.chosen);
  }
  return { encoder, steps, ids: prefix.slice(1),
           text: idsToText(cfg, prefix) };
}

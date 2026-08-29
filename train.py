#!/usr/bin/env python
"""onehop Phase 0 -- train the toy date-normalization transformer.

Architecture: Vaswani et al., "Attention Is All You Need" (NIPS 2017),
scaled down. Encoder-decoder, sinusoidal positional encoding (S3.5),
multi-head scaled dot-product attention (S3.2), post-LN residual
sub-layers (S3.1), position-wise FFN (S3.3), tied embedding / output
projection with sqrt(d_model) embedding scale (S3.4).

Task: character-level date normalization. "3 march 2012" -> "2012-03-03".
Inputs are lowercased; the JS front end lowercases user input to match.

Run:  python train.py
Out:  model.bin    flat float16 weights, layout described by config.json
      config.json  architecture + vocab + weight manifest
      golden.json  exact intermediate activations for parity testing

Weights are quantized to fp16 *before* golden.json is generated, so the
golden activations correspond exactly to the weights the browser loads.
"""

import json
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- config

SEED = 1337
D_MODEL = 48
N_HEADS = 4
D_K = D_MODEL // N_HEADS          # 12; d_k = d_v (S3.2.2)
D_FF = 128
N_ENC = 2
N_DEC = 2
MAX_SRC = 40                      # "the twenty-seventh of september 1994" = 36
TGT_LEN = 10                      # "yyyy-mm-dd"
DEC_LEN = TGT_LEN + 1             # BOS + 10 in -> 10 chars + EOS out
PE_MAX = 64
LN_EPS = 1e-5
DROPOUT = 0.1                     # residual dropout (S5.4), training only
NEG = -1e9                        # additive mask value; softmax -> exactly 0

BATCH = 128
EPOCHS = 48
LR = 1.5e-3          # peak; linear warmup then cosine decay
WARMUP = 500         # steps
N_TRAIN = 88000
VAL_N = 3000

random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------- vocab

PAD, BOS, EOS = 0, 1, 2
SPECIALS = ["<pad>", "<bos>", "<eos>"]
CHARS = list("0123456789abcdefghijklmnopqrstuvwxyz ,-./")
VOCAB = SPECIALS + CHARS
V = len(VOCAB)
STOI = {c: i for i, c in enumerate(VOCAB)}


def encode(s):
    return [STOI[c] for c in s]


# ---------------------------------------------------------------- data

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
MON3 = [m[:3] for m in MONTHS]
UNITS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
         "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
         "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
         "nineteenth"]
ORD = {i + 1: w for i, w in enumerate(UNITS)}
ORD[20] = "twentieth"
for i in range(21, 30):
    ORD[i] = "twenty-" + UNITS[i - 21]
ORD[30] = "thirtieth"
ORD[31] = "thirty-first"
DAYS_IN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


# Two-digit years: 00-29 -> 20xx, 30-99 -> 19xx. Year range 1930-2029
# makes that rule a bijection, so mm/dd/yy stays unambiguous.
FORMATS = [
    lambda y, m, d: f"{d} {MONTHS[m-1]} {y}",
    lambda y, m, d: f"{MONTHS[m-1]} {d} {y}",
    lambda y, m, d: f"{MON3[m-1]} {d}, {y}",
    lambda y, m, d: f"{d} {MON3[m-1]} {y}",
    lambda y, m, d: f"{m:02d}/{d:02d}/{y % 100:02d}",
    lambda y, m, d: f"{m:02d}/{d:02d}/{y}",
    lambda y, m, d: f"the {ORD[d]} of {MONTHS[m-1]} {y}",
    lambda y, m, d: f"{y}.{m:02d}.{d:02d}",
    lambda y, m, d: f"{y}-{m:02d}-{d:02d}",
]


# Training oversamples the formats where the year sits at a variable
# position (long month names, ordinal words) -- error analysis showed the
# model transposes year digits exactly there. Validation stays uniform.
FORMAT_WEIGHTS = [2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 3.0, 0.5, 0.5]


def make_dataset(n, rng, weights, seen):
    out = []
    while len(out) < n:
        y = rng.randint(1930, 2029)
        m = rng.randint(1, 12)
        dmax = 29 if (m == 2 and leap(y)) else DAYS_IN[m - 1]
        d = rng.randint(1, dmax)
        if weights is None:
            f = rng.randrange(len(FORMATS))
        else:
            f = rng.choices(range(len(FORMATS)), weights=weights)[0]
        key = (y, m, d, f)
        if key in seen:
            continue
        seen.add(key)
        src = FORMATS[f](y, m, d)
        tgt = f"{y}-{m:02d}-{d:02d}"
        assert len(src) <= MAX_SRC, src
        out.append((src, tgt))
    return out


# ---------------------------------------------------------------- trace helpers


def _rnd(o):
    if isinstance(o, list):
        return [_rnd(x) for x in o]
    return float(f"{o:.8g}")


def L(t):
    """Tensor -> nested lists, batch dim squeezed, 8 significant digits."""
    return _rnd(t.detach().squeeze(0).tolist())


# ---------------------------------------------------------------- model


class MHA(nn.Module):
    """Multi-head scaled dot-product attention, S3.2. Explicit per-matrix
    projections so every intermediate has a name."""

    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(D_MODEL, D_MODEL)
        self.wk = nn.Linear(D_MODEL, D_MODEL)
        self.wv = nn.Linear(D_MODEL, D_MODEL)
        self.wo = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, xq, xkv, mask=None, tr=None, key=None):
        B, Tq, _ = xq.shape
        Tk = xkv.size(1)
        q = self.wq(xq).view(B, Tq, N_HEADS, D_K).transpose(1, 2)   # B,h,Tq,dk
        k = self.wk(xkv).view(B, Tk, N_HEADS, D_K).transpose(1, 2)
        v = self.wv(xkv).view(B, Tk, N_HEADS, D_K).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D_K)           # eq. (1)
        if mask is not None:
            # Replace (not add): masked entries become exactly NEG, so the
            # JS engine can reproduce them bit-comparably.
            scores = scores.masked_fill(mask < 0, NEG)
        w = F.softmax(scores, dim=-1)
        heads = w @ v                                               # B,h,Tq,dv
        concat = heads.transpose(1, 2).reshape(B, Tq, D_MODEL)
        out = self.wo(concat)
        if tr is not None:
            tr[key] = {"q": L(q), "k": L(k), "v": L(v), "scores": L(scores),
                       "weights": L(w), "heads": L(heads),
                       "concat": L(concat), "out": L(out)}
        return out


class FFN(nn.Module):
    """Position-wise feed-forward, S3.3: max(0, xW1+b1)W2+b2."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(D_MODEL, D_FF)
        self.w2 = nn.Linear(D_FF, D_MODEL)

    def forward(self, x, tr=None):
        pre = self.w1(x)
        hidden = F.relu(pre)
        out = self.w2(hidden)
        if tr is not None:
            tr["ffn"] = {"pre": L(pre), "hidden": L(hidden), "out": L(out)}
        return out


class EncLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MHA()
        self.ln1 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.ffn = FFN()
        self.ln2 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, mask, tr=None):
        a = self.attn(x, x, mask, tr, "self_attn")
        r1 = x + self.drop(a)                       # post-LN residual, S3.1
        x = self.ln1(r1)
        if tr is not None:
            tr["res1"], tr["ln1"] = L(r1), L(x)
        f = self.ffn(x, tr)
        r2 = x + self.drop(f)
        x = self.ln2(r2)
        if tr is not None:
            tr["res2"], tr["ln2"] = L(r2), L(x)
        return x


class DecLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MHA()
        self.ln1 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.cross_attn = MHA()
        self.ln2 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.ffn = FFN()
        self.ln3 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, enc_out, causal, src_mask, tr=None):
        a = self.self_attn(x, x, causal, tr, "self_attn")
        r1 = x + self.drop(a)
        x = self.ln1(r1)
        if tr is not None:
            tr["res1"], tr["ln1"] = L(r1), L(x)
        c = self.cross_attn(x, enc_out, src_mask, tr, "cross_attn")
        r2 = x + self.drop(c)
        x = self.ln2(r2)
        if tr is not None:
            tr["res2"], tr["ln2"] = L(r2), L(x)
        f = self.ffn(x, tr)
        r3 = x + self.drop(f)
        x = self.ln3(r3)
        if tr is not None:
            tr["res3"], tr["ln3"] = L(r3), L(x)
        return x


def sinusoidal_pe(max_len, d):
    """S3.5, computed elementwise in float64 then cast, so the JS side can
    reproduce it with the same plain formula."""
    pe = torch.zeros(max_len, d)
    for pos in range(max_len):
        for i in range(0, d, 2):
            angle = pos / (10000.0 ** (i / d))
            pe[pos, i] = math.sin(angle)
            pe[pos, i + 1] = math.cos(angle)
    return pe


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D_MODEL)
        self.enc = nn.ModuleList(EncLayer() for _ in range(N_ENC))
        self.dec = nn.ModuleList(DecLayer() for _ in range(N_DEC))
        self.drop = nn.Dropout(DROPOUT)
        self.register_buffer("pe", sinusoidal_pe(PE_MAX, D_MODEL))

    def encode(self, src, mask=None, tr=None):
        T = src.size(1)
        emb = self.embed(src) * math.sqrt(D_MODEL)   # S3.4
        x = self.drop(emb + self.pe[:T])
        if tr is not None:
            tr["tok_emb"], tr["x0"] = L(emb), L(emb + self.pe[:T])
            tr["layers"] = []
        for layer in self.enc:
            lt = {} if tr is not None else None
            x = layer(x, mask, lt)
            if tr is not None:
                tr["layers"].append(lt)
        if tr is not None:
            tr["out"] = L(x)
        return x

    def decode(self, tgt, enc_out, src_mask=None, tr=None):
        T = tgt.size(1)
        causal = torch.full((T, T), NEG).triu(1)     # S3.2.3: mask future
        emb = self.embed(tgt) * math.sqrt(D_MODEL)
        x = self.drop(emb + self.pe[:T])
        if tr is not None:
            tr["tok_emb"], tr["x0"] = L(emb), L(emb + self.pe[:T])
            tr["layers"] = []
        for layer in self.dec:
            lt = {} if tr is not None else None
            x = layer(x, enc_out, causal, src_mask, lt)
            if tr is not None:
                tr["layers"].append(lt)
        if tr is not None:
            tr["out"] = L(x)
        return x

    def forward(self, src, tgt, src_mask=None):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask)
        return dec_out @ self.embed.weight.T         # tied projection, S3.4


# ---------------------------------------------------------------- batching


def pad_batch(srcs):
    T = max(len(s) for s in srcs)
    ids = torch.full((len(srcs), T), PAD, dtype=torch.long)
    for i, s in enumerate(srcs):
        ids[i, :len(s)] = torch.tensor(encode(s))
    mask = (ids == PAD).view(len(srcs), 1, 1, T).float() * NEG
    return ids, mask


def tgt_batch(tgts):
    dec_in = torch.full((len(tgts), DEC_LEN), PAD, dtype=torch.long)
    labels = torch.full((len(tgts), DEC_LEN), PAD, dtype=torch.long)
    for i, t in enumerate(tgts):
        ids = encode(t)
        dec_in[i] = torch.tensor([BOS] + ids)
        labels[i] = torch.tensor(ids + [EOS])
    return dec_in, labels


@torch.no_grad()
def exact_match(model, data, batch=256):
    model.eval()
    correct = 0
    for i in range(0, len(data), batch):
        chunk = data[i:i + batch]
        src, mask = pad_batch([s for s, _ in chunk])
        _, labels = tgt_batch([t for _, t in chunk])
        enc_out = model.encode(src, mask)
        ys = torch.full((len(chunk), 1), BOS, dtype=torch.long)
        for _ in range(DEC_LEN):
            dec_out = model.decode(ys, enc_out, mask)
            nxt = (dec_out[:, -1] @ model.embed.weight.T).argmax(-1)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        correct += (ys[:, 1:] == labels).all(dim=1).sum().item()
    return correct / len(data)


@torch.no_grad()
def greedy_str(model, src_str):
    src = torch.tensor([encode(src_str)])
    enc_out = model.encode(src)
    ys = [BOS]
    for _ in range(DEC_LEN):
        dec_out = model.decode(torch.tensor([ys]), enc_out)
        nxt = (dec_out[0, -1] @ model.embed.weight.T).argmax().item()
        if nxt == EOS:
            break
        ys.append(nxt)
    return "".join(VOCAB[i] for i in ys[1:])


# ---------------------------------------------------------------- train

def train():
    rng = random.Random(SEED)
    seen = set()
    val = make_dataset(VAL_N, rng, None, seen)       # uniform over formats
    tr_data = make_dataset(N_TRAIN, rng, FORMAT_WEIGHTS, seen)
    print(f"dataset: {len(tr_data)} train / {len(val)} val, vocab {V}")

    model = Model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params} ({n_params * 2 / 1024:.0f} KB fp16)")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    steps_per_epoch = math.ceil(len(tr_data) / BATCH)
    total_steps = EPOCHS * steps_per_epoch
    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng.shuffle(tr_data)
        total, nb = 0.0, 0
        for i in range(0, len(tr_data), BATCH):
            step += 1
            if step <= WARMUP:
                lr = LR * step / WARMUP
            else:
                t = (step - WARMUP) / (total_steps - WARMUP)
                lr = LR * 0.5 * (1 + math.cos(math.pi * t))
            for grp in opt.param_groups:
                grp["lr"] = lr
            chunk = tr_data[i:i + BATCH]
            src, mask = pad_batch([s for s, _ in chunk])
            dec_in, labels = tgt_batch([t for _, t in chunk])
            logits = model(src, dec_in, mask)
            loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if epoch % 4 == 0 or epoch == EPOCHS:
            acc = exact_match(model, val)
            print(f"epoch {epoch:2d}  loss {total / nb:.4f}  "
                  f"val exact {acc:.4f}  lr {lr:.2e}")
        else:
            print(f"epoch {epoch:2d}  loss {total / nb:.4f}")

    return model, val


# ---------------------------------------------------------------- export


def named_tensors(model):
    """(name, tensor) pairs in the fixed model.bin layout order. Single
    source of truth for both export and reload."""
    def mha(prefix, m):
        for nm, lin in [("wq", m.wq), ("wk", m.wk), ("wv", m.wv), ("wo", m.wo)]:
            yield f"{prefix}.{nm}.w", lin.weight
            yield f"{prefix}.{nm}.b", lin.bias

    def ln(prefix, module):
        yield f"{prefix}.g", module.weight
        yield f"{prefix}.b", module.bias

    def ffnp(prefix, f):
        yield f"{prefix}.w1.w", f.w1.weight
        yield f"{prefix}.w1.b", f.w1.bias
        yield f"{prefix}.w2.w", f.w2.weight
        yield f"{prefix}.w2.b", f.w2.bias

    yield "embed", model.embed.weight
    for i, l in enumerate(model.enc):
        yield from mha(f"enc{i}.attn", l.attn)
        yield from ln(f"enc{i}.ln1", l.ln1)
        yield from ffnp(f"enc{i}.ffn", l.ffn)
        yield from ln(f"enc{i}.ln2", l.ln2)
    for i, l in enumerate(model.dec):
        yield from mha(f"dec{i}.self", l.self_attn)
        yield from ln(f"dec{i}.ln1", l.ln1)
        yield from mha(f"dec{i}.cross", l.cross_attn)
        yield from ln(f"dec{i}.ln2", l.ln2)
        yield from ffnp(f"dec{i}.ffn", l.ffn)
        yield from ln(f"dec{i}.ln3", l.ln3)


GOLDEN_CASES = [
    ("3 march 2012", "2012-03-03"),
    ("the twenty-seventh of september 1994", "1994-09-27"),
    ("03/03/12", "2012-03-03"),
]


def write_golden(model, path="golden.json"):
    """Golden traces: teacher-forced full pass + greedy decode string.

    Computed in float64 from the fp16-quantized weights: the JS engine
    computes in float64 too, so golden must be the exact math of the
    shipped weights, not its float32 approximation — float32 rounding in
    the trained model's larger attention logits alone approaches the 1e-4
    parity tolerance.
    """
    model = model.double().eval()
    golden = {"tolerance": 1e-4, "cases": []}
    with torch.no_grad():
        for src_str, tgt_str in GOLDEN_CASES:
            src = torch.tensor([encode(src_str)])
            tgt_in = torch.tensor([[BOS] + encode(tgt_str)])
            enc_tr, dec_tr = {}, {}
            enc_out = model.encode(src, None, enc_tr)
            dec_out = model.decode(tgt_in, enc_out, None, dec_tr)
            logits = dec_out @ model.embed.weight.T
            golden["cases"].append({
                "src": src_str,
                "expected": tgt_str,
                "src_ids": encode(src_str),
                "tgt_in_ids": [BOS] + encode(tgt_str),
                "encoder": enc_tr,
                "decoder": dec_tr,
                "logits": L(logits),
                "greedy": greedy_str(model, src_str),
            })
    with open(path, "w") as f:
        json.dump(golden, f)
    import os
    print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB")
    for c in golden["cases"]:
        flag = "" if c["greedy"] == c["expected"] else "  (MISMATCH)"
        print(f"  {c['src']!r} -> {c['greedy']!r}{flag}")


def export(model, val):
    # Quantize weights to fp16 in place FIRST, so model.bin, golden.json,
    # and the reported accuracy all describe the same model.
    for p in model.parameters():
        p.data = p.data.half().float()
    acc16 = exact_match(model, val)
    print(f"val exact match after fp16 quantization: {acc16:.4f}")

    manifest = []
    buf = bytearray()
    for name, tensor in named_tensors(model):
        t = tensor.detach().to(torch.float16).contiguous()
        manifest.append({"name": name, "shape": list(t.shape),
                         "offset": len(buf) // 2})
        buf.extend(t.numpy().tobytes())

    with open("model.bin", "wb") as f:
        f.write(bytes(buf))
    print(f"model.bin: {len(buf) / 1024:.0f} KB")

    config = {
        "d_model": D_MODEL, "n_heads": N_HEADS, "d_k": D_K, "d_ff": D_FF,
        "n_enc": N_ENC, "n_dec": N_DEC,
        "vocab": VOCAB, "pad_id": PAD, "bos_id": BOS, "eos_id": EOS,
        "max_src": MAX_SRC, "tgt_len": TGT_LEN, "pe_max": PE_MAX,
        "pe_base": 10000, "ln_eps": LN_EPS, "neg_inf": NEG,
        "dtype": "float16", "seed": SEED,
        "lowercase_input": True,
        "two_digit_year_rule": "00-29 -> 20xx, 30-99 -> 19xx",
        "val_exact_match": acc16,
        "n_params": sum(p.numel() for p in model.parameters()),
        "manifest": manifest,
    }
    with open("config.json", "w") as f:
        json.dump(config, f, indent=1)

    write_golden(model)
    return acc16


if __name__ == "__main__":
    model, val = train()
    acc = export(model, val)
    print(f"done. final val exact match {acc:.4f}")

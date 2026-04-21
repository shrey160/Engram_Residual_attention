# PLAN.md — Tiny Qwen3-Style MoE + Engram + AttnRes

Implementation plan for a ~0.5B-parameter Mixture-of-Experts language model that follows the Qwen3 architecture, adds Qwen3-Next's **gated attention**, and integrates the two techniques already implemented in `../codes/`: **Engram** (DeepSeek, conditional memory) and **Block Attention Residuals** (Kimi, learned attention over depth). Trained from scratch on `../test_data/Asimov_the_foundation.pdf`.

This document is written to be picked up by Codex (per `../AGENTS.md`) or by Claude Code. Style conventions, naming, error-handling, and "what not to change" rules from `AGENTS.md` apply unchanged.

---

## 0. Naming & Scope Clarifications

- **"Qwen 3.6"** is not a public release. This plan targets the **Qwen3 family architecture** (Qwen Team, *Qwen3 Technical Report*, arXiv 2505.09388, May 2025) plus the **gated-attention upgrade** introduced in *Gated Attention for Large Language Models* (Qwen team, NeurIPS 2025 oral) and shipped in **Qwen3-Next**. If the user meant a different revision, swap the relevant sections.
- The implementation lives entirely in `codes_moe/`. It reuses `EngramModule` and `BlockAttnRes` by importing from `../codes/` (no duplication). The existing `codes/combined_model.py` is *not* edited; this is a separate, MoE-shaped sibling.
- **Honest sizing note:** the training corpus is one ~2 MB novel (~500K tokens depending on tokenizer). A 500M-parameter model on this corpus will memorize, not generalize. The plan optimizes for a **runnable demo that exercises every component** (MoE routing, gated attention, Engram lookup, AttnRes depth attention) — not for a publishable result. This is called out in §8.

---

## 1. Architectural Target

### 1.1 Backbone — Qwen3 dense layer

| Component | Choice | Source |
|-----------|--------|--------|
| Norm | RMSNorm, pre-norm | Qwen3 §3.1 |
| Positional encoding | RoPE, base 1,000,000 | Qwen3 §3.1 (ABF) |
| Attention | Grouped-Query Attention (GQA) | Qwen3 §3.1 |
| QK stabilization | **QK-Norm** (RMSNorm on Q and K before SDPA) | Qwen3 §3.1, replaces Qwen2's QKV-bias |
| Activation | SwiGLU (already in `combined_model.py`) | Qwen3 §3.1 |
| Tying | LM head tied to token embeddings | standard |

### 1.2 Gated Attention (Qwen3-Next)

After scaled-dot-product attention produces `attn_out ∈ R^{B,T,D}`:

```
g = sigmoid(W_g · x_input)          # input-dependent gate, [B, T, D]
attn_out = g * attn_out             # element-wise gate, then proceed to o_proj
```

- `W_g` is a single linear layer of shape `[D, D]`, initialized so that the gate starts ≈ 0.5 (bias = 0, weight ~ N(0, 0.02)).
- This is the only addition over standard GQA. It removes attention sinks and stabilizes large-scale training. We keep it because the user explicitly asked for the "gated attention block."

### 1.3 MoE Layer (replaces FFN at every layer)

Following Qwen3-MoE design (no shared expert, global-batch balancing), but **scaled down** for the 500M budget:

| Hyperparam | Qwen3-MoE (30B-A3B) | This implementation |
|------------|---------------------|---------------------|
| Total experts | 128 | **16** |
| Top-k routing | 8 | **2** |
| Shared expert | None | None (follow Qwen3) |
| Router | Linear → softmax over experts | Same |
| Load-balance loss | Global-batch | Per-batch (simpler; OK at this scale) |
| Per-expert FFN | SwiGLU, hidden = 4× d | SwiGLU, hidden = 2× d (budget) |

**Load-balance loss** (auxiliary): standard Switch-style:
```
L_balance = α * N_experts * Σ_e (f_e * P_e)
```
where `f_e` = fraction of tokens routed to expert `e`, `P_e` = mean router probability for `e`. `α = 0.01`.

**Router-z loss** (helps stability): `L_z = β * mean(logsumexp(router_logits)^2)`, `β = 0.001`. From ST-MoE (Zoph et al.).

### 1.4 Engram Integration

Reuse `EngramModule` from `../codes/engram.py` unchanged. Place at **2 layers** following the paper's "early + mid-depth" guidance: layers `2` and `num_layers // 2` (= layer 6 in a 12-layer model, layer 8 in a 16-layer model).

Position: **after self-attention, before MoE FFN** — same as `CombinedTransformerLayer` in `../codes/combined_model.py:153`. Engram output is added residually with a learned scalar (`engram_scale`, init 0.1).

`token_ids` must thread through every layer (see CLAUDE.md note in repo root).

### 1.5 Block AttnRes Integration

Reuse `BlockAttnRes` from `../codes/attention_residuals.py` unchanged. Apply **twice per layer** (before attention, before MoE), exactly as in `CombinedTransformerLayer`. `BlockAttnRes.query` zero-init is preserved (load-bearing, per AGENTS.md §"What NOT to Change").

`block_size = 6` (3 transformer layers per AttnRes block) for a 12-layer model.

---

## 2. Parameter Budget (~500M target)

Final sizing — all numbers below are computed from these exact values:

```python
HIDDEN_DIM   = 640
NUM_LAYERS   = 12
NUM_HEADS    = 10              # query heads
NUM_KV_HEADS = 2               # GQA: 5:1 ratio
HEAD_DIM     = 64              # = HIDDEN_DIM // NUM_HEADS
NUM_EXPERTS  = 16
TOP_K        = 2
EXPERT_FFN   = 1280            # 2× hidden
VOCAB_SIZE   = 8192            # BPE on Foundation; small vocab matches small corpus
MAX_SEQ_LEN  = 1024
ENGRAM_LAYERS = {2, 6}
BLOCK_SIZE    = 6              # AttnRes block (sub-layers)
```

### Per-component arithmetic

| Component | Formula | Params |
|-----------|---------|--------|
| Token emb (tied to LM head) | `8192 × 640` | 5.24M |
| Per layer — Attention | Q: `640×640` + K,V: `640×128` each + O: `640×640` + 2 RMSNorms + W_g (gated attn): `640×640` + QK-norm pair | ~1.23M |
| Per layer — Router | `640 × 16` | 10K |
| Per layer — 16 SwiGLU experts | `16 × 3 × 640 × 1280` | 39.32M |
| Per layer — 2× BlockAttnRes | `2 × (640 + 640)` | 2.6K |
| Per layer total | | ~40.6M |
| 12 layers | `12 × 40.6M` | ~487M |
| Final norm + lm_head (tied) | `640` + 0 | <1K |
| **Backbone subtotal** | | **~492M** |
| Engram tables (2 layers, table_size_hint=10007, num_heads=8, engram_dim=64, ngram_range=(2,3)) | per layer: `2 (n-grams) × 8 (heads) × 10007 (table) × 64 (dim)` ≈ 10.25M; 2 layers | ~20.5M |
| Engram fusion (W_K, W_V, ShortConv) per layer | small | ~1.5M |
| **Total (backbone + Engram)** | | **~514M** ≈ 0.5B ✓ |

Engram tables are **offloadable to CPU** per the paper. For activation memory and FLOPs, only top-2 of 16 experts fire per token, so **active params per token ≈ 5.24M (emb) + 12 × (1.23M attn + 2 × 3.93M expert) = ~110M active**. This is a true MoE: 22% activation rate.

### 2.1 If 500M is over-budget on the user's machine

Drop to `NUM_LAYERS=10, NUM_EXPERTS=12, EXPERT_FFN=1024`: ~280M total, ~70M active. Same architecture, smaller knobs.

---

## 3. File Layout (`codes_moe/`)

Create these files in order. Each is small and single-purpose, following AGENTS.md style.

```
codes_moe/
├── PLAN.md                       # this file
├── README.md                     # short user-facing description (write last)
├── config.py                     # all hyperparams as a dataclass (single source of truth)
├── tokenizer.py                  # BPE training + load/save (sentencepiece OR tokenizers lib)
├── data.py                       # PDF → text → token chunks → DataLoader
├── moe_layer.py                  # MoELayer: router + 16 experts + load-balance loss
├── gated_attention.py            # GatedGQA: GQA + RoPE + QK-norm + sigmoid gate
├── qwen3_moe_model.py            # Qwen3MoEBlock + Qwen3MoEModel (no Engram, no AttnRes)
├── combined_moe_model.py         # CombinedQwen3MoE: + Engram + Block AttnRes
├── train.py                      # training loop with AdamW + separate LR groups + cosine schedule
├── evaluate.py                   # perplexity, generation, expert-utilization, gate stats
├── generate.py                   # CLI sampling from a checkpoint (top-k / top-p)
├── tests/
│   ├── test_gated_attention.py   # shapes, causality, gate ≈ 0.5 at init, gradient flow
│   ├── test_moe_layer.py         # routing top-k correctness, load-balance loss decreases, all experts get gradient
│   ├── test_qwen3_moe_model.py   # forward/backward shapes, param count within ±5% of target
│   ├── test_combined_moe.py      # end-to-end: Engram only at designated layers, AttnRes block boundaries fire, single training step decreases loss
│   └── test_data.py              # tokenizer round-trip, dataset chunking is correct length
└── benchmark_moe.py              # train backbone-only vs +Engram vs +AttnRes vs +both for N steps; reuse codes/benchmark.py chart pattern
```

**Why no `__init__.py`:** matches the existing `codes/` directory style — flat, scriptable, `python file.py` runs each module's `__main__` smoke check.

**Imports across directories:** use `sys.path.insert(0, '../codes')` at the top of `combined_moe_model.py` to import `EngramModule` and `BlockAttnRes`. Do NOT copy those modules into `codes_moe/`.

---

## 4. Data Pipeline

### 4.1 PDF extraction (`data.py`)

```python
import pypdf  # or pdfplumber if pypdf gives garbled output
def pdf_to_text(pdf_path: Path) -> str:
    reader = pypdf.PdfReader(pdf_path)
    return "\n".join(p.extract_text() or "" for p in reader.pages)
```

Cleanup pass: strip page numbers, repeated headers, multi-blank-lines, hyphenation artifacts (`word-\nbreak` → `wordbreak`). Save to `codes_moe/cache/foundation.txt` so we don't re-parse.

**Expected output:** ~500K-700K characters of clean prose. Inspect manually before training — Asimov-era PDFs sometimes have OCR noise.

### 4.2 Tokenizer (`tokenizer.py`)

Train **byte-level BPE** with `huggingface tokenizers` library, vocab=8192. Why 8192:
- Corpus is ~100K-150K words; a larger vocab would have many rarely-seen tokens.
- 8192 fits in 13 bits → tight.
- Matches Karpathy minBPE conventions; known-good.

Special tokens: `<|pad|>`, `<|bos|>`, `<|eos|>`, `<|unk|>`. Save to `codes_moe/cache/tokenizer.json`.

If `tokenizers` is heavyweight, fall back to byte-level (vocab=256) and inflate `MAX_SEQ_LEN` to 2048. The model architecture is identical; only `VOCAB_SIZE` changes.

### 4.3 Dataset (`data.py`)

- Single train/val split: 90% / 10% by **chapter boundary** (split on the longest sequence of `\n\n` that separates parts/chapters). Keeps the val set as held-out narrative, not random tokens from the same paragraph.
- Sliding-window chunks of `MAX_SEQ_LEN` tokens with stride = `MAX_SEQ_LEN // 2`. Each chunk yields `(input_ids, labels)` with `labels = input_ids` shifted (standard causal LM).
- `DataLoader(batch_size=8, shuffle=True, drop_last=True)` for training.

---

## 5. Training Pipeline (`train.py`)

### 5.1 Optimizer with three parameter groups

Following Engram paper's recommendation (separate optimizer for embedding tables) plus standard LLM convention:

```python
groups = [
    # Group 1: backbone (attn, MoE, norms) — standard
    {"params": backbone_params, "lr": 3e-4, "weight_decay": 0.1},
    # Group 2: Engram embedding tables — 5x LR, no weight decay (per paper)
    {"params": engram_table_params, "lr": 1.5e-3, "weight_decay": 0.0},
    # Group 3: AttnRes queries — small, init at zero, standard LR
    {"params": attnres_query_params, "lr": 3e-4, "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
```

Detect group membership by name substring (matches `combined_model.py:param_summary` convention): `"engram" in name` → embedding tables, `"attn_res" in name or "mlp_res" in name` → AttnRes queries, else backbone.

### 5.2 Schedule

- Cosine LR decay with linear warmup over first 5% of steps.
- `max_steps`: 5000 for full demo, 200 for smoke test (CLI flag `--smoke`).
- Gradient clipping: `clip_grad_norm_(model.parameters(), 1.0)`.
- Mixed precision: `torch.amp.autocast("cuda", dtype=torch.bfloat16)` if GPU; full fp32 on CPU.

### 5.3 Loss

```
L_total = L_lm + L_balance + L_z
```

`L_lm` is standard cross-entropy over next tokens. `L_balance` and `L_z` are returned by each `MoELayer.forward` and accumulated; the model's `forward` sums them across layers.

### 5.4 Checkpointing

- Every 500 steps: save `{"model": state_dict, "optimizer": opt_state, "step": step, "config": cfg.to_dict()}` to `codes_moe/checkpoints/step_{N}.pt`.
- Keep last 3 + best-by-val-loss.

### 5.5 Logging

Plain prints + `losses.jsonl` line per step with `step, lm_loss, balance_loss, z_loss, lr, expert_util_entropy, mean_engram_gate`. No wandb dependency.

---

## 6. Evaluation (`evaluate.py`)

Five evaluations, run after training:

1. **Validation perplexity.** Compute `exp(mean cross-entropy)` on the held-out chapter split. Report as the headline number.
2. **Sample generation.** Prompt: `"Hari Seldon looked at"` → 200 tokens, top-k=40, top-p=0.9, temperature 0.8. Eyeball coherence.
3. **Expert utilization.** Histogram of per-expert routing fraction over the val set. Healthy MoE has roughly uniform usage; collapse looks like one expert at >50%. Save `expert_util.png`.
4. **Engram gate activation.** For 100 val sequences, log `mean(sigmoid_gate)` per token at each Engram layer. Plot as heatmap (token position × layer). High activation on rare/proper-noun tokens (e.g. "Trantor", "Foundation") = Engram is doing its job.
5. **AttnRes depth weights.** For each layer, extract `softmax(layer.attn_res.query @ RMSNorm(K).T)` averaged over val tokens. Plot `[layer × source_block]` heatmap. Reveals whether early/mid/late layers have specialized attention patterns.

### 6.1 Test plan (`tests/`)

Match `codes/test_*.py` conventions: standalone scripts with `main()`, `[PASS] test_name` prints, AssertionError on failure. No pytest. Run any single test via `python -c "from test_X import test_Y; test_Y()"`.

Coverage:

| File | Tests |
|------|-------|
| `test_gated_attention.py` | (1) output shape, (2) is causal, (3) gate ≈ 0.5 at init, (4) gradient flows to W_g, (5) GQA reduces KV params correctly |
| `test_moe_layer.py` | (1) top-k routing returns exactly k experts, (2) routing weights sum to 1, (3) all experts receive gradient over a small batch, (4) load-balance loss decreases under uniform-routing override, (5) router-z loss is non-negative |
| `test_qwen3_moe_model.py` | (1) forward/backward shapes, (2) param count within ±5% of `cfg.target_params`, (3) tied LM head, (4) loss decreases over 10 training steps on a tiny batch |
| `test_combined_moe.py` | (1) Engram only at designated layers, (2) AttnRes block boundaries fire correctly (`_is_block_boundary` returns True at the right layers), (3) `param_summary()` accounts for backbone/MoE/Engram/AttnRes separately, (4) full training step (forward + backward + step) on tiny batch |
| `test_data.py` | (1) tokenizer encode/decode round-trip on first 100 chars, (2) `__getitem__` returns `(input_ids, labels)` of correct length and `labels == input_ids[1:] + something`, (3) train/val split is disjoint |

---

## 7. Benchmark (`benchmark_moe.py`)

Replicate the structure of `codes/benchmark.py`. Train four variants for `NUM_STEPS=300` on a small batch:

| Variant | Description |
|---------|-------------|
| Backbone-only | Qwen3-MoE, no Engram, no AttnRes |
| +Engram | Backbone + Engram at layers {2, 6} |
| +AttnRes | Backbone + Block AttnRes everywhere |
| +Both | Backbone + Engram + AttnRes |

Output: a results table (init/final/min/avg-last-50 loss, time, steps/s) and a 2-panel matplotlib chart written to `benchmark_moe_chart.png`. Reuse the smoothing function and color scheme from `codes/benchmark.py:271-305`.

**Honesty caveat to print at the bottom of the chart and table:** "Trained on a single novel — results are dominated by memorization, not generalization. See PLAN.md §8."

---

## 8. Risks, Caveats & What Will Break

These should be in the PR description and the README of `codes_moe/`. Codex should not paper over them.

### 8.1 Data starvation

A 500M-param model needs ~10B tokens for Chinchilla-optimal training. The Foundation corpus is ~500K tokens — **20,000× under-trained**. The model will memorize. Validation perplexity will be misleadingly low because the val "chapters" share an author, vocabulary, and characters with the train set. **Treat this as a structural correctness test, not a capability test.**

### 8.2 MoE collapse on tiny data

With 16 experts and so few tokens, the router will likely collapse onto 2-3 experts within a few hundred steps regardless of the load-balance loss. Mitigations:
- Increase `α` (load-balance weight) from `0.01` to `0.05`.
- Reduce `NUM_EXPERTS` to 8.
- Add expert dropout (`p=0.1` on expert outputs during training) — drop only at non-MoE-layer level, not on routing decisions.
- If collapse persists: this is expected, document it honestly in `evaluate.py` output.

### 8.3 Engram on byte/small-vocab tokenizer

Same caveat as in `codes/benchmark.py` and the root README §"Limitations": Engram's tokenizer compression (NFKC/lowercasing → 23% vocab reduction) is reduced to a random projection on small vocabs. The hash + table machinery still works, so the architecture exercises correctly, but the technique's claimed gains will not manifest at this scale.

### 8.4 RoPE base 1M is overkill at seq_len=1024

Qwen3 uses base 1M for 32K+ contexts. At 1024 we could use base 10K. Keeping 1M for architectural fidelity to Qwen3; it costs nothing at inference.

### 8.5 No flash attention

`nn.MultiheadAttention` is fine on CPU and small GPU. If running on a modern GPU, switch to `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)` for ~2× speedup. Don't pull in `flash-attn` package — keeps the dep list at `torch + matplotlib + tokenizers + pypdf`.

### 8.6 PDF extraction quality

`pypdf` may produce noisy text on older typeset PDFs. If output looks garbled, switch to `pdfplumber` (heavier dep but better quality). Verify by reading the first 2KB of `cache/foundation.txt` before proceeding.

---

## 9. Implementation Order (concrete TODO for the executor)

Follow this sequence. Each step is independently runnable and testable.

1. **`config.py`** — define `Qwen3MoEConfig` dataclass with all hyperparams from §2.
2. **`data.py`** — PDF extraction + cleanup. Verify `cache/foundation.txt` looks clean.
3. **`tokenizer.py`** — train BPE on `cache/foundation.txt`. Verify round-trip.
4. **`gated_attention.py`** — GQA + RoPE + QK-Norm + sigmoid gate. Write `tests/test_gated_attention.py` first, make it pass.
5. **`moe_layer.py`** — router + experts + balance loss. Write `tests/test_moe_layer.py` first, make it pass.
6. **`qwen3_moe_model.py`** — assemble backbone (no Engram, no AttnRes). Write `tests/test_qwen3_moe_model.py`, confirm param count.
7. **`combined_moe_model.py`** — wire in Engram (from `../codes/engram.py`) and BlockAttnRes (from `../codes/attention_residuals.py`). Write `tests/test_combined_moe.py`. Run **all** existing tests in `../codes/test_*.py` to confirm we haven't broken sibling code.
8. **`train.py`** — three-group optimizer, cosine schedule, loss aggregation, checkpointing. Run with `--smoke` first (200 steps) to verify loss decreases.
9. **`evaluate.py`** — five evaluations from §6.
10. **`generate.py`** — CLI sampling.
11. **`benchmark_moe.py`** — 4-variant comparison chart.
12. **`README.md`** — user-facing summary, mirrors `../README.md` style. Include the §8 caveats prominently.

After step 7 (smoke-trainable model exists), run a 200-step smoke train end-to-end before continuing to evaluation polish. **Don't write all 12 files then debug at the end.**

---

## 10. Post-Implementation Review Hooks (for Codex)

When Codex reviews:

- **Architecture conformance:** does `gated_attention.py` actually implement the sigmoid-gate-after-SDPA pattern? Are QK-Norm, GQA, and RoPE base 1M correct?
- **MoE correctness:** are routing weights normalized over top-k (not over all experts)? Does the load-balance loss use `f_e * P_e` (Switch Transformer formulation) and not just one factor?
- **Reuse hygiene:** are `EngramModule` and `BlockAttnRes` imported, not copy-pasted?
- **Engram placement:** is Engram applied to `partial_block` (between attention and MoE) like `combined_model.py`, not to the residual stream pre-AttnRes?
- **Three-group optimizer:** do the parameter groups partition the model exactly (no missing params, no double-counted params)?
- **Param count:** does `model.param_summary()` (or equivalent) report total within ±5% of 500M?
- **Test coverage:** every new module has at least one test; tests follow the `codes/test_*.py` standalone-script convention.
- **No silent fallbacks:** PDF extraction failure, tokenizer not found, or checkpoint corruption should raise loudly, not silently fall back to defaults.

---

## References

- Qwen Team, *Qwen3 Technical Report*, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388), May 2025.
- Qiu et al., *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*, NeurIPS 2025 (Qwen team), code: [github.com/qiuzh20/gated_attention](https://github.com/qiuzh20/gated_attention).
- Zoph et al., *ST-MoE: Designing Stable and Transferable Sparse Expert Models* (router-z loss).
- Fedus, Zoph, Shazeer, *Switch Transformer* (load-balance loss formulation).
- DeepSeek, *Conditional Memory via Scalable Lookup* (Engram), [arXiv:2601.07372](https://arxiv.org/abs/2601.07372).
- Kimi / Moonshot AI, *Attention Residuals*, [arXiv:2603.15031](https://arxiv.org/abs/2603.15031).
- Existing implementations to reuse: `../codes/engram.py:EngramModule`, `../codes/attention_residuals.py:BlockAttnRes`, `../codes/combined_model.py:CombinedTransformerLayer` (as a structural reference for layer wiring).

# Engram (DeepSeek) + Attention Residuals (Kimi): Detailed Overview

## 1. Engram: Conditional Memory via Scalable Lookup (DeepSeek)

**Paper:** "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"
**arXiv:** [2601.07372](https://arxiv.org/abs/2601.07372) (January 2026)
**GitHub:** [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)

### Problem

Transformers lack a native primitive for knowledge lookup. MoE scales capacity via conditional
computation, but treats factual recall and complex reasoning identically -- both go through
expensive neural forward passes. Facts that should be O(1) lookups are "simulated" through
deep computation.

### Core Idea: Two Axes of Sparsity

| Axis | Mechanism | Lookup | Cost |
|------|-----------|--------|------|
| Conditional Computation (MoE) | Route tokens through expert subnetworks | Dynamic gating | O(d^2) per expert |
| **Conditional Memory (Engram)** | **Hash-based embedding table lookup** | **Deterministic hashing** | **O(1) per lookup** |

### Architecture

Engram has two phases: **Retrieval** and **Fusion**.

#### Phase 1: Retrieval

**Step 1 - Tokenizer Compression:**
A vocabulary projection collapses raw token IDs into canonical identifiers using NFKC
normalization, lowercasing, accent stripping, etc. Achieves ~23% vocabulary reduction
(128K -> ~99K).

```
P: V -> V',  x'_t = P(x_t)
```

**Step 2 - N-gram Hash Retrieval:**
Form suffix n-grams from compressed tokens:

```
g_{t,n} = (x'_{t-n+1}, ..., x'_t)
```

Hash using multiplicative-XOR:
```
mix = tokens[0] * multipliers[0]
for k in 1..n:
    mix = XOR(mix, tokens[k] * multipliers[k])
z = mix % prime_table_size
```

Retrieve embedding: `e_{t,n,k} = E_{n,k}[z_{t,n,k}]`

Final memory vector (concatenation across n-gram orders and hash heads):
```
e_t = concat_{n=2}^{N} concat_{k=1}^{K} e_{t,n,k}
```

- N-gram range: [2, 3] (bigrams and trigrams)
- Hash heads: K=8 per n-gram order (collision mitigation)
- Table sizes: Prime numbers (unique per head)

#### Phase 2: Fusion

**Step 3 - Context-Aware Gating:**
```
k_t = W_K * e_t
v_t = W_V * e_t

alpha_t = sigmoid(RMSNorm(h_t)^T . RMSNorm(k_t) / sqrt(d))
v_tilde_t = alpha_t * v_t
```

**Step 4 - Depthwise Convolution:**
```
Y = SiLU(Conv1D(RMSNorm(V_tilde))) + V_tilde
```
- Kernel size w=4, dilation = max n-gram order (3)
- Depthwise (grouped) convolution with residual connection

**Step 5 - Multi-branch Integration:**
With hyper-connections (M=4 branches), each branch gets its own gating. The Engram
output is added residually to the hidden states.

### Layer Placement

Engram is inserted at specific layers only -- **Layer 2** (early, for local pattern offloading)
and **Layer 15** (mid-depth, for refinement). NOT every layer.

### Sparsity Allocation

Given total sparse params P_sparse:
```
P_MoE = rho * P_sparse
P_Engram = (1 - rho) * P_sparse
```
Optimal rho ~ 75-80% (U-shaped loss curve). ~20-25% of sparse params go to Engram.

### Key Results (Engram-27B vs MoE-27B, same params & FLOPs)

| Benchmark | MoE-27B | Engram-27B | Delta |
|-----------|---------|------------|-------|
| MMLU | 57.4 | 60.4 | +3.0 |
| BBH | 50.9 | 55.9 | +5.0 |
| NIAH (Multi-Query) | 84.2 | 97.0 | **+12.8** |
| Variable Tracking | 77.0 | 89.0 | **+12.0** |
| GSM8K | 58.4 | 60.6 | +2.2 |

### Inference Advantage

Engram tables can be offloaded to system DRAM (hash indices are deterministic from input
tokens, no GPU needed). A 100B-param table adds only ~2-3% latency overhead via async
prefetch.

---

## 2. Attention Residuals (Kimi / Moonshot AI)

**Paper:** "Attention Residuals"
**arXiv:** [2603.15031](https://arxiv.org/abs/2603.15031) (March 2026)
**GitHub:** [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)

### Problem

Standard residual connections accumulate layer outputs with **fixed unit weights**:
```
h_l = h_{l-1} + f_{l-1}(h_{l-1})
```
Unrolling: `h_l = h_1 + sum_{i=1}^{l-1} f_i(h_i)`

Three specific problems:
1. **No selective access** -- attention and MLP layers receive the same aggregated state
2. **Irreversible loss** -- once blended into the residual stream, earlier representations
   cannot be selectively recovered
3. **Output growth** -- hidden-state magnitudes grow as O(L) with depth (PreNorm dilution)

### Core Idea: Time-Depth Duality

Just as Transformers replaced fixed-weight RNN recurrence over the **sequence** dimension
with attention, AttnRes replaces fixed-weight residual recurrence over the **depth** dimension
with attention.

### Full Attention Residuals

```
h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i
```

where attention weights:
```
alpha_{i->l} = exp(q_l^T . RMSNorm(k_i)) / sum_j exp(q_l^T . RMSNorm(k_j))
```

- `q_l = w_l` -- learned parameter vector per layer (NOT input-dependent)
- `k_i = v_i = h_1` if i=0 (token embedding), else `k_i = v_i = f_i(h_i)` (layer output)
- **Critical:** All query vectors initialized to zero (uniform attention at init = stable training)

### Block Attention Residuals (Practical Variant)

Partition L layers into N blocks of S layers each:

**Intra-block:** Standard residual accumulation within each block
```
b_n = sum_{j in B_n} f_j(h_j)
```

**Inter-block:** Attention over block summaries
```
V = [b_0, b_1, ..., b_{n-1}, b_n^partial]
h_l = softmax(q_l^T . RMSNorm(V)) . V
```

Reduces memory from O(Ld) to O(Nd). **N ~ 8 blocks** is the practical sweet spot.

Each transformer layer applies AttnRes **twice**: once before attention, once before MLP
(separate parameters for each).

### Key Results (48B total / 3B active MoE, 1.4T tokens)

| Benchmark | Baseline | AttnRes | Delta |
|-----------|----------|---------|-------|
| MMLU | 73.5 | 74.6 | +1.1 |
| GPQA-Diamond | 36.9 | 44.4 | **+7.5** |
| BBH | 76.3 | 78.0 | +1.7 |
| Math | 53.5 | 57.1 | +3.6 |
| HumanEval | 59.1 | 62.2 | +3.1 |

**Scaling law:** Block AttnRes matches baseline trained with **1.25x more compute**.
**Overhead:** <4% training, <2% inference.

### Architectural Implication

Standard residuals favor shallower, wider models. AttnRes shifts the optimum toward
**narrower, deeper** models (optimal d_model/L shifts from ~60 to ~45).

---

## 3. How They Can Be Combined

### Complementary Nature

| Dimension | Engram | AttnRes |
|-----------|--------|---------|
| **What it improves** | Information *injection* (new knowledge into the stream) | Information *routing* (how existing layer outputs compose) |
| **Mechanism** | O(1) hash-based memory lookup | Learned attention over depth |
| **Where it acts** | Specific layers (2, 15) | Every layer boundary |
| **Parameter overhead** | Large tables (offloadable to DRAM) | Tiny (one vector per layer) |
| **Strongest gains** | Factual recall, named entities | Reasoning, deep composition |

### Synergy Hypothesis

1. **Engram injects; AttnRes routes.** Engram provides rich factual embeddings at layers 2
   and 15. AttnRes then allows *all subsequent layers* to selectively attend back to those
   Engram-enriched representations -- rather than having them diluted by fixed-weight
   residual accumulation.

2. **Selective memory access.** Without AttnRes, an Engram injection at layer 2 gets
   uniformly mixed into every subsequent layer (including layers that don't need it). With
   AttnRes, layer 20 (doing reasoning) can *downweight* the Engram-heavy layer-2 output
   while layer 10 (doing factual grounding) can *upweight* it.

3. **More effective Engram placement.** AttnRes removes the constraint that Engram layers
   must be carefully chosen -- since any layer can attend back to any block, Engram outputs
   remain accessible regardless of placement.

4. **Sparsity budget optimization.** Engram's ~20-25% sparse parameter allocation is
   orthogonal to AttnRes's negligible parameter cost. They don't compete for the same
   budget.

### Combined Architecture Design

```
Input tokens
    |
    v
[Token Embedding] --> b_0 (initial block representation)
    |
[Layer 1: AttnRes -> Attn -> AttnRes -> MLP]
[Layer 2: AttnRes -> Attn -> AttnRes -> MLP + ENGRAM INJECTION]  <-- Engram Phase
[Layer 3: AttnRes -> Attn -> AttnRes -> MLP]
    ...
[Layer 6: Block boundary] --> b_1
    ...
[Layer 15: AttnRes -> Attn -> AttnRes -> MLP + ENGRAM INJECTION] <-- Engram Phase
    ...
[Layer L: Final output]
```

At Engram layers:
1. Retrieve n-gram embeddings via hash lookup
2. Apply context-aware gating against current hidden state
3. Apply depthwise convolution
4. Add Engram output to hidden state
5. AttnRes at subsequent layers can attend back to this enriched representation

### Expected Benefits of Combination

- **Factual recall**: Engram provides the raw knowledge; AttnRes ensures it persists
  across depth without dilution
- **Reasoning over facts**: AttnRes lets reasoning layers selectively access
  fact-enriched representations from Engram layers
- **Efficiency**: Both add minimal FLOPs -- Engram is O(1) lookup, AttnRes is O(N*d)
  per layer where N ~ 8-10
- **Scalability**: Engram tables scale in DRAM, AttnRes scales with depth -- independent
  scaling axes

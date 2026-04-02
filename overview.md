# Engram (DeepSeek) & Attention Residuals (Kimi): Detailed Overview

---

## 1. Engram -- Conditional Memory via Scalable Lookup (DeepSeek)

**Paper:** "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"
**arXiv:** [2601.07372](https://arxiv.org/abs/2601.07372) (January 12, 2026)
**Authors:** Xin Cheng, Wangding Zeng, Damai Dai, et al.
**Code:** [github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)

### Problem

Transformers treat factual recall and complex reasoning identically -- both go through expensive neural forward passes. Simple facts (e.g., "The capital of France is Paris") that should be retrievable in O(1) time are instead "simulated" through deep computation across many layers. While MoE introduced sparsity via conditional computation, there is no native primitive for direct knowledge lookup.

### Core Idea: Conditional Memory as a Second Sparsity Axis

Engram introduces **conditional memory** alongside conditional computation (MoE):

| Axis | Mechanism | Cost |
|------|-----------|------|
| Conditional Computation (MoE) | Dynamic routing through expert subnetworks | O(d^2) per token |
| Conditional Memory (Engram) | Static embedding lookup via deterministic hashing | O(1) per token |

The name "Engram" comes from neuroscience -- a memory trace stored in brain tissue.

### Architecture: Retrieval + Fusion

**Phase 1 -- Retrieval:**

1. **Tokenizer Compression:** A vocabulary projection collapses raw token IDs into canonical identifiers using NFKC normalization, accent stripping, lowercasing, and whitespace normalization. Achieves ~23% vocabulary reduction (128K -> ~99K).

2. **N-gram Hash Retrieval:** For each position t, form suffix N-grams (bigrams and trigrams). Hash them using a multiplicative-XOR scheme:
   ```
   mix = tokens[0] * multipliers[0]
   for k in 1..n:
       mix = XOR(mix, tokens[k] * multipliers[k])
   z = mix % prime_table_size
   ```
   Retrieve embeddings from K=8 parallel hash heads per N-gram order. Concatenate all retrieved embeddings into a memory vector e_t.

**Phase 2 -- Fusion:**

3. **Context-Aware Gating:** Compute a gating scalar using the hidden state and the retrieved memory:
   ```
   alpha_t = sigmoid(RMSNorm(h_t)^T * RMSNorm(W_K * e_t) / sqrt(d))
   v_tilde_t = alpha_t * (W_V * e_t)
   ```

4. **Depthwise Convolution:** Apply a 1D depthwise convolution with dilation equal to the max N-gram order:
   ```
   Y = SiLU(Conv1D(RMSNorm(v_tilde))) + v_tilde
   ```

5. **Multi-branch Integration:** With manifold-constrained hyper-connections (mHC, M=4 branches), each branch receives its own gated output, added residually to hidden states.

### Layer Placement

Engram is inserted at only **2 layers** (Layer 2 and Layer 15 in a 30-layer model):
- Layer 2: Early, for offloading local pattern reconstruction
- Layer 15: Mid-depth, for refinement with stronger contextual queries

### Sparsity Allocation Law

Given total sparse parameters P_sparse, the optimal split is:
- ~75-80% to MoE experts
- ~20-25% to Engram tables

This follows a U-shaped validation loss curve with optimum at rho ~ 0.75.

### Key Results (Engram-27B vs MoE-27B, iso-param iso-FLOPs)

| Benchmark | MoE-27B | Engram-27B | Delta |
|-----------|---------|------------|-------|
| MMLU | 57.4 | 60.4 | **+3.0** |
| BBH | 50.9 | 55.9 | **+5.0** |
| NIAH (Multi-Query) | 84.2 | 97.0 | **+12.8** |
| Variable Tracking | 77.0 | 89.0 | **+12.0** |
| HumanEval | 37.8 | 40.8 | +3.0 |
| GSM8K | 58.4 | 60.6 | +2.2 |

Engram tables can be offloaded to CPU DRAM with only ~2-3% throughput overhead since hash indices are deterministic.

---

## 2. Attention Residuals (Kimi / Moonshot AI)

**Paper:** "Attention Residuals"
**arXiv:** [2603.15031](https://arxiv.org/abs/2603.15031) (March 16, 2026)
**Authors:** Kimi Team (36+ members), Moonshot AI
**Code:** [github.com/MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)

### Problem

Standard residual connections accumulate all layer outputs with **fixed unit weights**:
```
h_l = h_{l-1} + f_{l-1}(h_{l-1})
```

Unrolled: `h_l = h_1 + sum of all prior layer outputs` -- every layer gets the same uniform mixture.

Three specific problems:
1. **No selective access:** Attention and MLP layers receive the same aggregated state despite needing different information
2. **Irreversible loss:** Once information blends into the residual stream, later layers cannot recover earlier representations
3. **Output growth:** Hidden-state magnitudes grow O(L) with depth (PreNorm dilution)

### Core Idea: Time-Depth Duality

Just as Transformers replaced fixed-weight RNN recurrence over the **sequence** dimension with attention, AttnRes replaces fixed-weight residual recurrence over the **depth** dimension with attention.

### Full Attention Residuals

```
h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i
```

where alpha weights are computed via softmax attention over depth:
```
alpha_{i->l} = exp(q_l^T * RMSNorm(k_i)) / sum_j exp(q_l^T * RMSNorm(k_j))
```

- q_l = learned parameter vector (one per layer, initialized to ZERO)
- k_i = v_i = layer output f_i(h_i) (or token embedding for i=0)

Zero-initialization ensures initial uniform weights, recovering equal-weight averaging at training start.

### Block Attention Residuals (Practical Variant)

Partitions L layers into N blocks of S layers. Within each block, outputs are summed normally. Between blocks, attention is applied:

```
b_n = sum of layer outputs within block n
V = [b_0, b_1, ..., b_{n-1}, partial_current_block]
h = softmax_attention(q_l, RMSNorm(V)) @ V
```

- Reduces memory from O(Ld) to O(Nd)
- N ~ 8 blocks recovers most of the benefit
- Each layer has **two** AttnRes operations: one before attention, one before MLP

### Two-Phase Efficient Inference

**Phase 1 (parallel):** Batch all queries within a block against cached block representations.
**Phase 2 (sequential):** Per-layer, compute intra-block attention on evolving partial sum, merge with Phase 1 via online softmax.

### Key Results (48B/3B-active MoE, 1.4T tokens)

| Benchmark | Baseline | AttnRes | Delta |
|-----------|----------|---------|-------|
| GPQA-Diamond | 36.9 | 44.4 | **+7.5** |
| Math | 53.5 | 57.1 | **+3.6** |
| HumanEval | 59.1 | 62.2 | +3.1 |
| MMLU | 73.5 | 74.6 | +1.1 |
| BBH | 76.3 | 78.0 | +1.7 |

Block AttnRes matches baseline trained with **1.25x more compute**. Overhead: <4% training, <2% inference.

### Architectural Implication

Standard residuals favor shallow, wide models. AttnRes shifts the optimum toward **narrower, deeper** models (optimal d_model/L ratio drops from ~60 to ~45).

---

## 3. How They Can Be Combined

### Complementary Nature

These two innovations operate on **orthogonal dimensions** and are naturally complementary:

| Dimension | Engram | AttnRes |
|-----------|--------|---------|
| **What it improves** | Information *source* (adds external memory) | Information *routing* (improves depth-wise flow) |
| **Where it acts** | Specific layers (2, 15) | Every layer boundary |
| **Mechanism** | O(1) hash lookup + gated fusion | Learned attention over depth |
| **What it replaces** | Part of MoE capacity | Fixed residual connections |
| **Parameter cost** | Large tables (offloadable to CPU) | Negligible (one vector per layer) |
| **Compute cost** | ~2-3% overhead | ~2-4% overhead |

### Combined Architecture: AttnRes + Engram

```
                    Token Embeddings
                          |
                    [Block 0 start]
                          |
            +---> Block AttnRes (before attn) <--- block summaries
            |             |
            |      Self-Attention
            |             |
            |      Block AttnRes (before MLP) <--- block summaries
            |             |
            |         MLP / MoE
            |             |
            |    [If Engram layer (2 or 15):]
            |      Engram Lookup + Gated Fusion
            |             |
            +--- accumulate into block sum ---+
                          |
                    [Block boundary?]
                    [Push block sum]
                          |
                       ... x L ...
                          |
                      Final Output
```

### Key Integration Points

1. **AttnRes provides better inputs to Engram gating.** The context-aware gating in Engram computes `alpha = sigmoid(h^T * memory)`. With AttnRes, h_l is a *selectively weighted* combination of prior layers rather than a uniform sum. This means:
   - The gating signal is more informative
   - Engram can activate more precisely on the right contexts

2. **Engram enriches the block representations that AttnRes attends over.** At Engram layers (2, 15), the layer output includes factual memory. When this gets folded into the block summary, all subsequent layers can attend back to memory-enriched representations via AttnRes.

3. **Depth-selective memory access.** Without AttnRes, Engram must be carefully placed at specific layers. With AttnRes, even if Engram is only at layer 2, deeper layers can directly attend back to layer 2's memory-enriched output (bypassing intermediate layers that may dilute it).

4. **Complementary scaling.** Engram scales by adding more hash table entries (memory capacity). AttnRes scales by enabling deeper models (compute depth). Together, they offer two independent scaling axes beyond width and MoE experts.

### Expected Synergies

- **Factual recall + reasoning chains:** Engram retrieves facts in O(1); AttnRes ensures those facts remain accessible across all depths for multi-step reasoning.
- **Reduced Engram layer count:** AttnRes may allow Engram at just 1 layer (instead of 2), since depth-wise attention can propagate memory information upward more effectively.
- **Better sparsity allocation:** With AttnRes reducing the need for width, more parameters can be allocated to Engram tables vs MoE experts.
- **Training stability:** AttnRes's zero-initialized queries provide a smooth warmup. Engram's gating also starts near-uniform. Both converge gracefully from identity-like initialization.

### Potential Challenges

1. **Gradient flow complexity:** Two novel gradient pathways (through hash tables and through depth attention) may require careful learning rate tuning.
2. **Memory budget:** Engram tables + block summaries for AttnRes both consume memory, though Engram can be offloaded to CPU.
3. **Optimal block boundaries:** Whether Engram layers should align with AttnRes block boundaries needs empirical study.

---

## Summary Table

| Feature | Engram | AttnRes | Combined |
|---------|--------|---------|----------|
| Innovation | O(1) memory lookup | Learned depth-wise attention | Both |
| Parameter overhead | Large (offloadable) | Negligible | Large (offloadable) |
| Compute overhead | ~2-3% | ~2-4% | ~5-7% (estimated) |
| Key benefit | Factual recall | Selective layer mixing | Facts + selective routing |
| Scaling axis | Memory capacity | Model depth | Both |
| Initialization | Hash tables (random) | Zero queries | Both |

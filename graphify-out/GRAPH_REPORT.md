# Graph Report - .  (2026-04-22)

## Corpus Check
- Corpus is ~39,893 words - fits in a single context window. You may not need a graph.

## Summary
- 221 nodes · 460 edges · 10 communities detected
- Extraction: 57% EXTRACTED · 43% INFERRED · 0% AMBIGUOUS · INFERRED: 196 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Engram Implementation Internals|Engram Implementation Internals]]
- [[_COMMUNITY_AttnRes Implementation Internals|AttnRes Implementation Internals]]
- [[_COMMUNITY_BlockAttnRes & Benchmark Harness|BlockAttnRes & Benchmark Harness]]
- [[_COMMUNITY_Cross-File Architecture Narrative|Cross-File Architecture Narrative]]
- [[_COMMUNITY_Engram Retrieval-Fusion Concepts|Engram Retrieval-Fusion Concepts]]
- [[_COMMUNITY_Combined Model Test Suite|Combined Model Test Suite]]
- [[_COMMUNITY_Paper Concepts & Synergy|Paper Concepts & Synergy]]
- [[_COMMUNITY_Benchmark Chart Findings|Benchmark Chart Findings]]
- [[_COMMUNITY_Engram Retrieve-Fuse Methods|Engram Retrieve-Fuse Methods]]
- [[_COMMUNITY_Benchmark Caveats Bridge|Benchmark Caveats Bridge]]

## God Nodes (most connected - your core abstractions)
1. `EngramModule` - 34 edges
2. `RMSNorm` - 32 edges
3. `BlockAttnRes` - 31 edges
4. `RMSNorm` - 30 edges
5. `AttnResTransformer` - 22 edges
6. `CombinedEngramAttnResTransformer` - 22 edges
7. `AttnResTransformerLayer` - 19 edges
8. `CombinedTransformerLayer` - 13 edges
9. `NgramHashMapping` - 13 edges
10. `ShortConv` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Engram layer placement (layers 2, 15)` --rationale_for--> `CombinedEngramAttnResTransformer`  [INFERRED]
  overview.md → codes/combined_model.py
- `test_block_attn_res_shapes()` --calls--> `BlockAttnRes`  [INFERRED]
  codes\test_attention_residuals.py → codes\attention_residuals.py
- `test_rms_norm()` --calls--> `RMSNorm`  [INFERRED]
  codes\test_attention_residuals.py → codes\engram.py
- `test_rms_norm()` --calls--> `RMSNorm`  [INFERRED]
  codes\test_engram.py → codes\engram.py
- `test_engram_module_shapes()` --calls--> `EngramModule`  [INFERRED]
  codes\test_engram.py → codes\engram.py

## Hyperedges (group relationships)
- **Three model variants trained side-by-side in benchmark** — attention_residuals_AttnResTransformer, benchmark_EngramOnlyTransformer, combined_model_CombinedEngramAttnResTransformer, benchmark_train_model [EXTRACTED 1.00]
- **Engram retrieval pipeline: compressor -> ngram hash -> multi-head embedding tables** — engram_TokenizerCompressor, engram_NgramHashMapping, engram_MultiHeadEmbedding, engram_retrieve [EXTRACTED 1.00]
- **Block AttnRes dataflow: two AttnRes per layer around attention and MLP with block-boundary state** — attention_residuals_BlockAttnRes, attention_residuals_AttnResTransformerLayer, attention_residuals_is_block_boundary, combined_model_CombinedTransformerLayer [EXTRACTED 1.00]

## Communities

### Community 0 - "Engram Implementation Internals"
Cohesion: 0.08
Nodes (35): MultiHeadEmbedding, _next_prime(), NgramHashMapping, Engram: Conditional Memory via Scalable Lookup Based on DeepSeek's paper (arXiv, Args:             compressed_ngrams: [batch, seq_len, ngram_size] compressed to, Multiple embedding tables for a single n-gram order.     Each head has its own, Args:             hash_indices: [batch, seq_len, num_heads]          Returns:, Depthwise convolution with SiLU activation and residual connection.     Equatio (+27 more)

### Community 1 - "AttnRes Implementation Internals"
Cohesion: 0.09
Nodes (32): AttnResTransformer, AttnResTransformerLayer, Attention Residuals (AttnRes) Based on the Kimi team's paper (arXiv: 2603.15031, Check if this layer starts a new block., Forward pass with Block Attention Residuals.          Args:             block, Root Mean Square Layer Normalization., Full Transformer with Block Attention Residuals.      Args:         vocab_siz, Args:             input_ids: [B, T] token IDs             labels: [B, T] targe (+24 more)

### Community 2 - "BlockAttnRes & Benchmark Harness"
Cohesion: 0.11
Nodes (21): BlockAttnRes, Block Attention Residuals module.      Instead of fixed h_l = h_{l-1} + f_{l-1, Compute attention-weighted combination over depth.          Args:, CharDataset, EngramOnlyTransformer, EngramOnlyTransformerLayer, Benchmark: Compare AttnRes-only, Engram-only (baseline transformer + Engram), a, CombinedTransformerLayer (+13 more)

### Community 3 - "Cross-File Architecture Narrative"
Cohesion: 0.12
Nodes (24): AttnResTransformer, AttnResTransformerLayer, BlockAttnRes, _is_block_boundary (layer_idx % (block_size//2)), CharDataset (byte-level), EngramOnlyTransformer (benchmark baseline), EngramOnlyTransformerLayer, SAMPLE_TEXT (repeated proverbs x500) (+16 more)

### Community 4 - "Engram Retrieval-Fusion Concepts"
Cohesion: 0.16
Nodes (19): RMSNorm (attention_residuals.py), layer_seed = 42 + layer_idx*100 convention for hash function diversity, RMSNorm duplicated across files for independent importability, Context-aware gating (sigmoid of query . key), Depthwise 1D convolution (Engram fusion step), Hyper-connections (M=4 branches), Multiplicative-XOR hash (Engram retrieval), Engram Retrieval + Fusion Architecture (+11 more)

### Community 5 - "Combined Model Test Suite"
Cohesion: 0.27
Nodes (10): CombinedEngramAttnResTransformer, Tests for the Combined Engram + AttnRes model. Run with: python test_combined.p, Verify Engram modules exist only at specified layers., Simulate a full training step., test_combined_backward(), test_combined_forward(), test_combined_with_loss(), test_engram_only_at_designated_layers() (+2 more)

### Community 6 - "Paper Concepts & Synergy"
Cohesion: 0.19
Nodes (13): Attention Residuals (arXiv:2603.15031), Engram: Conditional Memory via Scalable Lookup (arXiv:2601.07372), Sparsity allocation law (rho ~ 0.75 for MoE), Time-Depth Duality (AttnRes motivation), Two Axes of Sparsity (MoE vs Engram), Two-Phase Efficient Inference (parallel + sequential), Attention Residuals (learned depth attention), Combined Engram + AttnRes (+5 more)

### Community 7 - "Benchmark Chart Findings"
Cohesion: 0.23
Nodes (13): Training Step axis (0 to 500), Benchmark Chart: Training Loss & Final Performance, Final Performance Comparison (Avg Loss last 50 steps), Training Loss Curves (Loss vs Training Step 0-500), Insight: Engram and Combined variants converge faster and to ~half the final loss of AttnRes Only, Insight: Engram Only and Combined perform nearly identically (0.0373 vs 0.0376), Avg Loss over last 50 training steps, Training Loss (smoothed) (+5 more)

### Community 8 - "Engram Retrieve-Fuse Methods"
Cohesion: 0.25
Nodes (4): Build n-gram windows from compressed token IDs.          Args:             co, Phase 1: Retrieve embeddings from hash tables.          Args:             tok, Phase 2: Fuse retrieved memory into hidden states via gating + conv., Full Engram forward: retrieve + fuse.          Args:             hidden_state

### Community 9 - "Benchmark Caveats Bridge"
Cohesion: 1.0
Nodes (1): Benchmark limitations: repeated proverbs favor Engram memorization

## Knowledge Gaps
- **58 isolated node(s):** `Attention Residuals (AttnRes) Based on the Kimi team's paper (arXiv: 2603.15031`, `Root Mean Square Layer Normalization.`, `Block Attention Residuals module.      Instead of fixed h_l = h_{l-1} + f_{l-1`, `Compute attention-weighted combination over depth.          Args:`, `A single Transformer layer with Block Attention Residuals.      Each layer app` (+53 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Benchmark Caveats Bridge`** (2 nodes): `benchmark_chart.png`, `Benchmark limitations: repeated proverbs favor Engram memorization`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EngramModule` connect `BlockAttnRes & Benchmark Harness` to `Engram Implementation Internals`, `Engram Retrieve-Fuse Methods`, `Combined Model Test Suite`?**
  _High betweenness centrality (0.228) - this node is a cross-community bridge._
- **Why does `train_model()` connect `Cross-File Architecture Narrative` to `BlockAttnRes & Benchmark Harness`?**
  _High betweenness centrality (0.197) - this node is a cross-community bridge._
- **Why does `RMSNorm` connect `BlockAttnRes & Benchmark Harness` to `Engram Implementation Internals`, `AttnRes Implementation Internals`, `Combined Model Test Suite`?**
  _High betweenness centrality (0.183) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `EngramModule` (e.g. with `EngramOnlyTransformerLayer` and `EngramOnlyTransformer`) actually correct?**
  _`EngramModule` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 26 inferred relationships involving `RMSNorm` (e.g. with `EngramOnlyTransformerLayer` and `EngramOnlyTransformer`) actually correct?**
  _`RMSNorm` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `BlockAttnRes` (e.g. with `SwiGLU` and `CombinedTransformerLayer`) actually correct?**
  _`BlockAttnRes` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `RMSNorm` (e.g. with `EngramOnlyTransformerLayer` and `EngramOnlyTransformer`) actually correct?**
  _`RMSNorm` has 23 INFERRED edges - model-reasoned connections that need verification._
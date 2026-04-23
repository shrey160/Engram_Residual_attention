# Graph Report - .  (2026-04-22)

## Corpus Check
- 19 files · ~284,430 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 167 nodes · 371 edges · 20 communities detected
- Extraction: 50% EXTRACTED · 50% INFERRED · 0% AMBIGUOUS · INFERRED: 185 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Engram Module|Engram Module]]
- [[_COMMUNITY_Attention Residuals|Attention Residuals]]
- [[_COMMUNITY_Combined Model|Combined Model]]
- [[_COMMUNITY_Transformer Layers|Transformer Layers]]
- [[_COMMUNITY_Training & Data|Training & Data]]
- [[_COMMUNITY_Code Structure|Code Structure]]
- [[_COMMUNITY_Model Components|Model Components]]
- [[_COMMUNITY_Testing|Testing]]
- [[_COMMUNITY_Overview & Docs|Overview & Docs]]
- [[_COMMUNITY_Research Papers|Research Papers]]
- [[_COMMUNITY_Attention Mechanism|Attention Mechanism]]
- [[_COMMUNITY_Hash Retrieval|Hash Retrieval]]
- [[_COMMUNITY_Residual Connections|Residual Connections]]
- [[_COMMUNITY_MLP Layers|MLP Layers]]
- [[_COMMUNITY_RMSNorm|RMSNorm]]
- [[_COMMUNITY_Qwen3 Architecture|Qwen3 Architecture]]
- [[_COMMUNITY_MoE Experts|MoE Experts]]
- [[_COMMUNITY_Training Pipeline|Training Pipeline]]
- [[_COMMUNITY_Benchmarking|Benchmarking]]
- [[_COMMUNITY_Data Loading|Data Loading]]

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
- `test_block_attn_res_shapes()` --calls--> `BlockAttnRes`  [INFERRED]
  codes\test_attention_residuals.py → codes\attention_residuals.py
- `test_transformer_layer_shapes()` --calls--> `AttnResTransformerLayer`  [INFERRED]
  codes\test_attention_residuals.py → codes\attention_residuals.py
- `test_rms_norm()` --calls--> `RMSNorm`  [INFERRED]
  codes\test_attention_residuals.py → codes\engram.py
- `test_rms_norm()` --calls--> `RMSNorm`  [INFERRED]
  codes\test_engram.py → codes\engram.py
- `test_engram_module_shapes()` --calls--> `EngramModule`  [INFERRED]
  codes\test_engram.py → codes\engram.py

## Communities

### Community 0 - "Engram Module"
Cohesion: 0.14
Nodes (12): CombinedTransformerLayer, Combined Model: Engram + Attention Residuals Integrates DeepSeek's Engram (cond, Forward pass.          Args:             blocks: Completed block representati, SwiGLU feed-forward network., Transformer layer with both Block AttnRes and optional Engram injection., SwiGLU, EngramModule, Complete Engram module: Retrieval (hash lookup) + Fusion (gating + conv). (+4 more)

### Community 1 - "Attention Residuals"
Cohesion: 0.18
Nodes (15): AttnResTransformer, Full Transformer with Block Attention Residuals.      Args:         vocab_siz, Args:             input_ids: [B, T] token IDs             labels: [B, T] targe, Tests for the Attention Residuals module. Run with: python test_attention_resid, Ensure full backward pass works without errors., AttnRes should add negligible parameters., Verify token embedding and lm_head share weights., test_attn_res_params_are_tiny() (+7 more)

### Community 2 - "Combined Model"
Cohesion: 0.16
Nodes (5): CharDataset, EngramOnlyTransformer, EngramOnlyTransformerLayer, Benchmark: Compare AttnRes-only, Engram-only (baseline transformer + Engram), a, Dataset

### Community 3 - "Transformer Layers"
Cohesion: 0.24
Nodes (12): CombinedEngramAttnResTransformer, Full Transformer combining Engram (conditional memory) and Block AttnRes     (a, Print parameter breakdown., Tests for the Combined Engram + AttnRes model. Run with: python test_combined.p, Verify Engram modules exist only at specified layers., Simulate a full training step., test_combined_backward(), test_combined_forward() (+4 more)

### Community 4 - "Training & Data"
Cohesion: 0.25
Nodes (8): AttnResTransformerLayer, Check if this layer starts a new block., Forward pass with Block Attention Residuals.          Args:             block, A single Transformer layer with Block Attention Residuals.      Each layer app, Layer at a block boundary should append partial to blocks., Layer NOT at a block boundary should not change block count., test_transformer_layer_block_boundary(), test_transformer_layer_no_boundary()

### Community 5 - "Code Structure"
Cohesion: 0.27
Nodes (9): BlockAttnRes, Block Attention Residuals module.      Instead of fixed h_l = h_{l-1} + f_{l-1, Compute attention-weighted combination over depth.          Args:, At init, query=0 so attention should be uniform over all sources., Ensure gradients flow through AttnRes to all sources., With one source, output should equal that source (regardless of query)., test_block_attn_res_gradient_flow(), test_block_attn_res_single_source() (+1 more)

### Community 6 - "Model Components"
Cohesion: 0.31
Nodes (3): Root Mean Square Layer Normalization., RMSNorm, Args:             input_ids: [B, T] token IDs             labels: [B, T] optio

### Community 7 - "Testing"
Cohesion: 0.25
Nodes (6): Vocabulary projection that maps raw token IDs to compressed canonical IDs., Map raw token IDs to compressed IDs., TokenizerCompressor, Different token sequences should produce different memory vectors., test_engram_different_inputs_different_outputs(), test_tokenizer_compressor()

### Community 8 - "Overview & Docs"
Cohesion: 0.25
Nodes (7): Tests for the Engram module. Run with: python test_engram.py, Same tokens should always retrieve the same memory., Verify parameter count is reasonable., test_engram_module_shapes(), test_engram_param_count(), test_engram_retrieval_deterministic(), test_rms_norm()

### Community 9 - "Research Papers"
Cohesion: 0.25
Nodes (6): MultiHeadEmbedding, Multiple embedding tables for a single n-gram order.     Each head has its own, Args:             hash_indices: [batch, seq_len, num_heads]          Returns:, Ensure gradients flow through the Engram module., test_engram_gradient_flow(), test_multi_head_embedding()

### Community 10 - "Attention Mechanism"
Cohesion: 0.29
Nodes (8): Attention Residuals (Kimi, arXiv 2603.15031), BlockAttnRes, Codex (secondary reviewer), Engram (DeepSeek, arXiv 2601.07372), EngramModule from codes/, codes_moe/ directory, MoE plan, Qwen3 architecture

### Community 11 - "Hash Retrieval"
Cohesion: 0.33
Nodes (6): Depthwise convolution with SiLU activation and residual connection.     Equatio, Args:             x: [batch, seq_len, dim]         Returns:             out:, ShortConv, Output at position t should not depend on inputs at t+1., test_short_conv(), test_short_conv_causal()

### Community 12 - "Residual Connections"
Cohesion: 0.33
Nodes (6): NgramHashMapping, Args:             compressed_ngrams: [batch, seq_len, ngram_size] compressed to, Multiplicative-XOR hash function for n-gram lookup.     Maps n-gram tuples to e, Same input should always produce the same hash., test_ngram_hash_deterministic(), test_ngram_hash_mapping()

### Community 13 - "MLP Layers"
Cohesion: 0.47
Nodes (2): Root Mean Square Layer Normalization., RMSNorm

### Community 14 - "RMSNorm"
Cohesion: 0.33
Nodes (4): _next_prime(), Engram: Conditional Memory via Scalable Lookup Based on DeepSeek's paper (arXiv, Find the next prime number >= n., test_next_prime()

### Community 15 - "Qwen3 Architecture"
Cohesion: 0.67
Nodes (3): codes/ directory, codes_moe/ directory, graphify command

### Community 16 - "MoE Experts"
Cohesion: 0.67
Nodes (1): Asimov_the_foundation.pdf (training data)

### Community 17 - "Training Pipeline"
Cohesion: 1.0
Nodes (1): Attention Residuals (AttnRes) Based on the Kimi team's paper (arXiv: 2603.15031

### Community 18 - "Benchmarking"
Cohesion: 1.0
Nodes (1): Qwen3-style MoE model

### Community 19 - "Data Loading"
Cohesion: 1.0
Nodes (1): combined_model.py reference

## Knowledge Gaps
- **33 isolated node(s):** `Attention Residuals (AttnRes) Based on the Kimi team's paper (arXiv: 2603.15031`, `Root Mean Square Layer Normalization.`, `Block Attention Residuals module.      Instead of fixed h_l = h_{l-1} + f_{l-1`, `Compute attention-weighted combination over depth.          Args:`, `A single Transformer layer with Block Attention Residuals.      Each layer app` (+28 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Training Pipeline`** (2 nodes): `Attention Residuals (AttnRes) Based on the Kimi team's paper (arXiv: 2603.15031`, `attention_residuals.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Benchmarking`** (1 nodes): `Qwen3-style MoE model`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Data Loading`** (1 nodes): `combined_model.py reference`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EngramModule` connect `Engram Module` to `Combined Model`, `Transformer Layers`, `Model Components`, `Testing`, `Overview & Docs`, `Research Papers`, `Hash Retrieval`, `Residual Connections`, `MLP Layers`, `RMSNorm`?**
  _High betweenness centrality (0.263) - this node is a cross-community bridge._
- **Why does `RMSNorm` connect `MLP Layers` to `Engram Module`, `Attention Residuals`, `Combined Model`, `Transformer Layers`, `Model Components`, `Testing`, `Overview & Docs`, `Research Papers`, `Hash Retrieval`, `Residual Connections`, `RMSNorm`?**
  _High betweenness centrality (0.202) - this node is a cross-community bridge._
- **Why does `BlockAttnRes` connect `Code Structure` to `Engram Module`, `Attention Residuals`, `Transformer Layers`, `Training & Data`, `Model Components`, `Training Pipeline`?**
  _High betweenness centrality (0.143) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `EngramModule` (e.g. with `EngramOnlyTransformerLayer` and `EngramOnlyTransformer`) actually correct?**
  _`EngramModule` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 26 inferred relationships involving `RMSNorm` (e.g. with `EngramOnlyTransformerLayer` and `EngramOnlyTransformer`) actually correct?**
  _`RMSNorm` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `BlockAttnRes` (e.g. with `SwiGLU` and `CombinedTransformerLayer`) actually correct?**
  _`BlockAttnRes` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `RMSNorm` (e.g. with `EngramOnlyTransformerLayer` and `EngramOnlyTransformer`) actually correct?**
  _`RMSNorm` has 23 INFERRED edges - model-reasoned connections that need verification._
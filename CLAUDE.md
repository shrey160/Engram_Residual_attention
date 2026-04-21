# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

PyTorch reference implementation and benchmark harness for two 2026 transformer-architecture papers and their composition:

- **Engram** (DeepSeek, arXiv 2601.07372) — O(1) hash-based memory lookup as a sparsity axis alongside MoE.
- **Attention Residuals** (Kimi / Moonshot AI, arXiv 2603.15031) — softmax attention over depth replacing fixed unit-weight residual connections.
- **Combined** — Engram injection at designated layers + Block AttnRes everywhere.

This is a *demonstration* repo (laptop-scale, ~1.5M params, 500 steps, byte-level). The benchmark intentionally uses repeated proverbs as data, which heavily favors Engram's hash memorization — see the "Limitations & Caveats" section of `README.md` before drawing conclusions from `benchmark_chart.png`. Don't quote the loss numbers as evidence one method beats another; quote the papers' billion-scale results instead.

## In-Progress Work — `codes_moe/` (Qwen3-Style MoE)

A second model is being built in `codes_moe/`: a ~0.5B-parameter MoE language model that combines the **Qwen3 architecture** (GQA + RoPE base 1M + RMSNorm + QK-Norm + SwiGLU experts), the **Qwen3-Next gated attention** (sigmoid gate after SDPA), and the **Engram + Block AttnRes** techniques from `codes/`. Trained from scratch on `test_data/Asimov_the_foundation.pdf`.

**Status: plan-only.** The directory currently contains `PLAN.md` and nothing else. Read `codes_moe/PLAN.md` end-to-end before adding any implementation file — it is the source of truth for hyperparameters, file layout, the 12-step implementation order, the parameter budget arithmetic, and the §8 honesty caveats (data starvation, MoE collapse risk on ~500K tokens). When implementing:

- Imports cross sibling directories: `codes_moe/combined_moe_model.py` reuses `EngramModule` and `BlockAttnRes` from `codes/` via `sys.path.insert(0, '../codes')`. Do NOT duplicate those modules.
- The existing `codes/combined_model.py` is the structural reference for layer wiring (Engram between attention and MoE; AttnRes twice per layer). Don't edit it as part of MoE work.
- "Qwen 3.6" mentioned by the user resolves to the Qwen3 family (May 2025 tech report) plus Qwen3-Next gated attention (NeurIPS 2025 oral). PLAN.md §0 documents this assumption.
- Codex (`AGENTS.md`) is the secondary reviewer; keep style consistent with `codes/` per AGENTS.md conventions.

**Training data:** `test_data/Asimov_the_foundation.pdf` (~2 MB, ~500K tokens after BPE). Note this is ~20,000× under-trained for a 500M model — the build is a structural correctness exercise, not a capability claim.

**Outputs that will land here when implemented:** `codes_moe/cache/` (extracted text + tokenizer), `codes_moe/checkpoints/`, `codes_moe/losses.jsonl`, `codes_moe/benchmark_moe_chart.png`. Add to `.gitignore` as appropriate when they appear.

## Answering Questions With `graphify-out/`

The `graphify-out/` directory contains a precomputed knowledge graph of this repo (built by `/graphify`). Before grepping or reading source files for an architecture/concept question, check whether the graph already encodes it — it's faster than re-deriving structure, and it crosses the doc/code/paper boundary.

**Files:**

| File | Purpose |
|------|---------|
| `graph.json` | The full graph: 221 nodes, 460 edges, 10 communities. Loadable with `networkx.readwrite.json_graph.node_link_graph(data, edges='links')`. |
| `GRAPH_REPORT.md` | Human-readable summary: god nodes, surprising connections, per-community node lists, suggested questions. Read this first for orientation. |
| `cache/` | Per-file extraction cache. Don't touch — used by `/graphify --update`. |
| `manifest.json` | File hashes for incremental rebuilds. |
| `cost.json` | Cumulative token tracker. |

**When the graph is the right tool:**

- Cross-cutting questions ("what bridges X and Y?", "why is Z so connected?", "what depends on this concept across docs and code?")
- Conceptual questions where the answer lives in multiple files (paper → overview → implementation → test)
- "What community does X belong to?" / "What are the central abstractions?"

**When to skip the graph and read source directly:**

- Implementation details of a single function (just open the file)
- Anything time-sensitive or recently changed (graph may be stale relative to current `git status`)
- Exact line numbers, signatures, or behavior — the graph stores labels and relations, not code bodies

### Query process

1. **Read `GRAPH_REPORT.md` first.** Look at God Nodes (highest-degree), Communities, and Suggested Questions. The user's question may already match one.

2. **Pick start nodes by label match.** For a question like "how does X connect to Y?", find the nodes whose `label` fields equal or contain `X` and `Y`. Note that the same concept can appear as multiple nodes — one from the AST extractor (e.g. `EngramModule` from `codes/engram.py`) and one from the semantic extractor (the conceptual node). Use both.

3. **Choose traversal:**
   - **BFS depth 2** for "what's connected to X?" (broad neighborhood)
   - **DFS depth ≤6** for "how does X reach Y?" (specific chain)
   - **Shortest path** (`networkx.shortest_path`) for "is there any link between X and Y?"

4. **Group neighbors by `community` attribute** to see which clusters X bridges. Read each edge's `relation` (`calls`, `implements`, `references`, `rationale_for`, `semantically_similar_to`, etc.) and `confidence` (`EXTRACTED` is structural truth from AST or explicit citation; `INFERRED` is the semantic agent's guess; `AMBIGUOUS` is a flagged uncertainty). Quote `EXTRACTED` edges with confidence; mark `INFERRED` claims as inferences.

5. **Cite `source_file` from each node.** Every node carries the file it came from — use it as the citation when answering.

### Reusable query script

This is the script pattern that produced the `EngramModule` cross-community trace. Copy it into a temp `.py` file (do not commit), substitute the start-node label, and run with `python`. The `sys.stdout` wrapper is required on Windows because the default cp1252 codec chokes on the `↔` character that appears in some node labels.

```python
import json, sys, io
from networkx.readwrite import json_graph
from collections import defaultdict
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

G = json_graph.node_link_graph(
    json.loads(Path('graphify-out/graph.json').read_text(encoding='utf-8')),
    edges='links',
)

# 1. Find start node(s) by label
TARGET_LABEL = 'EngramModule'  # exact match, case-insensitive
start_nodes = [
    nid for nid, d in G.nodes(data=True)
    if d.get('label', '').lower() == TARGET_LABEL.lower()
]

# 2. BFS depth 2
subgraph = set(start_nodes)
frontier = set(start_nodes)
for _ in range(2):
    next_frontier = {nbr for n in frontier for nbr in G.neighbors(n) if nbr not in subgraph}
    subgraph |= next_frontier
    frontier = next_frontier

# 3. Group by community
by_comm = defaultdict(list)
for n in subgraph:
    by_comm[G.nodes[n].get('community')].append(n)
for c in sorted(by_comm, key=lambda x: (x is None, x)):
    print(f'\n--- Community {c}: {len(by_comm[c])} nodes ---')
    for nid in by_comm[c][:12]:
        d = G.nodes[nid]
        print(f'  {d.get("label", nid)}  [{d.get("source_file", "")}]')

# 4. Direct edges out of start node (shows relation + confidence + target community)
print('\n=== Direct edges ===')
for src in start_nodes:
    for nbr in G.neighbors(src):
        ed = G.get_edge_data(src, nbr) or {}
        print(f'  {G.nodes[src]["label"]} --{ed.get("relation","?")}'
              f'[{ed.get("confidence","?")}]--> {G.nodes[nbr].get("label", nbr)}'
              f'  (comm {G.nodes[nbr].get("community", "?")})')
```

For a **shortest path** between two concepts:

```python
import networkx as nx
src = next(n for n, d in G.nodes(data=True) if d.get('label','').lower() == 'engrammodule')
tgt = next(n for n, d in G.nodes(data=True) if 'engram paper' in d.get('label','').lower())
for nid in nx.shortest_path(G, src, tgt):
    print(G.nodes[nid].get('label'))
```

**Refresh discipline:** if you've made non-trivial code edits in a session, the graph reflects pre-edit state. Either flag this in your answer ("the graph predates today's changes to `combined_model.py`") or rebuild with `/graphify . --update` before relying on it.

## Commands

All commands run from the `codes/` directory.

```bash
cd codes/

# Test suites (pure-Python, no pytest dependency — each file has its own runner)
python test_engram.py                  # 13 tests
python test_attention_residuals.py     # 13 tests
python test_combined.py                # 6 tests

# Quick smoke check of any module via its __main__ block
python engram.py
python attention_residuals.py
python combined_model.py

# Full benchmark (trains all 3 models on CPU, ~10 min, writes
# benchmark_losses.json and benchmark_chart.png)
python benchmark.py

# Interactive walkthrough
jupyter notebook train_notebook.ipynb
```

Requirements: Python 3.8+, `torch>=2.0`, `matplotlib` (only for chart generation in `benchmark.py`).

There is no `pytest`, no linter config, no CI. Each test file is a script with a `main()` that runs the suite and prints PASS/FAIL — to run a single test, edit the file's `main()` or call the function directly via `python -c "from test_engram import test_xxx; test_xxx()"`.

## Architecture

### Three model variants live side-by-side

| File | Class | Role |
|------|-------|------|
| `attention_residuals.py` | `AttnResTransformer` | Pure AttnRes baseline (no Engram) |
| `benchmark.py` | `EngramOnlyTransformer` | Plain transformer + Engram (no AttnRes) — **only defined inside `benchmark.py`**, not its own module |
| `combined_model.py` | `CombinedEngramAttnResTransformer` | Both techniques integrated |

When changing the layer/forward signature, all three need to stay in sync — the benchmark instantiates and trains them with identical hyperparameters for comparison.

### Block AttnRes data flow (the non-obvious part)

Standard transformers carry a single hidden state `x` through layers. AttnRes carries **two** pieces of state:

- `blocks: List[Tensor]` — completed block representations. `blocks[0]` is the token+pos embedding (the "b_0" of the paper).
- `partial_block: Tensor` — the running intra-block sum being accumulated.

Each transformer layer (`AttnResTransformerLayer.forward` / `CombinedTransformerLayer.forward`):
1. If `_is_block_boundary()`: append current `partial_block` to `blocks`, reset `partial_block` to zero.
2. Compute `h = attn_res(blocks, partial_block)` — softmax-attention over depth — and feed `h` into self-attention.
3. Add attention output back into `partial_block` (intra-block addition is the "residual" inside a block).
4. Compute `h = mlp_res(blocks, partial_block)` — second AttnRes — feed into MLP.
5. Add MLP output back into `partial_block`.

So each layer applies AttnRes twice (once before attention, once before MLP) and the layer signature is `(blocks, partial_block) -> (blocks, partial_block)` — *not* `x -> x`. After all layers, a `final_attn_res` produces the final hidden state from `blocks + [partial_block]`.

`block_size` counts **sub-layers** (attention and MLP each count as one). So `block_size=4` means 2 transformer layers per block. `_is_block_boundary` uses `layers_per_block = block_size // 2`.

`BlockAttnRes.query` is **zero-initialized** — this is load-bearing. Zero query → uniform softmax → recovers standard residual averaging at training start. Don't change the init.

### Engram data flow

`EngramModule.forward(hidden_states, token_ids)` returns an additive update; the caller adds it residually. It needs the raw `token_ids` (not the hidden state) because retrieval is `compress(token_ids) → ngram → hash → embedding-table-lookup`. This means **`token_ids` must thread through every transformer layer in models that use Engram**, even layers without an Engram module — see how `CombinedEngramAttnResTransformer.forward` passes `token_ids=input_ids` to every layer call.

Engram is only added at layers in `engram_layers` (default `{1, num_layers // 2}`). In the combined model, Engram is injected **between attention and MLP**, into `partial_block` (so its output flows through the second AttnRes and into MLP).

The N-gram hash is multiplicative-XOR mod a per-head prime. Hash multipliers are seeded from `layer_seed = 42 + layer_idx * 100` so different layers get different hash functions. When adding new Engram layers, keep this seed convention to avoid collision-correlated layers.

### Parameter accounting

`CombinedEngramAttnResTransformer.param_summary()` classifies parameters by name substring — anything containing `"engram"` is counted as Engram, anything containing `"attn_res"` or `"mlp_res"` as AttnRes, the rest as backbone. If you rename modules, update this method too or the breakdown silently drifts.

### Causal masks

Both `AttnResTransformerLayer` and `CombinedTransformerLayer` build a causal mask via `nn.Transformer.generate_square_subsequent_mask` inside `forward` if none is passed, and call attention with `is_causal=True`. Engram's `ShortConv` uses asymmetric padding (`(kernel_size - 1) * dilation` on the left, then trims to original length) to stay causal — preserve this if you change the conv.

### RMSNorm duplication

`engram.py` and `attention_residuals.py` each define their own `RMSNorm`. They're identical in behavior. `combined_model.py` and `benchmark.py` both import and alias them (`EngramRMSNorm` vs `RMSNorm`). Don't try to "fix" this by deleting one — keeping the modules independently importable is intentional.

## Other notes

- The benchmark uses raw byte tokenization (vocab=256). The Engram tokenizer compressor is therefore reduced to a random projection — its real value is at 32K+ subword vocab.
- `papers/` contains the source PDFs and is in `.gitignore`-adjacent territory but currently tracked. `overview.md` (root) and `codes/overview.md` are detailed paper notes; consult them before changing fusion/gating math.
- `benchmark_losses.json` and `benchmark_chart.png` are committed artifacts — `benchmark.py` regenerates them. Don't hand-edit.

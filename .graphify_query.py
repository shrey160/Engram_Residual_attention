import json, sys, io
from networkx.readwrite import json_graph
import networkx as nx
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

data = json.loads(Path('graphify-out/graph.json').read_text(encoding='utf-8'))
G = json_graph.node_link_graph(data, edges='links')

question = 'Why does EngramModule connect BlockAttnRes Benchmark Harness to Engram Implementation Internals Engram Retrieve-Fuse Methods Combined Model Test Suite'
mode = 'bfs'
terms = [t.lower() for t in question.split() if len(t) > 3]

# Find the EngramModule node specifically
start_nodes = []
for nid, ndata in G.nodes(data=True):
    label = ndata.get('label', '').lower()
    if label == 'engrammodule':
        start_nodes.append(nid)

if not start_nodes:
    # fallback: term scoring
    scored = []
    for nid, ndata in G.nodes(data=True):
        label = ndata.get('label', '').lower()
        score = sum(1 for t in terms if t in label)
        if score > 0:
            scored.append((score, nid))
    scored.sort(reverse=True)
    start_nodes = [nid for _, nid in scored[:3]]

print(f'Start nodes: {[G.nodes[n].get("label", n) for n in start_nodes]}')
print(f'Communities of start nodes: {[G.nodes[n].get("community") for n in start_nodes]}')
print()

# BFS depth 2 from EngramModule
subgraph_nodes = set(start_nodes)
subgraph_edges = []
frontier = set(start_nodes)

for depth in range(2):
    next_frontier = set()
    for n in frontier:
        for neighbor in G.neighbors(n):
            if neighbor not in subgraph_nodes:
                next_frontier.add(neighbor)
                subgraph_edges.append((n, neighbor, depth + 1))
        # Also include incoming edges
        for pred in G.predecessors(n) if G.is_directed() else []:
            if pred not in subgraph_nodes:
                next_frontier.add(pred)
                subgraph_edges.append((pred, n, depth + 1))
    subgraph_nodes.update(next_frontier)
    frontier = next_frontier

# Group neighbors by community
from collections import defaultdict
by_comm = defaultdict(list)
for n in subgraph_nodes:
    c = G.nodes[n].get('community')
    by_comm[c].append(n)

print(f'BFS reached {len(subgraph_nodes)} nodes across {len(by_comm)} communities')
print()
for c in sorted(by_comm.keys(), key=lambda x: (x is None, x)):
    nodes = by_comm[c]
    print(f'--- Community {c}: {len(nodes)} nodes ---')
    for nid in nodes[:12]:
        d = G.nodes[nid]
        print(f'  {d.get("label", nid)}  [src={d.get("source_file","")}]')
    if len(nodes) > 12:
        print(f'  ... +{len(nodes)-12} more')
    print()

# Show all edges from start node directly
print('=== Direct edges from EngramModule (depth 1) ===')
for src in start_nodes:
    for nbr in G.neighbors(src):
        ed = G.edges[src, nbr] if not G.is_directed() else G.get_edge_data(src, nbr, {})
        rel = ed.get('relation', '?')
        conf = ed.get('confidence', '?')
        nlbl = G.nodes[nbr].get('label', nbr)
        ncomm = G.nodes[nbr].get('community', '?')
        print(f'  EngramModule --{rel}[{conf}]--> {nlbl}  (comm {ncomm})')

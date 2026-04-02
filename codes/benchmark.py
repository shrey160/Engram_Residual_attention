"""
Benchmark: Compare AttnRes-only, Engram-only (baseline transformer + Engram),
and Combined (Engram + AttnRes) on a character-level language modeling task.
Produces a results table and training loss chart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import math
import json
import os

from engram import EngramModule, RMSNorm as EngramRMSNorm
from attention_residuals import AttnResTransformer, RMSNorm
from combined_model import CombinedEngramAttnResTransformer


# ---------------------------------------------------------------------------
# Simple baseline transformer with Engram (no AttnRes)
# ---------------------------------------------------------------------------
class EngramOnlyTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, layer_idx, has_engram=False,
                 engram_config=None, dropout=0.0):
        super().__init__()
        self.has_engram = has_engram
        self.attn_norm = RMSNorm(hidden_dim)
        self.mlp_norm = RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)
        nn.init.normal_(self.w_down.weight, std=0.02 / math.sqrt(2 * (layer_idx + 1)))

        if has_engram and engram_config:
            self.engram = EngramModule(
                vocab_size=engram_config.get("vocab_size", 256),
                compressed_vocab_size=engram_config.get("compressed_vocab_size", 200),
                hidden_dim=hidden_dim,
                engram_dim=engram_config.get("engram_dim", 16),
                ngram_range=engram_config.get("ngram_range", (2, 3)),
                num_heads=engram_config.get("num_heads", 4),
                table_size_hint=engram_config.get("table_size_hint", 2003),
                layer_seed=42 + layer_idx * 100,
            )
            self.engram_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, token_ids=None, attn_mask=None):
        h = self.attn_norm(x)
        if attn_mask is None:
            T = h.size(1)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device, dtype=h.dtype)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, is_causal=True)
        x = x + attn_out

        if self.has_engram and token_ids is not None:
            engram_out = self.engram(x, token_ids)
            x = x + self.engram_scale * engram_out

        h = self.mlp_norm(x)
        gate = F.silu(self.w_gate(h))
        x = x + self.w_down(gate * self.w_up(h))
        return x


class EngramOnlyTransformer(nn.Module):
    def __init__(self, vocab_size=256, hidden_dim=128, num_heads=4, num_layers=6,
                 ffn_dim=256, engram_layers=None, engram_config=None, max_seq_len=128):
        super().__init__()
        if engram_layers is None:
            engram_layers = {1, 3}
        if engram_config is None:
            engram_config = {"vocab_size": vocab_size}
        else:
            engram_config.setdefault("vocab_size", vocab_size)

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.layers = nn.ModuleList([
            EngramOnlyTransformerLayer(
                hidden_dim, num_heads, ffn_dim, i,
                has_engram=(i in engram_layers), engram_config=engram_config,
            ) for i in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x, token_ids=input_ids)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return logits, loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CharDataset(Dataset):
    def __init__(self, text, seq_len=128):
        self.seq_len = seq_len
        self.data = torch.tensor([b for b in text.encode("utf-8", errors="replace")], dtype=torch.long)
        self.num_samples = max(0, len(self.data) - seq_len - 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, dataloader, num_steps=500, lr=3e-4, device="cpu"):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    step = 0
    epoch = 0
    start = time.time()

    while step < num_steps:
        for input_ids, labels in dataloader:
            if step >= num_steps:
                break
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits, loss = model(input_ids, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            step += 1
        epoch += 1

    elapsed = time.time() - start
    return losses, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SAMPLE_TEXT = (
        "The quick brown fox jumps over the lazy dog. A stitch in time saves nine. "
        "To be or not to be, that is the question. All that glitters is not gold. "
        "Actions speak louder than words. Knowledge is power. Time is money. "
        "The early bird catches the worm. Practice makes perfect. "
        "Every cloud has a silver lining. Fortune favors the bold. "
        "Where there is a will, there is a way. Rome was not built in a day. "
    ) * 500

    SEQ_LEN = 128
    BATCH_SIZE = 32
    NUM_STEPS = 500
    LR = 3e-4
    HIDDEN = 128
    HEADS = 4
    LAYERS = 6
    FFN = 256
    VOCAB = 256

    ENGRAM_CFG = {
        "vocab_size": VOCAB,
        "compressed_vocab_size": 200,
        "engram_dim": 16,
        "num_heads": 4,
        "table_size_hint": 2003,
    }

    dataset = CharDataset(SAMPLE_TEXT, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = "cpu"

    # --- Build models ---
    print("=" * 70)
    print("BENCHMARK: AttnRes vs Engram-Only vs Combined (Engram + AttnRes)")
    print("=" * 70)
    print(f"Config: hidden={HIDDEN}, heads={HEADS}, layers={LAYERS}, ffn={FFN}")
    print(f"Training: {NUM_STEPS} steps, batch={BATCH_SIZE}, seq_len={SEQ_LEN}, lr={LR}")
    print(f"Device: {device}")
    print()

    models = {}

    models["AttnRes Only"] = AttnResTransformer(
        vocab_size=VOCAB, hidden_dim=HIDDEN, num_heads=HEADS,
        num_layers=LAYERS, ffn_dim=FFN, block_size=4, max_seq_len=SEQ_LEN,
    )

    models["Engram Only"] = EngramOnlyTransformer(
        vocab_size=VOCAB, hidden_dim=HIDDEN, num_heads=HEADS,
        num_layers=LAYERS, ffn_dim=FFN, engram_layers={1, 3},
        engram_config=ENGRAM_CFG, max_seq_len=SEQ_LEN,
    )

    models["Combined"] = CombinedEngramAttnResTransformer(
        vocab_size=VOCAB, hidden_dim=HIDDEN, num_heads=HEADS,
        num_layers=LAYERS, ffn_dim=FFN, block_size=4,
        engram_layers={1, 3}, engram_config=ENGRAM_CFG, max_seq_len=SEQ_LEN,
    )

    # --- Train all models ---
    results = {}
    all_losses = {}

    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Training '{name}' ({n_params:,} params)...")
        losses, elapsed = train_model(model, dataloader, NUM_STEPS, LR, device)
        results[name] = {
            "params": n_params,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "avg_last_50": sum(losses[-50:]) / 50,
            "time_sec": round(elapsed, 1),
            "steps_per_sec": round(NUM_STEPS / elapsed, 1),
        }
        all_losses[name] = losses
        print(f"  Done in {elapsed:.1f}s | Final loss: {losses[-1]:.4f}")
        print()

    # --- Results Table ---
    print()
    print("=" * 90)
    print("RESULTS TABLE")
    print("=" * 90)
    header = f"{'Model':<20} {'Params':>10} {'Init Loss':>10} {'Final Loss':>11} {'Min Loss':>10} {'Avg Last50':>11} {'Time(s)':>8} {'Steps/s':>8}"
    print(header)
    print("-" * 90)
    for name, r in results.items():
        row = (f"{name:<20} {r['params']:>10,} {r['initial_loss']:>10.4f} "
               f"{r['final_loss']:>11.4f} {r['min_loss']:>10.4f} "
               f"{r['avg_last_50']:>11.4f} {r['time_sec']:>8.1f} {r['steps_per_sec']:>8.1f}")
        print(row)
    print("=" * 90)

    # Best model
    best = min(results.items(), key=lambda x: x[1]["avg_last_50"])
    print(f"\nBest model by avg last-50 loss: {best[0]} ({best[1]['avg_last_50']:.4f})")

    # --- Save losses for chart ---
    json_path = os.path.join(os.path.dirname(__file__), "benchmark_losses.json")
    with open(json_path, "w") as f:
        json.dump(all_losses, f)
    print(f"\nLosses saved to {json_path}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def smooth(vals, window=15):
            out = []
            for i in range(len(vals)):
                s = max(0, i - window)
                out.append(sum(vals[s:i+1]) / (i - s + 1))
            return out

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        colors = {"AttnRes Only": "#2196F3", "Engram Only": "#FF9800", "Combined": "#4CAF50"}

        # Left: smoothed training loss
        for name, losses in all_losses.items():
            ax1.plot(smooth(losses), label=name, color=colors[name], linewidth=2)
        ax1.set_xlabel("Training Step", fontsize=12)
        ax1.set_ylabel("Loss (smoothed)", fontsize=12)
        ax1.set_title("Training Loss Curves", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Right: bar chart of final metrics
        names = list(results.keys())
        final_losses = [results[n]["avg_last_50"] for n in names]
        bars = ax2.bar(names, final_losses, color=[colors[n] for n in names], width=0.5, edgecolor="black")
        for bar, val in zip(bars, final_losses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Avg Loss (last 50 steps)", fontsize=12)
        ax2.set_title("Final Performance Comparison", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        chart_path = os.path.join(os.path.dirname(__file__), "benchmark_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {chart_path}")
        plt.close()

    except ImportError:
        print("matplotlib not installed — skipping chart generation.")
        print("Install with: pip install matplotlib")

"""
Combined Model: Engram + Attention Residuals
Integrates DeepSeek's Engram (conditional memory) with Kimi's Block AttnRes
(learned depth-wise attention) into a single Transformer architecture.

Key insight: Engram injects factual knowledge at specific layers via O(1) lookup.
AttnRes allows all subsequent layers to selectively attend back to those enriched
representations instead of having them diluted by fixed-weight residuals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Set

from engram import EngramModule, RMSNorm as EngramRMSNorm
from attention_residuals import BlockAttnRes, RMSNorm


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class CombinedTransformerLayer(nn.Module):
    """
    Transformer layer with both Block AttnRes and optional Engram injection.

    Architecture per layer:
        1. Block AttnRes (before attention)
        2. Self-Attention
        3. [Optional] Engram injection (if this is an Engram layer)
        4. Block AttnRes (before MLP)
        5. MLP

    Args:
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        ffn_dim: FFN intermediate dimension
        block_size: Block size for AttnRes
        layer_idx: Layer index (0-based)
        has_engram: Whether this layer has an Engram module
        engram_config: Config dict for Engram module (if has_engram=True)
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        block_size: int = 6,
        layer_idx: int = 0,
        has_engram: bool = False,
        engram_config: Optional[dict] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.has_engram = has_engram

        # Block Attention Residuals (two per layer)
        self.attn_res = BlockAttnRes(hidden_dim)
        self.mlp_res = BlockAttnRes(hidden_dim)

        # Norms
        self.attn_norm = RMSNorm(hidden_dim)
        self.mlp_norm = RMSNorm(hidden_dim)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # MLP
        self.mlp = SwiGLU(hidden_dim, ffn_dim, dropout)

        # Small init for MLP output projection (stable training)
        nn.init.normal_(self.mlp.w_down.weight, std=0.02 / math.sqrt(2 * (layer_idx + 1)))

        # Engram module (only at designated layers)
        if has_engram and engram_config is not None:
            self.engram = EngramModule(
                vocab_size=engram_config.get("vocab_size", 32000),
                compressed_vocab_size=engram_config.get("compressed_vocab_size", 25000),
                hidden_dim=hidden_dim,
                engram_dim=engram_config.get("engram_dim", 64),
                ngram_range=engram_config.get("ngram_range", (2, 3)),
                num_heads=engram_config.get("num_heads", 8),
                table_size_hint=engram_config.get("table_size_hint", 10007),
                kernel_size=engram_config.get("kernel_size", 4),
                layer_seed=42 + layer_idx * 100,
            )
            # Learnable scale for Engram contribution
            self.engram_scale = nn.Parameter(torch.tensor(0.1))

    def _is_block_boundary(self) -> bool:
        layers_per_block = self.block_size // 2
        return self.layer_idx % layers_per_block == 0 and self.layer_idx > 0

    def forward(
        self,
        blocks: List[torch.Tensor],
        partial_block: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            blocks: Completed block representations
            partial_block: Current intra-block partial sum
            token_ids: Raw token IDs (needed for Engram layers)
            attn_mask: Optional attention mask

        Returns:
            blocks, partial_block: Updated state
        """
        # Block boundary check
        if self._is_block_boundary():
            blocks.append(partial_block)
            partial_block = torch.zeros_like(partial_block)

        # --- AttnRes before self-attention ---
        h = self.attn_res(blocks, partial_block)

        # Self-attention (generate causal mask if none provided)
        h_normed = self.attn_norm(h)
        if attn_mask is None:
            T = h_normed.size(1)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                T, device=h_normed.device, dtype=h_normed.dtype
            )
        attn_out, _ = self.attn(
            h_normed, h_normed, h_normed,
            attn_mask=attn_mask,
            is_causal=True,
        )
        partial_block = partial_block + attn_out

        # --- Engram injection (after attention, before MLP) ---
        if self.has_engram and token_ids is not None:
            engram_out = self.engram(partial_block, token_ids)
            partial_block = partial_block + self.engram_scale * engram_out

        # --- AttnRes before MLP ---
        h = self.mlp_res(blocks, partial_block)

        # MLP
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block


class CombinedEngramAttnResTransformer(nn.Module):
    """
    Full Transformer combining Engram (conditional memory) and Block AttnRes
    (attention over depth).

    Engram is placed at specific layers (e.g., layers 1 and 7 in a 12-layer model)
    for factual knowledge injection. AttnRes operates at every layer boundary,
    allowing selective access to Engram-enriched representations across depth.

    Args:
        vocab_size: Vocabulary size
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        ffn_dim: FFN intermediate dimension
        block_size: Block size for AttnRes (number of sub-layers)
        engram_layers: Set of layer indices where Engram is applied
        engram_config: Configuration for Engram modules
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        ffn_dim: int = 1024,
        block_size: int = 6,
        engram_layers: Optional[Set[int]] = None,
        engram_config: Optional[dict] = None,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        if engram_layers is None:
            # Default: early + mid-depth (like the paper's layers 2 and 15)
            engram_layers = {1, num_layers // 2}
        if engram_config is None:
            engram_config = {"vocab_size": vocab_size}
        else:
            engram_config.setdefault("vocab_size", vocab_size)

        self.engram_layers = engram_layers

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        # Scale embeddings for stable init
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.layers = nn.ModuleList([
            CombinedTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                block_size=block_size,
                layer_idx=i,
                has_engram=(i in engram_layers),
                engram_config=engram_config,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        self.final_attn_res = BlockAttnRes(hidden_dim)
        self.final_norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [B, T] token IDs
            labels: [B, T] optional target IDs for loss

        Returns:
            logits: [B, T, V]
            loss: Optional cross-entropy loss
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        h = self.token_emb(input_ids) + self.pos_emb(positions)

        # AttnRes state
        blocks = [h]  # b_0 = token embedding
        partial_block = torch.zeros_like(h)

        for layer in self.layers:
            blocks, partial_block = layer(
                blocks, partial_block, token_ids=input_ids
            )

        output = self.final_attn_res(blocks, partial_block)
        output = self.final_norm(output)
        logits = self.lm_head(output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def param_summary(self) -> dict:
        """Print parameter breakdown."""
        total = sum(p.numel() for p in self.parameters())
        engram_params = sum(
            p.numel() for name, p in self.named_parameters() if "engram" in name
        )
        attn_res_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "attn_res" in name or "mlp_res" in name
        )
        backbone_params = total - engram_params - attn_res_params

        return {
            "total": total,
            "backbone": backbone_params,
            "engram": engram_params,
            "attn_res": attn_res_params,
            "engram_pct": 100 * engram_params / total,
            "attn_res_pct": 100 * attn_res_params / total,
        }


if __name__ == "__main__":
    B, T = 2, 32
    vocab_size = 32000

    model = CombinedEngramAttnResTransformer(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_heads=4,
        num_layers=12,
        ffn_dim=512,
        block_size=6,
        engram_layers={1, 6},
        engram_config={
            "vocab_size": vocab_size,
            "compressed_vocab_size": 25000,
            "engram_dim": 32,
            "num_heads": 4,
            "table_size_hint": 10007,
        },
        max_seq_len=T,
    )

    input_ids = torch.randint(0, vocab_size, (B, T))
    labels = torch.randint(0, vocab_size, (B, T))

    logits, loss = model(input_ids, labels)
    print(f"Logits: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    summary = model.param_summary()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:,}")

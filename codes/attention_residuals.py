"""
Attention Residuals (AttnRes)
Based on the Kimi team's paper (arXiv: 2603.15031)

A PyTorch implementation of Block Attention Residuals that replace fixed-weight
residual connections with learned attention over depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


class BlockAttnRes(nn.Module):
    """
    Block Attention Residuals module.

    Instead of fixed h_l = h_{l-1} + f_{l-1}(h_{l-1}), this computes:
        h_l = sum_i alpha_{i->l} * v_i
    where alpha are softmax attention weights over completed block representations
    and the current partial block sum.

    Args:
        hidden_dim: Model hidden dimension
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Pseudo-query: learned vector (NOT input-dependent)
        # Critical: initialized to zero for uniform attention at start
        self.query = nn.Parameter(torch.zeros(hidden_dim))
        self.norm = RMSNorm(hidden_dim)

    def forward(
        self,
        blocks: List[torch.Tensor],
        partial_block: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention-weighted combination over depth.

        Args:
            blocks: List of completed block representations, each [B, T, D].
                    blocks[0] is always the token embedding.
            partial_block: [B, T, D] current intra-block partial sum

        Returns:
            h: [B, T, D] attention-weighted hidden state for current layer
        """
        # Stack all sources: completed blocks + current partial block
        sources = blocks + [partial_block]
        V = torch.stack(sources, dim=0)  # [N+1, B, T, D]

        # Normalize keys
        K = self.norm(V)  # [N+1, B, T, D]

        # Compute attention logits: q^T . RMSNorm(k_i)
        # query: [D] -> broadcast against [N+1, B, T, D]
        logits = torch.einsum("d, n b t d -> n b t", self.query, K)  # [N+1, B, T]

        # Softmax over depth dimension (dim=0)
        attn_weights = logits.softmax(dim=0)  # [N+1, B, T]

        # Weighted sum
        h = torch.einsum("n b t, n b t d -> b t d", attn_weights, V)  # [B, T, D]

        return h


class AttnResTransformerLayer(nn.Module):
    """
    A single Transformer layer with Block Attention Residuals.

    Each layer applies AttnRes twice:
    1. Before self-attention
    2. Before MLP

    Args:
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network intermediate dimension
        block_size: Number of sub-layers per block (attn + MLP = 2 per transformer layer)
        layer_idx: Index of this layer (0-based)
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        block_size: int = 6,
        layer_idx: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_size = block_size

        # Two AttnRes modules: one before attention, one before MLP
        self.attn_res = BlockAttnRes(hidden_dim)
        self.mlp_res = BlockAttnRes(hidden_dim)

        # Standard Transformer components
        self.attn_norm = RMSNorm(hidden_dim)
        self.mlp_norm = RMSNorm(hidden_dim)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # MLP (SwiGLU-style)
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.mlp_dropout = nn.Dropout(dropout)

        # Small init for output projections (stable training)
        nn.init.normal_(self.w_down.weight, std=0.02 / math.sqrt(2 * (layer_idx + 1)))

    def _is_block_boundary(self) -> bool:
        """Check if this layer starts a new block."""
        # block_size counts sub-layers (attn + MLP); each transformer layer has 2
        layers_per_block = self.block_size // 2
        return self.layer_idx % layers_per_block == 0 and self.layer_idx > 0

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU MLP."""
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.mlp_dropout(self.w_down(gate * up))

    def forward(
        self,
        blocks: List[torch.Tensor],
        partial_block: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass with Block Attention Residuals.

        Args:
            blocks: List of completed block representations [B, T, D]
            partial_block: [B, T, D] running intra-block sum
            attn_mask: Optional causal attention mask

        Returns:
            blocks: Updated list of block representations
            partial_block: Updated intra-block partial sum
        """
        # Check if we need to start a new block
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

        # --- AttnRes before MLP ---
        h = self.mlp_res(blocks, partial_block)

        # MLP
        mlp_out = self._mlp(self.mlp_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block


class AttnResTransformer(nn.Module):
    """
    Full Transformer with Block Attention Residuals.

    Args:
        vocab_size: Vocabulary size
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        ffn_dim: FFN intermediate dimension
        block_size: Number of sub-layers per block (default 6 = 3 transformer layers)
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
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        # Scale embeddings for stable init
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.layers = nn.ModuleList([
            AttnResTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                block_size=block_size,
                layer_idx=i,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        self.final_attn_res = BlockAttnRes(hidden_dim)
        self.final_norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [B, T] token IDs
            labels: [B, T] target token IDs for loss computation

        Returns:
            logits: [B, T, V]
            loss: Optional cross-entropy loss
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        # Token + positional embedding
        h = self.token_emb(input_ids) + self.pos_emb(positions)

        # Initialize blocks with token embedding (b_0)
        blocks = [h]
        partial_block = torch.zeros_like(h)

        # Forward through layers
        for layer in self.layers:
            blocks, partial_block = layer(blocks, partial_block)

        # Final AttnRes to produce output from all block representations
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


if __name__ == "__main__":
    B, T = 2, 32
    vocab_size = 32000

    model = AttnResTransformer(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_heads=4,
        num_layers=6,
        ffn_dim=512,
        block_size=4,  # 2 layers per block
        max_seq_len=T,
    )

    input_ids = torch.randint(0, vocab_size, (B, T))
    labels = torch.randint(0, vocab_size, (B, T))

    logits, loss = model(input_ids, labels)
    print(f"Input: {input_ids.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

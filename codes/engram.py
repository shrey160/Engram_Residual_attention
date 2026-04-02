"""
Engram: Conditional Memory via Scalable Lookup
Based on DeepSeek's paper (arXiv: 2601.07372)

A PyTorch implementation of the Engram module that provides O(1) hash-based
memory lookup as a complement to MoE conditional computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


def _next_prime(n: int) -> int:
    """Find the next prime number >= n."""
    if n <= 2:
        return 2
    candidate = n if n % 2 != 0 else n + 1
    while True:
        is_prime = True
        for i in range(3, int(math.sqrt(candidate)) + 1, 2):
            if candidate % i == 0:
                is_prime = False
                break
        if is_prime:
            return candidate
        candidate += 2


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


class TokenizerCompressor(nn.Module):
    """
    Vocabulary projection that maps raw token IDs to compressed canonical IDs.
    In the paper, this uses NFKC normalization, lowercasing, etc.
    Here we use a learnable embedding-based compression for flexibility.
    """

    def __init__(self, vocab_size: int, compressed_vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.compressed_vocab_size = compressed_vocab_size
        # Deterministic projection: map each token to a compressed ID
        # In practice, this is built from tokenizer metadata
        self.register_buffer(
            "projection",
            torch.randint(0, compressed_vocab_size, (vocab_size,))
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map raw token IDs to compressed IDs."""
        return self.projection[token_ids.clamp(0, self.vocab_size - 1)]


class NgramHashMapping(nn.Module):
    """
    Multiplicative-XOR hash function for n-gram lookup.
    Maps n-gram tuples to embedding table indices.
    """

    def __init__(
        self,
        ngram_size: int,
        num_heads: int,
        table_size_hint: int,
        seed: int = 42,
    ):
        super().__init__()
        self.ngram_size = ngram_size
        self.num_heads = num_heads

        # Each head gets a unique prime table size
        table_sizes = []
        current = table_size_hint
        for _ in range(num_heads):
            p = _next_prime(current)
            table_sizes.append(p)
            current = p + 2
        self.register_buffer("table_sizes", torch.tensor(table_sizes, dtype=torch.long))

        # Multipliers for the hash function (one per n-gram position per head)
        rng = torch.Generator().manual_seed(seed)
        multipliers = torch.randint(
            1, 2**31 - 1, (num_heads, ngram_size), generator=rng, dtype=torch.long
        )
        self.register_buffer("multipliers", multipliers)

    def forward(self, compressed_ngrams: torch.Tensor) -> torch.Tensor:
        """
        Args:
            compressed_ngrams: [batch, seq_len, ngram_size] compressed token IDs

        Returns:
            hash_indices: [batch, seq_len, num_heads] indices into embedding tables
        """
        B, L, N = compressed_ngrams.shape
        # [B, L, 1, N] * [1, 1, H, N] -> multiplicative hash
        ngrams = compressed_ngrams.unsqueeze(2).long()  # [B, L, 1, N]
        mults = self.multipliers.unsqueeze(0).unsqueeze(0)  # [1, 1, H, N]

        # Multiplicative-XOR hash
        products = ngrams * mults  # [B, L, H, N]
        # XOR-fold across n-gram positions
        mix = products[..., 0]
        for i in range(1, N):
            mix = mix ^ products[..., i]

        # Modulo by per-head prime table size
        table_sizes = self.table_sizes.view(1, 1, -1)  # [1, 1, H]
        hash_indices = mix.abs() % table_sizes

        return hash_indices  # [B, L, H]


class MultiHeadEmbedding(nn.Module):
    """
    Multiple embedding tables for a single n-gram order.
    Each head has its own table with a unique prime size.
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        table_sizes: torch.Tensor,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # Create separate embedding tables per head
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(table_sizes[h].item()), embed_dim)
            for h in range(num_heads)
        ])

        # Initialize with small values
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, hash_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hash_indices: [batch, seq_len, num_heads]

        Returns:
            embeddings: [batch, seq_len, num_heads * embed_dim]
        """
        parts = []
        for h, emb in enumerate(self.embeddings):
            parts.append(emb(hash_indices[..., h]))  # [B, L, D]
        return torch.cat(parts, dim=-1)  # [B, L, H*D]


class ShortConv(nn.Module):
    """
    Depthwise convolution with SiLU activation and residual connection.
    Equation 5 in the paper.
    """

    def __init__(self, dim: int, kernel_size: int = 4, dilation: int = 3):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,  # causal padding
            groups=dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            out: [batch, seq_len, dim]
        """
        residual = x
        h = self.norm(x)
        h = h.transpose(1, 2)  # [B, D, L]
        h = self.conv(h)
        # Causal: trim future positions
        h = h[..., :x.size(1)]
        h = h.transpose(1, 2)  # [B, L, D]
        h = F.silu(h)
        return h + residual


class EngramModule(nn.Module):
    """
    Complete Engram module: Retrieval (hash lookup) + Fusion (gating + conv).

    Args:
        vocab_size: Raw vocabulary size
        compressed_vocab_size: Compressed vocabulary size (after projection)
        hidden_dim: Transformer hidden dimension
        engram_dim: Internal engram embedding dimension per head
        ngram_range: Tuple of (min_n, max_n) for n-gram orders, e.g. (2, 3)
        num_heads: Number of hash heads per n-gram order
        table_size_hint: Base size for hash tables
        kernel_size: Convolution kernel size
        num_branches: Number of hyper-connection branches (M)
        layer_seed: Seed for layer-specific hash multipliers
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        compressed_vocab_size: int = 25000,
        hidden_dim: int = 512,
        engram_dim: int = 64,
        ngram_range: Tuple[int, int] = (2, 3),
        num_heads: int = 8,
        table_size_hint: int = 100003,
        kernel_size: int = 4,
        num_branches: int = 1,
        layer_seed: int = 42,
    ):
        super().__init__()
        self.ngram_range = ngram_range
        self.num_heads = num_heads
        self.num_branches = num_branches
        self.hidden_dim = hidden_dim
        self.max_ngram = ngram_range[1]

        # Tokenizer compression
        self.compressor = TokenizerCompressor(vocab_size, compressed_vocab_size)

        # Hash mappings and embedding tables for each n-gram order
        self.hash_mappings = nn.ModuleDict()
        self.embed_tables = nn.ModuleDict()

        total_retrieval_dim = 0
        for n in range(ngram_range[0], ngram_range[1] + 1):
            key = f"ngram_{n}"
            mapping = NgramHashMapping(
                ngram_size=n,
                num_heads=num_heads,
                table_size_hint=table_size_hint,
                seed=layer_seed + n * 1000,
            )
            self.hash_mappings[key] = mapping

            embed = MultiHeadEmbedding(
                num_heads=num_heads,
                embed_dim=engram_dim,
                table_sizes=mapping.table_sizes,
            )
            self.embed_tables[key] = embed
            total_retrieval_dim += num_heads * engram_dim

        self.total_retrieval_dim = total_retrieval_dim

        # Fusion: context-aware gating
        self.W_K = nn.Linear(total_retrieval_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(total_retrieval_dim, hidden_dim, bias=False)
        self.query_norm = RMSNorm(hidden_dim)
        self.key_norm = RMSNorm(hidden_dim)

        # Depthwise convolution
        self.short_conv = ShortConv(
            dim=hidden_dim,
            kernel_size=kernel_size,
            dilation=self.max_ngram,
        )

        # Per-branch gating (if using hyper-connections)
        if num_branches > 1:
            self.branch_query_norms = nn.ModuleList([
                RMSNorm(hidden_dim) for _ in range(num_branches)
            ])
            self.branch_key_projs = nn.ModuleList([
                nn.Linear(total_retrieval_dim, hidden_dim, bias=False)
                for _ in range(num_branches)
            ])

        self.scale = 1.0 / math.sqrt(hidden_dim)

    def _build_ngrams(
        self, compressed_ids: torch.Tensor, n: int
    ) -> torch.Tensor:
        """
        Build n-gram windows from compressed token IDs.

        Args:
            compressed_ids: [batch, seq_len]
            n: n-gram order

        Returns:
            ngrams: [batch, seq_len, n] with zero-padding for early positions
        """
        B, L = compressed_ids.shape
        # Pad the beginning with zeros for positions where full n-gram isn't available
        padded = F.pad(compressed_ids, (n - 1, 0), value=0)  # [B, L + n - 1]
        # Unfold to create n-gram windows
        ngrams = padded.unfold(1, n, 1)  # [B, L, n]
        return ngrams

    def retrieve(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Phase 1: Retrieve embeddings from hash tables.

        Args:
            token_ids: [batch, seq_len] raw token IDs

        Returns:
            memory: [batch, seq_len, total_retrieval_dim]
        """
        compressed = self.compressor(token_ids)

        all_embeds = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            key = f"ngram_{n}"
            ngrams = self._build_ngrams(compressed, n)  # [B, L, n]
            hash_idx = self.hash_mappings[key](ngrams)  # [B, L, H]
            embeds = self.embed_tables[key](hash_idx)  # [B, L, H*D]
            all_embeds.append(embeds)

        return torch.cat(all_embeds, dim=-1)  # [B, L, total_retrieval_dim]

    def fuse(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        branch_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Phase 2: Fuse retrieved memory into hidden states via gating + conv.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            memory: [batch, seq_len, total_retrieval_dim]
            branch_idx: If using multi-branch, which branch to gate for

        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        # Key and Value projections from memory
        k = self.W_K(memory)  # [B, L, D]
        v = self.W_V(memory)  # [B, L, D]

        # Context-aware gating
        if branch_idx is not None and self.num_branches > 1:
            q_norm = self.branch_query_norms[branch_idx](hidden_states)
            k_norm = self.key_norm(self.branch_key_projs[branch_idx](memory))
        else:
            q_norm = self.query_norm(hidden_states)
            k_norm = self.key_norm(k)

        # Gating score: dot product -> sigmoid
        gate = torch.sigmoid(
            (q_norm * k_norm).sum(dim=-1, keepdim=True) * self.scale
        )  # [B, L, 1]

        # Gated value
        v_gated = gate * v  # [B, L, D]

        # Depthwise convolution with residual
        output = self.short_conv(v_gated)  # [B, L, D]

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        branch_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Full Engram forward: retrieve + fuse.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            token_ids: [batch, seq_len] raw token IDs
            branch_idx: Optional branch index for multi-branch gating

        Returns:
            output: [batch, seq_len, hidden_dim] (additive to hidden states)
        """
        memory = self.retrieve(token_ids)
        return self.fuse(hidden_states, memory, branch_idx)


if __name__ == "__main__":
    # Quick sanity check
    B, L, D = 2, 16, 512
    vocab_size = 32000

    model = EngramModule(
        vocab_size=vocab_size,
        compressed_vocab_size=25000,
        hidden_dim=D,
        engram_dim=64,
        ngram_range=(2, 3),
        num_heads=8,
        table_size_hint=10007,
    )

    token_ids = torch.randint(0, vocab_size, (B, L))
    hidden = torch.randn(B, L, D)

    output = model(hidden, token_ids)
    print(f"Input hidden: {hidden.shape}")
    print(f"Engram output: {output.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

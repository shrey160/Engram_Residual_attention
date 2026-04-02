"""
Tests for the Engram module.
Run with: python test_engram.py
"""

import torch
import torch.nn as nn
from engram import (
    EngramModule,
    TokenizerCompressor,
    NgramHashMapping,
    MultiHeadEmbedding,
    ShortConv,
    RMSNorm,
    _next_prime,
)


def test_next_prime():
    assert _next_prime(2) == 2
    assert _next_prime(4) == 5
    assert _next_prime(10) == 11
    assert _next_prime(100) == 101
    assert _next_prime(10007) == 10007
    print("[PASS] test_next_prime")


def test_rms_norm():
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64)
    out = norm(x)
    assert out.shape == x.shape
    # RMSNorm should roughly normalize the RMS to ~1
    rms = out.float().pow(2).mean(-1).sqrt()
    assert rms.mean().item() < 2.0, f"RMS too large: {rms.mean().item()}"
    print("[PASS] test_rms_norm")


def test_tokenizer_compressor():
    comp = TokenizerCompressor(vocab_size=1000, compressed_vocab_size=500)
    ids = torch.randint(0, 1000, (2, 16))
    compressed = comp(ids)
    assert compressed.shape == ids.shape
    assert compressed.max() < 500
    assert compressed.min() >= 0
    print("[PASS] test_tokenizer_compressor")


def test_ngram_hash_mapping():
    mapping = NgramHashMapping(ngram_size=2, num_heads=4, table_size_hint=1009)
    ngrams = torch.randint(0, 500, (2, 16, 2))
    indices = mapping(ngrams)
    assert indices.shape == (2, 16, 4), f"Expected (2,16,4), got {indices.shape}"
    # All indices should be within table sizes
    for h in range(4):
        assert indices[:, :, h].max() < mapping.table_sizes[h]
    print("[PASS] test_ngram_hash_mapping")


def test_ngram_hash_deterministic():
    """Same input should always produce the same hash."""
    mapping = NgramHashMapping(ngram_size=3, num_heads=8, table_size_hint=10007)
    ngrams = torch.randint(0, 500, (1, 4, 3))
    idx1 = mapping(ngrams)
    idx2 = mapping(ngrams)
    assert torch.equal(idx1, idx2), "Hash should be deterministic"
    print("[PASS] test_ngram_hash_deterministic")


def test_multi_head_embedding():
    table_sizes = torch.tensor([1009, 1013, 1019, 1021])
    emb = MultiHeadEmbedding(num_heads=4, embed_dim=32, table_sizes=table_sizes)
    indices = torch.randint(0, 1009, (2, 16, 4))
    # Clamp per head
    for h in range(4):
        indices[:, :, h] = indices[:, :, h] % table_sizes[h]
    out = emb(indices)
    assert out.shape == (2, 16, 4 * 32), f"Expected (2,16,128), got {out.shape}"
    print("[PASS] test_multi_head_embedding")


def test_short_conv():
    conv = ShortConv(dim=64, kernel_size=4, dilation=3)
    x = torch.randn(2, 16, 64)
    out = conv(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("[PASS] test_short_conv")


def test_short_conv_causal():
    """Output at position t should not depend on inputs at t+1."""
    conv = ShortConv(dim=32, kernel_size=4, dilation=3)
    x = torch.randn(1, 20, 32, requires_grad=False)

    # Get output at position 10
    out_full = conv(x)
    val_at_10 = out_full[0, 10].clone()

    # Modify input at position 15 (future)
    x2 = x.clone()
    x2[0, 15:] = torch.randn_like(x2[0, 15:])
    out_modified = conv(x2)
    val_at_10_modified = out_modified[0, 10]

    assert torch.allclose(val_at_10, val_at_10_modified, atol=1e-5), \
        "ShortConv is not causal!"
    print("[PASS] test_short_conv_causal")


def test_engram_module_shapes():
    B, L, D = 2, 16, 256
    vocab_size = 5000

    engram = EngramModule(
        vocab_size=vocab_size,
        compressed_vocab_size=4000,
        hidden_dim=D,
        engram_dim=32,
        ngram_range=(2, 3),
        num_heads=4,
        table_size_hint=1009,
    )

    token_ids = torch.randint(0, vocab_size, (B, L))
    hidden = torch.randn(B, L, D)

    output = engram(hidden, token_ids)
    assert output.shape == (B, L, D), f"Expected ({B},{L},{D}), got {output.shape}"
    print("[PASS] test_engram_module_shapes")


def test_engram_retrieval_deterministic():
    """Same tokens should always retrieve the same memory."""
    engram = EngramModule(
        vocab_size=5000, compressed_vocab_size=4000,
        hidden_dim=128, engram_dim=16, num_heads=4, table_size_hint=1009,
    )
    token_ids = torch.randint(0, 5000, (1, 8))

    mem1 = engram.retrieve(token_ids)
    mem2 = engram.retrieve(token_ids)
    assert torch.equal(mem1, mem2), "Retrieval should be deterministic"
    print("[PASS] test_engram_retrieval_deterministic")


def test_engram_gradient_flow():
    """Ensure gradients flow through the Engram module."""
    engram = EngramModule(
        vocab_size=5000, compressed_vocab_size=4000,
        hidden_dim=128, engram_dim=16, num_heads=4, table_size_hint=1009,
    )

    token_ids = torch.randint(0, 5000, (1, 8))
    hidden = torch.randn(1, 8, 128, requires_grad=True)

    output = engram(hidden, token_ids)
    loss = output.sum()
    loss.backward()

    # Check that embedding tables have gradients
    has_grad = False
    for name, param in engram.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients flowing through Engram"

    # Check hidden state gradient
    assert hidden.grad is not None, "No gradient on hidden states"
    print("[PASS] test_engram_gradient_flow")


def test_engram_different_inputs_different_outputs():
    """Different token sequences should produce different memory vectors."""
    engram = EngramModule(
        vocab_size=5000, compressed_vocab_size=4000,
        hidden_dim=128, engram_dim=16, num_heads=4, table_size_hint=1009,
    )

    ids1 = torch.tensor([[100, 200, 300, 400]])
    ids2 = torch.tensor([[500, 600, 700, 800]])

    mem1 = engram.retrieve(ids1)
    mem2 = engram.retrieve(ids2)

    assert not torch.equal(mem1, mem2), "Different inputs should give different memories"
    print("[PASS] test_engram_different_inputs_different_outputs")


def test_engram_param_count():
    """Verify parameter count is reasonable."""
    engram = EngramModule(
        vocab_size=32000, compressed_vocab_size=25000,
        hidden_dim=512, engram_dim=64, num_heads=8, table_size_hint=10007,
    )
    total = sum(p.numel() for p in engram.parameters())
    # Should have substantial params in embedding tables
    assert total > 1_000_000, f"Too few params: {total}"
    print(f"[PASS] test_engram_param_count (total: {total:,})")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Engram Tests")
    print("=" * 60)

    test_next_prime()
    test_rms_norm()
    test_tokenizer_compressor()
    test_ngram_hash_mapping()
    test_ngram_hash_deterministic()
    test_multi_head_embedding()
    test_short_conv()
    test_short_conv_causal()
    test_engram_module_shapes()
    test_engram_retrieval_deterministic()
    test_engram_gradient_flow()
    test_engram_different_inputs_different_outputs()
    test_engram_param_count()

    print("=" * 60)
    print("All Engram tests passed!")
    print("=" * 60)
